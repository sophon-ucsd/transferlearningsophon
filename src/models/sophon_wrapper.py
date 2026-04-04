"""Sophon pretrained ParticleTransformer wrapper for transfer learning.

Supports strategies: frozen, partial_ft, full_ft, from_scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from weaver.nn.model.ParticleTransformer import ParticleTransformer


# Sophon's exact architecture (from example_ParticleTransformer_sophon.py get_model())
SOPHON_KWARGS = dict(
    input_dim=17,
    pair_input_dim=4,
    use_pre_activation_pair=True,
    embed_dims=[128, 512, 128],
    pair_embed_dims=[64, 64, 64],
    num_heads=8,
    num_layers=8,
    num_cls_layers=2,
    block_params=None,
    cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
    fc_params=[],
    activation="gelu",
    trim=True,
    for_inference=False,
)

EMBED_DIM = 128  # last element of embed_dims


def _detect_num_classes_from_state(state: dict[str, torch.Tensor]) -> int | None:
    """Detect num_classes from the FC layer weight shape in a state dict."""
    for key in ("mod.fc.0.weight", "fc.0.weight"):
        if key in state:
            return state[key].shape[0]
    return None


def _load_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return state dict with 'mod.' prefix removed if present, so it can be
    loaded directly into a bare ParticleTransformer."""
    # Check if keys start with 'mod.' (from ParticleTransformerSophonWrapper)
    sample_key = next(iter(state))
    if sample_key.startswith("mod."):
        return {k[len("mod."):]: v for k, v in state.items()}
    return state


class SophonTransferModel(nn.Module):
    """Wraps Sophon pretrained ParticleTransformer for transfer learning.

    Strategies:
      - frozen: backbone frozen, only MLP head trains
      - partial_ft: first N transformer blocks frozen, rest + head train
      - full_ft: everything trains (backbone gets separate lower LR)
      - from_scratch: random init, everything trains
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        num_classes: int = 10,
        head_type: str = "mlp",
        export_embed: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.export_embed = export_embed
        self.embed_dim = EMBED_DIM

        # --- Determine pretrained num_classes and load checkpoint ---
        checkpoint_state: dict[str, torch.Tensor] | None = None
        pretrained_nc: int = num_classes  # default if no checkpoint

        if checkpoint_path is not None:
            raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            checkpoint_state = raw["model_state_dict"] if "model_state_dict" in raw else raw
            detected = _detect_num_classes_from_state(checkpoint_state)
            if detected is not None:
                pretrained_nc = detected
                print(f"Detected pretrained num_classes={pretrained_nc} from checkpoint")

        # --- Create ParticleTransformer with pretrained num_classes ---
        self.mod = ParticleTransformer(num_classes=pretrained_nc, **SOPHON_KWARGS)

        # --- Load pretrained weights ---
        if checkpoint_state is not None:
            clean_state = _load_state_dict(checkpoint_state)
            missing, unexpected = self.mod.load_state_dict(clean_state, strict=False)
            loaded = len(clean_state) - len(unexpected)
            print(
                f"Loaded pretrained weights from {checkpoint_path} "
                f"({loaded} loaded, {len(missing)} missing, {len(unexpected)} unexpected)"
            )

        # --- Check for split forward API ---
        self._has_split_forward = hasattr(self.mod, "_forward_encoder")

        # --- Replace classification head ---
        self._replace_head(num_classes, head_type)

    def _replace_head(self, num_classes: int, head_type: str) -> None:
        """Replace the ParT FC layer with a new head."""
        if head_type == "linear":
            self.mod.fc = nn.Sequential(nn.Linear(self.embed_dim, num_classes))
        elif head_type == "mlp":
            self.mod.fc = nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type!r}")

    # ---- forward -----------------------------------------------------------

    def _forward_compat(
        self,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fallback encoder path with the .view() attn_mask fix."""
        mod = self.mod
        with torch.amp.autocast("cuda", enabled=mod.use_amp):
            padding_mask = ~mask.squeeze(1) if mask is not None else None

            if lorentz_vectors is not None and mod.pair_embed is not None:
                attn_mask = mod.pair_embed(lorentz_vectors).view(
                    -1, lorentz_vectors.size(-1), lorentz_vectors.size(-1)
                )
            else:
                attn_mask = None

            x = mod.embed(features)
            if mask is not None:
                x = x.masked_fill(~mask.permute(2, 0, 1), 0)

            for block in mod.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            cls_tokens = mod.cls_token.expand(-1, x.size(1), -1)
            for block in mod.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = cls_tokens.squeeze(0)

            if hasattr(mod, "norm") and mod.norm is not None:
                x_cls = mod.norm(x_cls)

        return x_cls

    def forward(
        self,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features: (B, 17, N) Sophon input features.
            lorentz_vectors: (B, 4, N) scaled px, py, pz, E.
            mask: (B, 1, N) bool, True where particle exists.

        Returns:
            (logits, embeddings) if export_embed, else logits.
            logits: (B, num_classes), embeddings: (B, 128).
        """
        if self._has_split_forward:
            x, padding_mask = self.mod._forward_encoder(
                features, v=lorentz_vectors, mask=mask
            )
            with torch.amp.autocast("cuda", enabled=self.mod.use_amp):
                x_cls = self.mod._forward_aggregator(x, padding_mask)
        else:
            x_cls = self._forward_compat(features, lorentz_vectors, mask)

        logits = self.mod.fc(x_cls)

        if self.export_embed:
            return logits, x_cls
        return logits

    # ---- freeze / unfreeze -------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone params. Only self.mod.fc (the head) remains trainable."""
        for name, param in self.mod.named_parameters():
            if name.startswith("fc."):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.mod.parameters():
            param.requires_grad = True

    def freeze_first_n_layers(self, n: int) -> None:
        """Freeze embed, pair_embed, and first n encoder blocks.

        Trainable: blocks[n:], cls_blocks, cls_token, norm, fc.
        """
        # Freeze embedding layers
        for param in self.mod.embed.parameters():
            param.requires_grad = False
        if self.mod.pair_embed is not None:
            for param in self.mod.pair_embed.parameters():
                param.requires_grad = False

        # Freeze first n encoder blocks, unfreeze the rest
        for i, block in enumerate(self.mod.blocks):
            requires_grad = i >= n
            for param in block.parameters():
                param.requires_grad = requires_grad

        # Keep cls_token, cls_blocks, norm, fc trainable
        self.mod.cls_token.requires_grad = True
        for block in self.mod.cls_blocks:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(self.mod, "norm") and self.mod.norm is not None:
            for param in self.mod.norm.parameters():
                param.requires_grad = True
        for param in self.mod.fc.parameters():
            param.requires_grad = True

    # ---- param groups for discriminative LR --------------------------------

    def get_param_groups(
        self, backbone_lr: float, head_lr: float, weight_decay: float = 0.01
    ) -> list[dict]:
        """Return param groups with different LRs for backbone vs head."""
        backbone_params = []
        head_params = []
        for name, param in self.mod.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("fc."):
                head_params.append(param)
            else:
                backbone_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        return groups

    # ---- utilities ---------------------------------------------------------

    @staticmethod
    def count_params(model: nn.Module) -> tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable


def create_model(
    strategy: str,
    checkpoint_path: str | None,
    num_classes: int = 10,
    head_type: str = "mlp",
    frozen_layers: int = 4,
) -> SophonTransferModel:
    """Factory: create a SophonTransferModel configured for the given strategy.

    Args:
        strategy: 'frozen', 'partial_ft', 'full_ft', or 'from_scratch'.
        checkpoint_path: path to Sophon .pt file. None for from_scratch.
        num_classes: output classes for the new head.
        head_type: 'mlp' or 'linear'.
        frozen_layers: how many encoder blocks to freeze (for partial_ft).
    """
    if strategy == "from_scratch":
        model = SophonTransferModel(
            checkpoint_path=None, num_classes=num_classes, head_type=head_type
        )
        print(f"Strategy: from_scratch — random init, all params trainable")
        return model

    if checkpoint_path is None:
        raise ValueError(f"Strategy {strategy!r} requires a checkpoint_path")

    model = SophonTransferModel(
        checkpoint_path=checkpoint_path, num_classes=num_classes, head_type=head_type
    )

    if strategy == "frozen":
        model.freeze_backbone()
        print("Strategy: frozen — backbone frozen, training head only")
    elif strategy == "partial_ft":
        model.freeze_first_n_layers(frozen_layers)
        print(f"Strategy: partial_ft — first {frozen_layers} encoder blocks frozen")
    elif strategy == "full_ft":
        model.unfreeze_backbone()
        print("Strategy: full_ft — all params trainable")
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return model
