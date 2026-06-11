"""Input channel adapter for optional anatomical-prior channels.

When extra input channels are supplied (e.g. 4 MRI modalities + 3 tissue-prior
maps = 7), :class:`ChannelAdapter` maps them back to the model's native input
channel count with a 1x1x1 convolution placed *before* the first encoder layer.
The rest of the network is unchanged.

Checkpoint compatibility
------------------------
A model wrapped with :class:`AdaptedModel` adds exactly the parameters
``adapter.proj.weight`` and ``adapter.proj.bias``. Loading a checkpoint trained
without the adapter therefore requires ``strict=False``; the only missing keys
are those two. :func:`load_with_adapter_report` performs the load and logs the
new/missing keys at INFO level (never silently).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ChannelAdapter(nn.Module):
    """1x1x1 conv mapping ``in_channels`` inputs to ``native_channels``.

    When ``in_channels == native_channels`` it is initialised to the identity so
    a wrapped model is a bit-exact no-op by default.
    """

    def __init__(
        self,
        in_channels: int,
        native_channels: int,
        bias: bool = True,
        identity_map: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.native_channels = int(native_channels)
        self.proj = nn.Conv3d(in_channels, native_channels, kernel_size=1, bias=bias)
        if identity_map is not None:
            with torch.no_grad():
                self.proj.weight.zero_()
                for src, dst in identity_map:
                    src_i = int(src)
                    dst_i = int(dst)
                    if 0 <= src_i < in_channels and 0 <= dst_i < native_channels:
                        self.proj.weight[dst_i, src_i, 0, 0, 0] = 1.0
                if bias and self.proj.bias is not None:
                    self.proj.bias.zero_()
        elif in_channels == native_channels:
            with torch.no_grad():
                self.proj.weight.zero_()
                for c in range(native_channels):
                    self.proj.weight[c, c, 0, 0, 0] = 1.0
                if bias and self.proj.bias is not None:
                    self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AdaptedModel(nn.Module):
    """Wraps a segmentation model with a leading :class:`ChannelAdapter`.

    The wrapped model is stored under ``self.model`` so its ``state_dict`` keys
    are preserved verbatim (prefixed ``model.``); adapter keys are
    ``adapter.proj.*``. Forward args/kwargs pass through unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        in_channels: int,
        native_channels: int,
        identity_map: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.adapter = ChannelAdapter(in_channels, native_channels, identity_map=identity_map)
        self.model = model

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.model(self.adapter(x), *args, **kwargs)


def load_with_adapter_report(model: nn.Module, state_dict: dict, strict: bool = False
                             ) -> Tuple[List[str], List[str]]:
    """Load ``state_dict`` (strict=False by default) and log key differences.

    Returns ``(missing_keys, unexpected_keys)``. Adapter keys missing from an old
    checkpoint are expected and logged at INFO, not WARNING.
    """
    result = model.load_state_dict(state_dict, strict=strict)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    adapter_missing = [k for k in missing if k.startswith("adapter.")]
    other_missing = [k for k in missing if not k.startswith("adapter.")]
    if adapter_missing:
        logger.info("adapter keys initialised fresh (expected): %s", adapter_missing)
    if other_missing:
        logger.warning("unexpected missing keys on load: %s", other_missing)
    if unexpected:
        logger.warning("unexpected keys in checkpoint: %s", unexpected)
    return missing, unexpected
