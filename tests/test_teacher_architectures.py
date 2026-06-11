import sys
import types

import torch
import torch.nn as nn
import pytest


class _FakeMamba(nn.Module):
    def __init__(self, d_model, **_kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.proj(x)


def test_segmamba_uses_multiscale_tsmamba_blocks(monkeypatch):
    fake = types.ModuleType("mamba_ssm")
    fake.Mamba = _FakeMamba
    monkeypatch.setitem(sys.modules, "mamba_ssm", fake)

    from ogap.models.teacher import SegMambaTeacher

    model = SegMambaTeacher(4, 4, base=2, block_style="mednext").eval()
    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        y, feats = model(x, return_features=True)
    assert y.shape == (1, 4, 32, 32, 32)
    assert feats["bottleneck"].shape[1] == model.bottleneck_channels
    assert hasattr(model.stage0, "tom") and hasattr(model.stage4, "tom")
    assert hasattr(model, "fue0") and hasattr(model, "fue3")


def test_large_kernel_teacher_block_builds_and_has_wide_kernel():
    """The mednext_large block style gives the teacher a wide (5³) depthwise kernel
    (MedNeXt's defining ingredient) for blurry low-field boundaries; additive, so the
    default checkpoint-frozen blocks are untouched."""
    from ogap.models.blocks import LargeKernelConvBlock3D, _teacher_block_class
    from ogap.models.teacher import build_teacher

    assert _teacher_block_class("mednext_large") is LargeKernelConvBlock3D
    blk = LargeKernelConvBlock3D(3, 6)
    # The depthwise conv is the first block layer; kernel is 5×5×5.
    dw = blk.block[0]
    assert tuple(dw.kernel_size) == (5, 5, 5) and dw.groups == 3
    model = build_teacher(4, 4, base=4, arch="unet3d", block_style="mednext_large").eval()
    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, 32, 32, 32)


def test_swin_unetr_teacher_uses_monai_when_available():
    pytest.importorskip("monai")
    from ogap.models.teacher import build_teacher

    model = build_teacher(4, 4, base=12, arch="swin_unetr").eval()
    x = torch.randn(1, 4, 64, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, 64, 64, 64)
