"""Drift guard for the strangler-fig duplication. [audit S2-C]

The v9.1 refactor keeps model classes in BOTH ``ogap/legacy.py`` and
``ogap/models/*`` (intentionally, for checkpoint safety). They are documented as
byte-identical extractions, so a freshly-seeded instance of each must produce the
identical forward output. If a future edit "improves" one copy and not the other,
these tests fail loudly — catching the divergence the audit warned about. The
specific S2-C example — ``ogap.models.teacher.build_teacher`` not handling
``arch='ode'`` while ``ogap.legacy.build_teacher`` does — is now RESOLVED and
pinned by ``test_package_build_teacher_dispatches_ode`` below.
"""
import torch

import ogap.legacy as legacy
from ogap.models.blocks import ResConvBlock3D as PkgResBlock
from ogap.models.student import UNet3DStudent as PkgStudent
from ogap.models.teacher import UNet3DTeacher as PkgTeacher


def _eval_forward(cls, ctor_args, x, seed=0):
    torch.manual_seed(seed)
    m = cls(*ctor_args).eval()
    with torch.no_grad():
        out = m(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def test_resconvblock_legacy_matches_package():
    x = torch.randn(2, 8, 8, 8, 8)
    a = _eval_forward(legacy.ResConvBlock3D, (8, 16), x)
    b = _eval_forward(PkgResBlock, (8, 16), x)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_student_legacy_matches_package():
    x = torch.randn(1, 8, 32, 32, 32)
    a = _eval_forward(legacy.UNet3DStudent, (8, 4), x)
    b = _eval_forward(PkgStudent, (8, 4), x)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_teacher_legacy_matches_package():
    x = torch.randn(1, 4, 32, 32, 32)
    a = _eval_forward(legacy.UNet3DTeacher, (4, 4), x)
    b = _eval_forward(PkgTeacher, (4, 4), x)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_package_build_teacher_dispatches_ode():
    """S2-C resolved: ogap.models.build_teacher(arch='ode') builds the ODE teacher
    (consistent with ogap.legacy.build_teacher) instead of raising ValueError."""
    from ogap.models import ODEBottleneckTeacher
    from ogap.models.teacher import build_teacher
    m = build_teacher(4, 4, base=8, arch="ode")
    assert isinstance(m, ODEBottleneckTeacher)
    # error message also advertises 'ode' as a valid arch
    try:
        build_teacher(4, 4, base=8, arch="not_a_real_arch")
    except ValueError as exc:
        assert "ode" in str(exc)
    else:
        raise AssertionError("build_teacher should raise on an unknown arch")
