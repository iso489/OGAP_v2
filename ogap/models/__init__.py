"""OGAP neural-network models.

Conv blocks, the full-modality teacher(s), the missing-modality student, and the
new continuous-depth variants (ODE bottleneck teacher, weight-tied ODE student).
The classic classes are extracted verbatim from the v9 monolith for checkpoint
compatibility; the ODE variants are additive and opt-in.
"""
from __future__ import annotations

from .blocks import (
    CurriculumDropout3D,
    DenseConvBlock3D,
    ECABlock3D,
    FeatureProjector,
    MixStyle3D,
    ResConvBlock3D,
    SEBlock3D,
)
from .student import UNet3DStudent
from .teacher import SegMambaTeacher, SwinUNETRTeacher, UNet3DTeacher, build_teacher
from .ode import ConvODEFunc3D, ODEBlock3D, ODEBottleneckTeacher, build_ode_teacher
from .student_ode import WeightTiedODEBlock3D
from .domain_adversarial import DomainClassifier, GradientReversalLayer, grad_reverse
from .auxiliary_heads import AuxiliaryTissueHead, auxiliary_tissue_loss


def build_student(in_channels: int, num_classes: int, base: int = 16, **kwargs):
    """Convenience constructor mirroring :func:`build_teacher`.

    The v9 monolith instantiates ``UNet3DStudent`` directly inside ``train``;
    this thin builder gives the NAS / supernet code (:mod:`ogap.nas`) a single
    seam to construct students from an architecture spec.
    """
    return UNet3DStudent(in_channels, num_classes, base=base, **kwargs)


__all__ = [
    "CurriculumDropout3D", "DenseConvBlock3D", "ECABlock3D", "FeatureProjector",
    "MixStyle3D", "ResConvBlock3D", "SEBlock3D", "UNet3DStudent",
    "SegMambaTeacher", "SwinUNETRTeacher", "UNet3DTeacher", "build_teacher", "build_student",
    "ConvODEFunc3D", "ODEBlock3D", "ODEBottleneckTeacher", "build_ode_teacher",
    "WeightTiedODEBlock3D",
    "DomainClassifier", "GradientReversalLayer", "grad_reverse",
    "AuxiliaryTissueHead", "auxiliary_tissue_loss",
]
