"""Guards for the shipped config profiles and pipeline defaults. [audit S1-A/B]

The production launcher must default to the *validated* path (conv UNet teacher +
residual student + legacy task loss + Tier-1 physics), with the full experimental
stack reachable only by explicitly selecting the production profile.
"""
import sys
from pathlib import Path

from ogap.config.loader import load_config

CFG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_baseline_profile_loads_and_validates():
    cfg = load_config(str(CFG_DIR / "baseline_v91.yaml"))
    # Validated path: legacy task loss (so the v9.1 KD assembly stays off), no
    # tissue priors required, conv UNet teacher, residual student.
    assert str(cfg.loss.task.type) == "legacy"
    assert bool(cfg.data.use_tissue_priors) is False
    assert str(cfg.model.teacher.arch) == "unet3d"
    assert str(cfg.model.student.arch) in ("standard", "res")


def test_production_profile_still_loads_and_validates():
    load_config(str(CFG_DIR / "production_v91.yaml"))  # must not raise


def test_pipeline_defaults_to_validated_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch_pipeline.py"])
    import launch_pipeline
    args = launch_pipeline.parse_args()
    assert args.teacher_arch == "unet3d"
    assert args.student_block_style == "res"
    assert "baseline" in str(args.config)
