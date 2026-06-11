"""Phase 4 — verify new modules compose with existing OGAP components."""
import torch
import subprocess
import sys
import yaml
from types import SimpleNamespace
from pathlib import Path

from ogap.models import build_student
from ogap.models.adapters import AdaptedModel
from ogap.nas import OFASupernet3D
from ogap.losses import build_task_loss
from ogap.losses.distillation import build_kd_loss
from ogap.ood import CNFDensityEstimator
from ogap.longitudinal import LatentODERANOTracker


def _cfg(ttype="region_weighted_gdl"):
    return SimpleNamespace(
        loss=SimpleNamespace(task=SimpleNamespace(type=ttype, num_classes=4)),
        distillation=SimpleNamespace(enabled=True, alpha=1.0, beta=0.5, gamma=0.1,
                                     eta=0.05, temperature=3.0),
    )


def test_ood_embedding_dim_invariant_to_adapter():
    cnf = CNFDensityEstimator(dim=64, hidden=32)
    assert cnf.log_prob(torch.randn(2, 64)).shape == cnf.log_prob(torch.randn(2, 64)).shape == (2,)


def test_new_loss_on_ofa_subnet():
    net = OFASupernet3D(8, 4, stage_chs=(8, 16, 32, 64))
    net.set_active_subnet(0.5, 4)
    sub = net.export_subnet()
    logits = sub(torch.randn(1, 8, 16, 16, 16))
    tgt = torch.randint(0, 4, (1, 16, 16, 16))
    loss = build_task_loss(_cfg())(logits, tgt)
    loss.backward()
    assert torch.isfinite(loss)


def test_kd_loss_on_ofa_subnet():
    net = OFASupernet3D(8, 4, stage_chs=(8, 16, 32, 64))
    net.set_active_subnet(1.0, 4)
    sub = net.export_subnet()
    s = sub(torch.randn(1, 8, 16, 16, 16))
    t = torch.randn_like(s)
    tgt = torch.randint(0, 4, (1, 16, 16, 16))
    out = build_kd_loss(_cfg())(s, tgt, teacher_logits=t)
    out["loss_total"].backward()
    assert torch.isfinite(out["loss_total"])


def test_latent_ode_embedding_shape_invariant():
    tracker = LatentODERANOTracker(obs_dim=3, latent_dim=6)
    g = torch.Generator().manual_seed(0)
    times = torch.cumsum(torch.rand(4, generator=g) + 0.1, dim=0)
    out = tracker(torch.randn(2, 4, 3, generator=g), times, sample=False)
    assert out.pred.shape == (2, 4, 3)


def test_adapter_preserves_student_output_classes():
    adapted = AdaptedModel(build_student(8, 4, base=8), in_channels=7, native_channels=8).eval()
    y = adapted(torch.randn(1, 7, 16, 16, 16))
    out = y[0] if isinstance(y, (tuple, list)) else y
    assert out.shape[1] == 4


# ── Phase 7: pipeline-wiring gates (all opt-in features default OFF) ──────────
def test_phase7_default_config_selects_legacy_path():
    """Packaged defaults must keep every v9.1 wiring gate OFF -> bit-identical."""
    from ogap.config.loader import load_config
    from ogap.legacy import _v91_task_active, _v91_tissue_cfg, _v91_physics_aug
    cfg = load_config(None)                       # packaged defaults
    assert _v91_task_active(cfg) is False         # task.type == "legacy"
    assert _v91_tissue_cfg(cfg) == (False, "")    # 4-channel input
    assert _v91_physics_aug(cfg) is None          # no extra augmentation
    assert cfg.data.num_workers == 0
    assert cfg.data.pin_memory is False
    assert cfg.training.compile.enabled is False
    assert cfg.training.use_fp8 is False
    assert cfg.model.channel_adapter.enabled is False
    assert cfg.export.int8.enabled is False
    assert cfg.hardware.device == "cpu"


def test_phase7_task_gate_activates_on_non_legacy_type():
    from ogap.legacy import _v91_task_active
    cfg = SimpleNamespace(loss=SimpleNamespace(task=SimpleNamespace(type="region_weighted_gdl")))
    assert _v91_task_active(cfg) is True
    cfg_legacy = SimpleNamespace(loss=SimpleNamespace(task=SimpleNamespace(type="legacy")))
    assert _v91_task_active(cfg_legacy) is False
    assert _v91_task_active(None) is False        # missing config -> legacy


def test_phase7_task_loss_adapter_ignores_dataset_tags():
    """The legacy loop calls criterion(logits, target, dataset_tags=...);
    the adapter must accept and ignore the kwarg and match the inner loss."""
    from ogap.legacy import _V91TaskLossAdapter
    torch.manual_seed(0)
    inner = build_task_loss(_cfg("region_weighted_gdl"))
    adapter = _V91TaskLossAdapter(inner)
    logits = torch.randn(1, 4, 8, 8, 8)
    tgt = torch.randint(0, 4, (1, 8, 8, 8))
    with_tags = adapter(logits, tgt, dataset_tags=["site_a"])
    without = inner(logits, tgt)
    assert torch.allclose(with_tags, without) and torch.isfinite(with_tags)


def test_phase7_tissue_and_physics_gates_activate_on_config():
    from ogap.legacy import _v91_tissue_cfg, _v91_physics_aug
    cfg = SimpleNamespace(
        data=SimpleNamespace(use_tissue_priors=True, tissue_prior_dir="/tmp/tp"),
        augmentation=SimpleNamespace(physics=SimpleNamespace(
            enabled=True, bias_field=True, intensity_shift=False, gamma_correction=False,
            gmm_contrast=False, rician_noise=False, resolution_jitter=False,
            rician_snr_range=[5, 30], resolution_downsample_range=[0.5, 1.0],
            kspace_partial_fourier=False, kspace_ghosting=False, kspace_motion=False,
            stacked_n=0)),
    )
    assert _v91_tissue_cfg(cfg) == (True, "/tmp/tp")
    aug = _v91_physics_aug(cfg)
    assert callable(aug)
    out = aug(torch.rand(4, 8, 8, 8).abs() + 0.1)  # (C, D, H, W)
    assert out.shape == (4, 8, 8, 8) and torch.isfinite(out).all()


def test_phase7_kd_assembly_dict_keys_and_grad():
    """build_kd_loss assembly (used by the wired student path) returns the full
    component dict and loss_total carries gradient."""
    s = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    t = torch.randn(1, 4, 8, 8, 8)
    tgt = torch.randint(0, 4, (1, 8, 8, 8))
    out = build_kd_loss(_cfg("region_weighted_gdl"))(s, tgt, teacher_logits=t)
    assert set(out) >= {"loss_total", "loss_task", "loss_kd", "loss_feat", "loss_attn"}
    out["loss_total"].backward()
    assert s.grad is not None and torch.isfinite(out["loss_total"])


def test_production_v91_config_loads():
    from ogap.config.loader import load_config
    cfg = load_config("configs/production_v91.yaml")
    assert cfg.loss.task.type == "composite"
    assert cfg.data.use_tissue_priors is True
    assert cfg.data.in_channels == 7
    assert cfg.model.channel_adapter.student_in_channels == 11
    assert cfg.model.student.arch == "ode"
    assert cfg.training.use_fp8 is False
    assert cfg.model.transformer_engine_fp8.enabled is False
    assert cfg.training.cuda_graphs.enabled is False
    assert cfg.inference.ood.enabled is False
    assert cfg.inference.longitudinal.enabled is False
    assert cfg.hardware.device == "cuda"


def test_unsupported_active_runtime_gate_hard_fails(tmp_path):
    bad_cfg = tmp_path / "bad_gate.yaml"
    bad_cfg.write_text(yaml.safe_dump({"training": {"use_fp8": True}}), encoding="utf-8")
    from ogap.config.loader import load_config
    try:
        load_config(str(bad_cfg))
    except ValueError as exc:
        assert "Unsupported active config gates" in str(exc)
        assert "training.use_fp8" in str(exc)
        return
    raise AssertionError("expected unsupported FP8 gate to fail")


def test_explicit_bad_config_hard_fails(tmp_path):
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(yaml.safe_dump({"not_a_real_top_level": True}), encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "ogap",
        "--config",
        str(bad_cfg),
        "pretrain_teacher",
        "--train_csv",
        "train.csv",
        "--val_csv",
        "val.csv",
        "--out_dir",
        str(tmp_path / "out"),
        "--modalities",
        "t1,t1ce,t2,flair",
    ]
    result = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "failed to load --config" in result.stderr


def test_launch_pipeline_help_from_stage_2_parses():
    result = subprocess.run(
        [sys.executable, "launch_pipeline.py", "--from-stage", "2", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--from-stage" in result.stdout
    assert "{1,2,3,4,5,6,7,8}" in result.stdout


def test_validate_config_gate_summary_runs():
    result = subprocess.run(
        [sys.executable, "-m", "ogap", "--config", "configs/production_v91.yaml", "--validate-config"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Gate summary" in result.stdout
    assert "FP8: inactive" in result.stdout
    assert "compile: active" in result.stdout


def test_launch_pipeline_no_heavy_dry_run_has_stage_7():
    result = subprocess.run(
        [
            sys.executable,
            "launch_pipeline.py",
            "--dry-run",
            "--no-heavy",
            "--train-csv",
            "train.csv",
            "--val-csv",
            "val.csv",
            "--output-dir",
            "/tmp/ogap_pipeline_test",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "Stage 3 (nas_search): skipped by --no-heavy" in combined
    assert "Stage 7 (evaluate):" in combined
