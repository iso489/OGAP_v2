import torch
import numpy as np
import nibabel as nib

from ogap.models.adapters import ChannelAdapter, AdaptedModel, load_with_adapter_report
from ogap.models import build_student
from ogap.legacy import MultiModalSegDataset, prepare_student_input


def test_adapter_maps_7_to_native():
    adapter = ChannelAdapter(in_channels=7, native_channels=8)
    assert adapter(torch.randn(1, 7, 16, 16, 16)).shape == (1, 8, 16, 16, 16)


def test_adapter_identity_when_equal():
    adapter = ChannelAdapter(8, 8).eval()
    x = torch.randn(1, 8, 8, 8, 8)
    assert torch.allclose(adapter(x), x, atol=1e-6)


def test_adapter_identity_map_for_tissue_student_layout():
    identity_map = [(0, 0), (1, 1), (2, 2), (3, 3), (7, 4), (8, 5), (9, 6), (10, 7)]
    adapter = ChannelAdapter(11, 8, identity_map=identity_map).eval()
    x = torch.randn(1, 11, 4, 4, 4)
    y = adapter(x)
    assert torch.allclose(y[:, :4], x[:, :4])
    assert torch.allclose(y[:, 4:], x[:, 7:])


def test_adapted_student_forward_7ch():
    model = AdaptedModel(build_student(in_channels=8, num_classes=4, base=8),
                         in_channels=7, native_channels=8).eval()
    y = model(torch.randn(1, 7, 16, 16, 16))
    out = y[0] if isinstance(y, (tuple, list)) else y
    assert out.shape[:2] == (1, 4)


def test_strict_false_load_only_adapter_missing():
    plain_sd = build_student(in_channels=8, num_classes=4, base=8).state_dict()
    model = AdaptedModel(build_student(8, 4, base=8), in_channels=7, native_channels=8)
    sd = {f"model.{k}": v for k, v in plain_sd.items()}
    missing, unexpected = load_with_adapter_report(model, sd, strict=False)
    assert all(k.startswith("adapter.") for k in missing), missing
    assert unexpected == []


def test_prepare_student_input_preserves_tissue_priors():
    x = torch.arange(1 * 7 * 2 * 2 * 2, dtype=torch.float32).reshape(1, 7, 2, 2, 2)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    out = prepare_student_input(x, mask)
    assert out.shape == (1, 11, 2, 2, 2)
    assert torch.equal(out[:, 0], x[:, 0])
    assert torch.count_nonzero(out[:, 1]) == 0
    assert torch.equal(out[:, 2], x[:, 2])
    assert torch.count_nonzero(out[:, 3]) == 0
    assert torch.equal(out[:, 4:7], x[:, 4:7])
    assert torch.equal(out[:, 7], torch.ones_like(out[:, 7]))
    assert torch.equal(out[:, 8], torch.zeros_like(out[:, 8]))


def test_full_volume_tissue_priors_use_training_crop(tmp_path):
    case_id = "case001"
    base = np.arange(10 * 11 * 12, dtype=np.float32).reshape(10, 11, 12)
    for offset, tissue in enumerate(("csf", "wm", "gm")):
        nib.save(
            nib.Nifti1Image(base + offset * 10000.0, affine=np.eye(4)),
            tmp_path / f"{case_id}_{tissue}.nii.gz",
        )

    ds = MultiModalSegDataset.__new__(MultiModalSegDataset)
    ds.tissue_prior_dir = str(tmp_path)
    ds.patch_size = (8, 8, 8)
    crop = (slice(1, 9), slice(2, 10), slice(3, 11))
    tissue = ds._load_tissue_priors(
        case_id,
        (8, 8, 8),
        native_spatial_shape=(10, 11, 12),
        crop_slices=crop,
        padded_spatial_shape=(10, 11, 12),
    )
    assert tissue.shape == (3, 8, 8, 8)
    assert torch.equal(tissue[0], torch.from_numpy(base[crop]).float())
