"""Test-time augmentation / adaptation (opt-in, gated by inference.tta).

flip_tta_predict averages softmax predictions over axis flips; bn_adapt
recalibrates BatchNorm running statistics to the test batch. All CPU, no data.
"""
import torch
import torch.nn as nn

from ogap.inference.tta import bn_adapt, flip_tta_predict


def test_flip_tta_equals_plain_for_equivariant_model():
    torch.manual_seed(0)
    model = nn.Conv3d(4, 3, kernel_size=1)        # pointwise => flip-equivariant
    x = torch.randn(2, 4, 5, 5, 5)
    tta = flip_tta_predict(model, x)
    plain = torch.softmax(model(x), dim=1)
    assert tta.shape == (2, 3, 5, 5, 5)
    assert torch.allclose(tta, plain, atol=1e-5)


def test_flip_tta_probabilities_sum_to_one():
    model = nn.Conv3d(4, 3, kernel_size=1)
    p = flip_tta_predict(model, torch.randn(1, 4, 4, 4, 4))
    assert torch.allclose(p.sum(dim=1), torch.ones(1, 4, 4, 4), atol=1e-5)


def test_flip_tta_handles_tuple_output():
    class TupleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(4, 3, kernel_size=1)

        def forward(self, x):
            return self.conv(x), torch.zeros(1)        # (logits, aux)

    p = flip_tta_predict(TupleModel(), torch.randn(1, 4, 4, 4, 4))
    assert p.shape == (1, 3, 4, 4, 4)


def test_flip_tta_batches_all_eight_flips_in_one_forward():
    class CountingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(4, 3, kernel_size=1)
            self.batch_sizes = []

        def forward(self, x):
            self.batch_sizes.append(x.shape[0])
            return self.conv(x)

    model = CountingModel()
    p = flip_tta_predict(model, torch.randn(2, 4, 4, 4, 4))
    assert p.shape == (2, 3, 4, 4, 4)
    assert model.batch_sizes == [16]


def test_bn_adapt_updates_running_stats_and_preserves_original():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv3d(4, 4, kernel_size=3, padding=1), nn.BatchNorm3d(4))
    before = model[1].running_mean.clone()
    x = torch.randn(2, 4, 6, 6, 6) + 5.0                # shifted -> stats should move
    adapted = bn_adapt(model, x, steps=3)
    assert not torch.allclose(adapted[1].running_mean, before)
    assert torch.allclose(model[1].running_mean, before)   # original untouched


def test_bn_adapt_without_batchnorm_returns_module():
    out = bn_adapt(nn.Conv3d(4, 3, kernel_size=1), torch.randn(1, 4, 3, 3, 3))
    assert isinstance(out, nn.Module)


def test_bn_adapt_warns_when_no_batchnorm(caplog):
    """bn_adapt on a BatchNorm-free (e.g. GroupNorm) model must not silently
    no-op — it logs a warning so the inert call is visible. [audit S1-C]"""
    import logging
    model = nn.Sequential(nn.Conv3d(4, 4, 3, padding=1), nn.GroupNorm(2, 4))
    with caplog.at_level(logging.WARNING):
        bn_adapt(model, torch.randn(1, 4, 4, 4, 4))
    assert any("BatchNorm" in r.message for r in caplog.records)


def test_tent_adapt_updates_groupnorm_affine_and_preserves_original():
    """Tent-style entropy-minimization TTA adapts the affine params of any norm
    layer (incl. GroupNorm), so it works where bn_adapt cannot. [audit S1-C]"""
    from ogap.inference.tta import tent_adapt
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv3d(4, 4, 3, padding=1), nn.GroupNorm(2, 4), nn.Conv3d(4, 3, 1)
    )
    gn = [m for m in model.modules() if isinstance(m, nn.GroupNorm)][0]
    before = gn.weight.clone()
    x = torch.randn(2, 4, 6, 6, 6)
    adapted = tent_adapt(model, x, steps=5, lr=1e-1)
    a_gn = [m for m in adapted.modules() if isinstance(m, nn.GroupNorm)][0]
    assert not torch.allclose(a_gn.weight, before)      # affine adapted
    assert torch.allclose(gn.weight, before)            # original untouched


def test_tent_adapt_only_optimizes_norm_affine_params():
    """Tent must freeze conv weights and adapt only the norm affine params."""
    from ogap.inference.tta import tent_adapt
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv3d(4, 4, 3, padding=1), nn.GroupNorm(2, 4))
    conv_before = model[0].weight.clone()
    adapted = tent_adapt(model, torch.randn(2, 4, 6, 6, 6), steps=5, lr=1e-1)
    a_conv = adapted[0].weight
    assert torch.allclose(a_conv, conv_before)          # conv weights frozen
