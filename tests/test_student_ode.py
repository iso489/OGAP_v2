
import torch
from ogap.models import WeightTiedODEBlock3D, ResConvBlock3D


def test_weight_tying_param_count_independent_of_depth():
    shallow = WeightTiedODEBlock3D(16, 16, steps=2)
    deep    = WeightTiedODEBlock3D(16, 16, steps=16)
    assert shallow.param_count() == deep.param_count()  # depth is free


def test_more_depth_changes_output_not_params():
    torch.manual_seed(0)
    blk2 = WeightTiedODEBlock3D(8, 8, steps=2)
    blk8 = WeightTiedODEBlock3D(8, 8, steps=8)
    blk8.load_state_dict(blk2.state_dict())  # same weights
    x = torch.randn(1, 8, 8, 8, 8)
    y2, y8 = blk2(x), blk8(x)
    assert y2.shape == y8.shape == (1, 8, 8, 8, 8)
    assert not torch.allclose(y2, y8)  # deeper integration -> different output


def test_channel_projection_and_shape():
    blk = WeightTiedODEBlock3D(4, 16, steps=4)
    y = blk(torch.randn(2, 4, 8, 8, 8))
    assert y.shape == (2, 16, 8, 8, 8)


def test_param_efficiency_vs_resconv_stack():
    # a single weight-tied ODE block with depth 8 vs 8 distinct ResConvBlocks
    ode = WeightTiedODEBlock3D(16, 16, steps=8).param_count()
    stack = sum(ResConvBlock3D(16, 16).param_count() if hasattr(ResConvBlock3D(16,16),"param_count")
                else sum(p.numel() for p in ResConvBlock3D(16,16).parameters()) for _ in range(8))
    assert ode < stack  # weight-tied is far smaller at equal effective depth


def test_solver_mode_runs():
    blk = WeightTiedODEBlock3D(8, 8, steps=4, deploy_mode="solver")
    y = blk(torch.randn(1, 8, 8, 8, 8))
    assert y.shape == (1, 8, 8, 8, 8)
