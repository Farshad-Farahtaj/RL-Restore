import torch

from rlrestore.baselines.monolith import MonoRestoreCNN, count_parameters


def test_param_count_depth14_width64_matches_toolbox():
    # 448,195 = 102.8% of the 436,068-param toolbox (BN buffers are not params).
    assert count_parameters(MonoRestoreCNN(depth=14, width=64)) == 448_195


def test_param_count_depth14_width48_variant():
    assert count_parameters(MonoRestoreCNN(depth=14, width=48)) == 253_203


def test_forward_preserves_shape_and_is_finite():
    model = MonoRestoreCNN(depth=14, width=64)
    x = torch.rand(2, 3, 63, 63)
    y = model(x)
    assert y.shape == (2, 3, 63, 63)
    assert torch.isfinite(y).all()


def test_forward_is_unclamped():
    """Global residual out = x - body(x) must NOT clamp (PSNR convention)."""
    model = MonoRestoreCNN(depth=4, width=8)
    # Drive the body to a large constant so the residual leaves [0,1].
    with torch.no_grad():
        for m in model.body:
            if isinstance(m, torch.nn.Conv2d):
                m.weight.zero_()
                m.bias.fill_(0.0)
        # final conv bias large -> out = x - large -> strongly negative
        last_conv = [m for m in model.body if isinstance(m, torch.nn.Conv2d)][-1]
        last_conv.bias.fill_(5.0)
    with torch.no_grad():
        out = model(torch.rand(1, 3, 8, 8))
    assert float(out.min()) < 0.0  # unclamped: values escaped [0,1]


def test_restore_for_display_clamps_to_unit_interval():
    model = MonoRestoreCNN(depth=4, width=8)
    with torch.no_grad():
        last_conv = [m for m in model.body if isinstance(m, torch.nn.Conv2d)][-1]
        last_conv.bias.fill_(5.0)
    with torch.no_grad():
        disp = model.restore_for_display(torch.rand(1, 3, 8, 8))
    assert float(disp.min()) >= 0.0 and float(disp.max()) <= 1.0


def test_depth_below_two_rejected():
    import pytest

    with pytest.raises(ValueError, match="depth"):
        MonoRestoreCNN(depth=1)
