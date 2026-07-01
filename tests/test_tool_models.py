import torch

from rlrestore.tools.models import LargeToolCNN, SmallToolCNN, build_tool
from rlrestore.tools.specs import TOOL_SPECS


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_small_param_count_matches_original_checkpoint():
    assert _n_params(SmallToolCNN()) == 21_827


def test_large_param_count_matches_original_checkpoint():
    assert _n_params(LargeToolCNN()) == 50_851


def test_tools_are_fully_convolutional_and_shape_preserving():
    for model in (SmallToolCNN(), LargeToolCNN()):
        for size in (63, 96):
            x = torch.rand(2, 3, size, size)
            assert model(x).shape == x.shape


def test_specs_table():
    assert len(TOOL_SPECS) == 12
    kinds = [s.kind for s in TOOL_SPECS]
    assert kinds == ["blur"] * 4 + ["noise"] * 4 + ["jpeg"] * 4
    archs = [s.arch for s in TOOL_SPECS]
    assert archs == ["small", "small", "large", "large"] * 3
    assert (TOOL_SPECS[0].lo, TOOL_SPECS[0].hi) == (0.0, 1.25)
    assert (TOOL_SPECS[11].lo, TOOL_SPECS[11].hi) == (10.0, 20.0)


def test_build_tool_dispatches_arch():
    assert isinstance(build_tool(TOOL_SPECS[0]), SmallToolCNN)
    assert isinstance(build_tool(TOOL_SPECS[2]), LargeToolCNN)


def test_large_tool_skip_connections_are_wired():
    """Wiring probes for bugs that preserve param count and shape.

    The b0 skip must carry the PRE-relu conv1_2 output: if it carried
    relu(b0) instead, the gradient through conv1_2.bias would be ~halved
    (negative-b0 paths contribute zero). The s3 skip must carry signal.
    """
    torch.manual_seed(0)
    m = LargeToolCNN()
    x = torch.randn(1, 3, 63, 63)

    m.zero_grad()
    m(x).sum().backward()
    b0_skip_grad = m.conv1_2.bias.grad.abs().sum().item()
    assert b0_skip_grad > 0.05, (
        f"conv1_2.bias gradient too small ({b0_skip_grad:.4f}); "
        "b0 skip may use relu(b0) instead of b0"
    )

    ref = m(x).detach()
    with torch.no_grad():
        orig = m.conv2d_.weight.data.clone()
        m.conv2d_.weight.zero_()
        out_no_branch = m(x).detach()
        m.conv2d_.weight.data.copy_(orig)
    assert not torch.allclose(ref, out_no_branch), (
        "zeroing conv2d_ had no effect; s3 skip may be broken"
    )
