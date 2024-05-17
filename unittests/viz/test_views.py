from torch import eye, ones, tensor, empty

from ais.viz.views import reduce_level_dim, quantise_level, normalise_level, stack_levels, align_levels, to_graph, Graph
from unittests import assert_close, rot3d


def test_reduces_level_dimension():
    orig = eye(4)
    orig[-1] = 0

    assert reduce_level_dim(orig).shape[0] == 3


def test_skips_reducing_level_dimension():
    orig = eye(3)

    assert_close(reduce_level_dim(orig), orig.numpy())


def test_quantises_level():
    states = ones(3, 3)
    states[:, -1] += 1e-2

    assert_close(quantise_level(states, radius=1e-1), ones(3, 3).numpy())


def test_normalises_level():
    states = eye(3)

    assert_close(normalise_level(states), 2 * eye(3).numpy() - 1)


def test_stacks_levels():
    bottom, top = eye(3, 2), eye(3, 4)

    s_bottom, s_top = stack_levels([bottom, top], pad=1.)

    shifted_top = top.clone()
    shifted_top[-1] += 1
    assert_close(s_bottom, bottom.numpy())
    assert_close(s_top, shifted_top.numpy())


def test_aligns_levels():
    bottom, top = eye(3, 4), rot3d(deg=90.) @ eye(3, 4)

    a_bottom, a_top = align_levels([bottom, top])

    assert_close(a_bottom, bottom.numpy(), atol=1e-6)
    assert_close(a_top, eye(3, 4).numpy(), atol=1e-6)


def test_convert_states_to_graph():
    states = tensor([[1, 0, 0], [0, 0, 0], [1, 0, 0]]).T.numpy()

    assert to_graph(states), Graph(vertices=tensor([[1, 0, 0], [0, 0, 0]]).T.numpy(),
                                   edges=((0, 1), (1, 0)),
                                   outputs=empty(2).numpy(),
                                   vertex_labels=(),
                                   edge_labels=())
