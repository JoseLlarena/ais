from torch import tensor

from ais.spectral import estimate_parameters
from unittests import assert_close


def test_estimate_wfsa_parameters_from_hankel_matrix():
    hankel = tensor([
        [[0, 0, 1],
         [0, 0, 1],
         [1, 1, 0]],

        [[0, 0, 1],
         [0, 0, 1],
         [1, 1, 0]],

        [[1, 1, 0],
         [1, 1, 0],
         [0, 0, 1]]]) * 1.

    initial, transitions, final = estimate_parameters(hankel)

    assert_close(initial, tensor([.841, 0]), rtol=0, atol=1e-3)
    assert_close(final, tensor([0, -.841]), rtol=0, atol=1e-3)
    assert_close(transitions.permute(-1, -2, -3), tensor([
        [[1, 0],
         [0, 1]],

        [[0, -.707 * 2],
         [-.707, 0]]]), rtol=0, atol=1e-3)
