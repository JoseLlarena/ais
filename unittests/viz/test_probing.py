from math import sqrt

from torch import tensor, eye

from ais.viz import vec
from ais.viz.geometry import normed
from ais.viz.probing import find_decision_regions, find_isolines
from unittests import assert_close

TRUE_CLASS, FALSE_CLASS = eye(2)


def test_finds_decision_regions():
    false_region, true_region = find_decision_regions(*[(-1., 1.), (0., 5.)],
                                                      decoder=lambda s: TRUE_CLASS if s[0] >= 0 else FALSE_CLASS,
                                                      classes=(FALSE_CLASS, TRUE_CLASS),
                                                      min_dist=sqrt(2) * 1e-2,
                                                      steps=3)

    assert_close(false_region, tensor([[-1, -1, -1],
                                       [0, 2.5, 5.]]))
    assert_close(true_region, tensor([[0, 0, 0, 1, 1, 1],
                                      [0, 2.5, 5., 0, 2.5, 5.0]]))


def test_finds_isolines():
    lines = find_isolines(tensor([[-1, 0, 1], [-1, 1, 1.]]),
                          decoder=lambda state: TRUE_CLASS if normed(state.numpy()) @ vec(1, 1) == 1 else FALSE_CLASS,
                          tol=1e-3,
                          steps=3)

    assert lines == ((0, 2),)
