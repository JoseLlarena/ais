"""
Functions to probe the internal representations of models
"""
from collections import defaultdict
from itertools import combinations
from math import sqrt
from numbers import Number
from typing import Tuple, Sequence

from torch import Tensor, inference_mode, linspace, stack, meshgrid, zeros_like, ones, allclose, tensor
from torch.linalg import norm

from ais import check, Fn
from ais.viz.geometry import Limits

MAX_DIST = sqrt(2)  # maximum distance in simplex (0, 1), (1, 0)


@inference_mode()
def find_decision_regions(*limits: Limits,
                          decoder: Fn[[Tensor], Tensor],
                          classes: Sequence[Tensor] = (tensor([1., 0.]), tensor([0., 1.])),
                          min_dist: float = MAX_DIST * 1e-2,
                          steps: int = 9) -> Tuple[Tensor, ...]:
    """
    Finds decision regions for the given classes. A grid of points is built and fed to the decoder, whose output
    determines which points belong to which region.

    :param limits: the lower and upper bounds of the segment to probe, for each dimension in the grid
    :param decoder: the function that provides the y value for each x in the grid
    :param classes: the y values deciding the decision regions, their dimensionality (K) should equal `decoder`'s output
    :param min_dist: the minimum distance to the class that should count as part of that class
    :param steps: the number of probing points in each dimension
    :return: A collection of vectors of size NxK, each corresponding to the given classes
    """
    check(limits, Sequence, lambda: len(limits))
    check(val=all(len(bounds) == 2 for bounds in limits))
    check(decoder, Fn).check(classes, Sequence, lambda: len(classes))
    check(min_dist, float, lambda: min_dist >= 0).check(steps, int, lambda: steps > 0)

    ds = [linspace(*bounds, steps) for bounds in limits]
    dds = meshgrid(ds, indexing='ij')  # TODO USE product() rather than meshgrid()
    grid = stack([dd.reshape(-1) for dd in dds], dim=-1)  # NxD
    out = stack([decoder(point) for point in grid])  # NxD -> NxK

    regions = []
    for _class in classes:
        inside = (out.add(zeros_like(out) - _class).pow(2).sum(dim=-1).sqrt() / MAX_DIST) <= min_dist
        regions.append(grid[inside].T)  # NxK -> KxN

    return tuple(regions)  # C:KxN...


@inference_mode()
def find_isolines(states: Tensor, decoder: Fn[[Tensor], Tensor], tol: float = 1e-3, steps: int = 7) \
        -> Tuple[Tuple[int, ...], ...]:
    """
    finds lines between every pair of the given states along which all probing points have the same output value. This
    is an alternative to finding decision regions when the dimensionality is high.

    :param states: the hidden states to find isolines between
    :param decoder: the function that provides the y value for each point in the grid
    :param tol: the minimum similarity the outputs of two states should have be counted as the same
    :param steps: the number of probing points between each pair of states
    :return: a collection of sets of state indexes that lie on the same isoline
    """
    check(states, Tensor, lambda: states.ndim == 2 and states.numel())  # assumes `states` is DxN
    check(decoder, Fn).check(tol, Number, lambda: tol >= 0).check(steps, int, lambda: steps > 1)

    state_to_coaffines = defaultdict(set)
    done = set()
    num_steps = steps
    steps = linspace(0, 1, num_steps)

    for start, end in combinations(range(states.shape[-1]), r=2):
        #  D -> K
        outputs = [decoder(states[:, start] * (1 - alpha) + alpha * states[:, end]) for alpha in steps]
        coaffine = True
        for oa, ob in combinations(outputs, r=2):
            if not is_similar(oa, ob, atol=0, rtol=tol):
                coaffine = False
                break
        if coaffine and end not in done:
            state_to_coaffines[start].add(end)
            done.update([start, end])

    return tuple((state,) + tuple(sorted(state_to_coaffines[state])) for state in sorted(state_to_coaffines))


def is_similar(a: Tensor, ref: Tensor, *, atol: float = 0, rtol: float = 0) -> bool:
    check(atol, Number, lambda: 0 <= atol)
    check(rtol, Number, lambda: 0 <= rtol <= 1)

    if a.numel() == 0 and ref.numel() == 0:
        return True

    if a.ndim != ref.ndim:
        return False

    if a.ndim == 0:
        a = a.unsqueeze(0)
        ref = ref.unsqueeze(0)

    # this complex comparison is needed because it's tricky to assess similarity when
    # one or more of the components are 0
    a_n, ref_n = norm(a), norm(ref)

    if a_n == 0. or ref_n == 0.:
        return allclose(a_n, ref_n, atol=atol, rtol=rtol)

    return (allclose(((a @ ref.transpose(-1, 0)) / (a_n * ref_n)).clip(-1, 1), ones(1, dtype=a.dtype,
                                                                                    device=a.device),
                     atol=rtol) and
            allclose(a_n, ref_n, atol=atol, rtol=rtol))
