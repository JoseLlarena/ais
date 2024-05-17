"""
High-level functions for drawing hidden representations
"""
from collections.abc import Sequence
from itertools import starmap, product, combinations, zip_longest
from typing import Tuple, List

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.pyplot import savefig, show
from numpy import ndarray
from torch import Tensor, inference_mode, cat
from torch.nn import Module

from ais import to_ndarray, check, BOS, Fn, outm
from ais.data import one_hot_coder_from, ZERO, ONE, id_coder_from, binary_coder_from
from ais.exps import BINARY_COLOURS
from ais.tracing import get_levels
from ais.training import WrapperModel
from ais.viz import Color
from ais.viz.drawing import num_to_subs, B, R, make_3d_plot, draw_vec, OR
from ais.viz.fsm_viz import plot_fsm
from ais.viz.geometry import compute_dim_limits
from ais.viz.level_viz import draw_level_links
from ais.viz.probing import MAX_DIST, find_isolines, find_decision_regions
from ais.viz.views import to_3d, as_multi_input, quantise_level, normalise_level, stack_levels, align_levels, to_graph, \
    reduce_level_dim

ORANGE = to_rgba(OR, .5)


@inference_mode()
def viz_ffw_model(model: Module,
                  task: str,
                  seq_length: int = 4,
                  quant_rad: float = .01,
                  tol: float = MAX_DIST * 5e-1,
                  steps: int = 37,
                  save: bool = False,
                  kind: str = 'general',
                  elev: float = 30,
                  azim: float = -60):
    """
    Visualises a feedforward model's hidden state and output. The model should have a `logits` and a `y` module, used
    to find the decision regions.

    :param model: the model to visualise
    :param task: the task to visualise hidden states for
    :param seq_length: the length of the sequence to use as input examples
    :param quant_rad: the radius of the state quantisation
    :param tol: the tolerance when deciding if a hidden state correspond to a given class, when finding decision regions
    :param steps: the number of points to probe in each dimension, when finding decision regions
    :param save: whether to save the figure or to show it on screen
    :param kind: the type of model
    :param elev: elevation of 3D viewport
    :param azim: azimuth of 3D viewport
    :return: nothing
    """
    check(model, Module).check(task, str).check(seq_length, int, lambda: seq_length >= 0)
    check(quant_rad, float, lambda: quant_rad >= 0).check(tol, float, lambda: tol > 0)
    check(steps, int, lambda: steps > 0, f'[{steps}]')
    check(val=kind in {'general', 'tfm', 'chiang-tfm', 'rumelhart-mlp'})

    seqs = tuple(BOS + ''.join(seq) for seq in product(ZERO + ONE, repeat=seq_length))
    match kind:
        case 'rumelhart-mlp':
            encoder = binary_coder_from(ZERO, ONE)
            seqs = tuple(map(''.join, product(ZERO + ONE, repeat=seq_length)))
        case 'chiang-tfm':
            encoder = id_coder_from([ZERO, ONE, BOS])
            seqs = tuple(BOS + ''.join([ZERO, ONE] * (length // 2)) for length in [2, seq_length // 2, seq_length])
        case 'tfm':
            encoder = id_coder_from([BOS, ZERO, ONE])
        case _:
            encoder = one_hot_coder_from([BOS, ZERO, ONE])

    model = WrapperModel(model, encoder)

    def _decoder(x: Tensor) -> Tensor:

        if kind == 'tfm':
            x = x.unsqueeze(0)

        return model.wrapped.y(model.wrapped.logits(x.unsqueeze(0))).squeeze(0)

    counters, fsas = [], []
    for seq in seqs:
        print(seq)
        outm(model(seq))
        counter, fsa = get_levels(model.wrapped)
        counters.append(counter)
        fsas.append(fsa)
    d, _ = counters[0].shape

    counter_views, fsa_views, region_views, y_views, isolines = _make_counting_views(counters,
                                                                                     fsas,
                                                                                     _decoder,
                                                                                     steps,
                                                                                     quant_rad,
                                                                                     tol)

    for seq, counter_view, fsa_view, region_view, y_view, isolines_ in \
            zip_longest(seqs, counter_views, fsa_views, region_views, y_views, isolines):

        _, n = counter_view.shape
        state_labels = _make_vertex_labels(num_states=n, start=seq[0] if kind == 'rumelhart-mlp' else '↑')
        ax = make_3d_plot(width=.2, pos=.25, bottom=.025, title=f'{task} hidden and output vectors for [{seq}]')

        draw_level_links(ax, counter_view, fsa_view)
        for idx, level_view in enumerate([counter_view, fsa_view][:]):

            graph = to_graph(path=level_view,
                             outputs=y_view,
                             state_labels=state_labels,
                             trans_labels=_make_edge_labels(seq[1:]))
            ax = plot_fsm(ax=ax,
                          vertices=graph.vertices,
                          edges=graph.edges,
                          outputs=graph.outputs,
                          vertex_labels=graph.vertex_labels,
                          edge_labels=graph.edge_labels,
                          x_label='DIM 1',
                          y_label='DIM 2',
                          z_label='DIM 3' if d >= 3 else '',
                          limits=(-1.05, 1.05) if idx == 0 else (-.75, .75),
                          floor_z=level_view[-1].min(),
                          background=idx == 0,
                          grid=False and not idx,
                          caption='Counts (hidden layer)' if idx == 0 else 'FSA (output layer)',
                          guidelines=False)

            if idx == 0 and len(region_view):
                draw_decision_regions(ax, region_view, BINARY_COLOURS, size=10)
            elif idx == 0 and isolines_ is not None and isolines_.size:
                _draw_isolines(ax, isolines_)

        if seq == BOS + ZERO + ONE: # TODO REPLACE VISUALISATION HACK FOR 01 INPUT TO CHIANG TRANSFORMER
            ax.view_init(elev=24, azim=80)
        else:
            ax.view_init(elev=elev, azim=azim)

        if save:
            _save_fig(ax, task, seq)
        else:
            show()


@inference_mode()
def viz_recurrent_model(model: Module,
                        task: str,
                        seq_length: int = 4,
                        quant_rad: float = .01,
                        tol: float = MAX_DIST * 1e-2,
                        steps: int = 37,
                        save: bool = False,
                        elev: float = 30,
                        azim: float = -60):
    """
    Visualises a recurrent model's hidden state. The model should have a `logits` and a `y` module, used
    to find the decision regions.

    :param model: the model to visualise
    :param task: the task to visualise hidden states for
    :param seq_length: the length of the sequence to use as input examples
    :param quant_rad: the radius of the state quantisation
    :param tol: the tolerance when deciding if a hidden state correspond to a given class, when finding decision regions
    :param steps: the number of points to probe in each dimension, when finding decision regions
    :param save: whether to save the figure or to show it on screen
    :param elev: elevation of 3D viewport
    :param azim: azimuth of 3D viewport
    :return: nothing
    """
    check(model, Module).check(task, str).check(seq_length, int, lambda: seq_length >= 0)
    check(quant_rad, float, lambda: quant_rad >= 0).check(tol, float, lambda: tol > 0)
    check(steps, int, lambda: steps > 0, f'[{steps}]')
    model = WrapperModel(model, one_hot_coder_from([ZERO, ONE]))

    def _decoder(x: Tensor) -> Tensor:
        return model.wrapped.y(model.wrapped.logits(x.unsqueeze(0))).squeeze(0)

    seqs = tuple(map(''.join, product(ZERO + ONE, repeat=seq_length)))

    fsas, ys = [], []
    for seq in seqs:
        print(seq)
        model(seq)
        fsa, y = get_levels(model.wrapped)
        fsas.append(fsa)
        ys.append(y)
    d, n = fsas[0].shape

    fsa_views, y_views, region_views, isolines = _make_views(fsas, ys, _decoder, steps, quant_rad, tol)
    state_labels = _make_vertex_labels(num_states=n)

    for seq, fsa_view, y_view, isolines_ in zip_longest(seqs, fsa_views, y_views, isolines):

        graph = to_graph(path=fsa_view, outputs=y_view, state_labels=state_labels, trans_labels=_make_edge_labels(seq))

        ax = plot_fsm(ax=make_3d_plot(width=.2, pos=.25, bottom=.025, title=_make_title(task, seq)),
                      vertices=graph.vertices,
                      edges=graph.edges,
                      outputs=graph.outputs,
                      vertex_labels=graph.vertex_labels,
                      edge_labels=graph.edge_labels,
                      x_label='DIM 1',
                      y_label='DIM 2',
                      z_label='DIM 3' if d >= 3 else '',
                      limits=(-1.05, 1.05),
                      guidelines=False,
                      grid=False)

        ax.view_init(elev=elev, azim=azim)

        if region_views:
            draw_decision_regions(ax, region_views, BINARY_COLOURS, size=15)

        elif isolines_ is not None and isolines_.size:
            _draw_isolines(ax, isolines_)

        if save:
            _save_fig(ax, task, seq)
        else:
            show()


def _draw_isolines(ax: Axes, states: ndarray, c: Color = ORANGE) -> Axes:
    # assumes states is Dx2*K for K >= 0
    for bounds in np.split(states, states.shape[-1] // 2, axis=-1):
        head, tail = bounds.T
        overshoot = (head - tail) * .075  # MAGIC NUMBER
        draw_vec(ax, head=head + overshoot, tail=tail - overshoot, c=c, ls=':', ratio=0, lw=1.5, zorder=1)

    return ax


def _make_views(fsas: List[Tensor], ys: List[Tensor], decoder: Fn, steps: int, quant_rad: float, tol: float) \
        -> Tuple[Tuple[ndarray, ...], Tuple[ndarray, ...], Tuple[ndarray, ...], Tuple[ndarray, ...]]:
    """ turns hidden states as simulated FSAs into viewable points """
    regions = ()
    coaffine_sets = ()
    if fsas[0].shape[0] < 3:  # memory limit on my computer, plus probing points are hard to see in 3D
        limits = compute_dim_limits(cat(fsas, dim=-1).numpy())
        regions = find_decision_regions(*limits, decoder=decoder, steps=steps, min_dist=tol)
    elif fsas[0].shape[0] != 10:  # TODO REPLACE HACK TO SKIP CHIANG TRANSFORMER
        coaffine_sets = tuple(find_isolines(fsa, decoder, tol=1e-1) for fsa in fsas)

    fsa_n = len(fsas)
    views = as_multi_input(reduce_level_dim)(tuple(fsas) + regions)
    fsa_views = as_multi_input(quantise_level)(views[:fsa_n], radius=quant_rad)
    views = as_multi_input(normalise_level)(fsa_views + views[fsa_n:])
    fsa_views, region_views = views[:fsa_n], views[fsa_n:]

    isoline_views = []
    for fsa_view, coaffines in zip(fsa_views, coaffine_sets):
        isolines = []
        for coaffine_set in coaffines:
            for start, end in combinations(coaffine_set, r=2):
                isolines.extend([fsa_view[:, start], fsa_view[:, end]])
        isoline_views.append(np.stack(isolines, axis=-1) if isolines else np.empty((3, 0)))

    return fsa_views, tuple(y[-1].clip(0, 1).numpy() for y in ys), region_views, tuple(isoline_views)


def draw_decision_regions(ax: Axes,
                          regions: Sequence[Tensor | ndarray],
                          colours: Sequence[Color] = (to_rgba(R, .1), to_rgba(B, .1)),
                          size: int = 30,
                          kind: str = 'all') -> Axes:
    """
    Draws decision regions as a cloud of probing points in 3d

    :param ax: the axes from the existing plot
    :param regions: the regions, a sequence of point clouds
    :param colours: the colours for each of the regions
    :param size: the size of the marker to represent a probing point
    :param kind: 'all' if all the points should be shown
    :return: the axes passed in
    """
    check(ax, Axes).check(regions, Sequence, lambda: len(regions))
    check(colours, Sequence, lambda: len(colours) == len(regions)).check(size, int, lambda: size > 0)
    check(val=kind in {'all'})

    for region, colour, marker in zip(regions, colours, ['s', 'o'] * ((len(regions) + 1) // 2)):
        check(region, Tensor | ndarray, lambda: 0 <= region.shape[0] <= 3, f'{region.shape}')  # expects DxN
        if not region.size:
            continue

        ax.scatter(*zip(to_3d(to_ndarray(region))), color=colour, marker=marker, s=size, zorder=0)

    return ax


def _make_counting_views(counters: List[Tensor],
                         fsas: List[Tensor],
                         decoder: Fn,
                         steps: int,
                         quant_rad: float,
                         tol: float) \
        -> Tuple[Tuple[ndarray, ...],
        Tuple[ndarray, ...],
        Tuple[Tuple[ndarray, ...], ...],
        Tuple[ndarray, ...],
        Tuple[ndarray, ...]]:
    """ turns hidden states as counters and outputs as simulated FSAs into viewable points """
    counter_views, y_views, region_views, _ = _make_views(counters, fsas, decoder, steps, quant_rad, tol)
    coaffine_sets = ()
    if counters[0].shape[0] >= 3 and counters[0].shape[0] != 10:  # TODO REPLACE HACK TO SKIP CHIANG TRANSFORMER
        coaffine_sets = tuple(find_isolines(counter, decoder, tol=1e-1) for counter in counters)

    ami = as_multi_input

    fsa_views = ami(normalise_level)(ami(quantise_level)(fsas, radius=quant_rad))
    rescaled_fsa_views = []
    rescaled_region_views = []
    rescaled_counter_views = []
    for counter_view, fsa_view in zip(counter_views, fsa_views):
        # assumes both previous normalisation and plot data limits are in [-1, 1]
        levels = stack_levels((counter_view, fsa_view), pad=.5, lower=-1., upper=.75)
        counter_view, fsa_view = align_levels(levels)
        fsa_view[:2] = fsa_view[:2] * .5
        rescaled_fsa_views.append(fsa_view)
        rescaled_counter_views.append(counter_view)

        num_points = sum(r.shape[-1] for r in region_views)
        if num_points:
            _region_views, _ = stack_levels((np.concatenate(region_views, axis=-1),) + (fsa_view,),
                                            pad=.5,
                                            lower=-1.,
                                            upper=.75)
            region_views = (_region_views[:, : region_views[0].shape[-1]], _region_views[:, region_views[0].shape[-1]:])

        rescaled_region_views.append(region_views)

    isoline_views = []
    for counter_view, coaffines in zip(rescaled_counter_views, coaffine_sets):
        isolines = []
        for coaffine_set in coaffines:
            for start, end in combinations(coaffine_set, r=2):
                isolines.extend([counter_view[:, start], counter_view[:, end]])
        isoline_views.append(np.stack(isolines, axis=-1) if isolines else np.empty((3, 0)))

    return tuple(counter_views), tuple(rescaled_fsa_views), tuple(rescaled_region_views), y_views, tuple(isoline_views)


def _make_title(task: str, seq: str) -> str:
    return f'{task} hidden state sequence for [{seq}]'


def _save_fig(ax: Axes, task: str, seq: str):
    ax.get_figure().set_size_inches(9, 9)
    savefig(f'{seq}-{task}.png'.replace(ZERO, '0').replace(ONE, '1'), bbox_inches='tight')


def _make_edge_labels(seq: Sequence[str], zero_indexed: bool = False) -> Tuple[str, ...]:
    """
    Makes labels for graph edges by adding a character representing the order of the given strings

    :param seq: sequence
    :param zero_indexed: whether the subscripts should start at 0 or 1
    :return the indexed transition labels
    """
    return tuple(starmap(lambda i, c: c + num_to_subs(i + int(not zero_indexed)), enumerate(seq)))


def _make_vertex_labels(num_states: int, start: str = '↑') -> Tuple[str, ...]:
    """ makes labels for vertices"""
    return (start,) + ('',) * (num_states - 1)
