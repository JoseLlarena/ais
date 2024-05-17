"""
Functions to draw clouds of 3D points and links between them
"""
from itertools import starmap, chain, pairwise
from typing import Sequence, Tuple

from matplotlib.axes import Axes
from matplotlib.colors import to_rgba, to_rgb
from numpy import ndarray, zeros, pi as PI, array, cos, sin, mean, linspace, asarray
from scipy.spatial import KDTree

from ais import Fn, check
from ais.viz import Color
from ais.viz.drawing import draw_vec, KISH, draw_arc, draw_text, K, RGBA
from ais.viz.geometry import normed, Z_AXIS

EDGE_COLOUR = to_rgba(KISH, .35)
PROJ_COLOUR = to_rgba(KISH, .25)


def draw_level(ax: Axes,
               states: ndarray,
               outputs: ndarray,
               labels: Sequence[str],
               scale: float,
               palette: Fn[[float], RGBA],
               top: bool) -> Axes:
    """
    Draws a cloud of labelled 3D states.

    :param ax: the plot axes
    :param states: the states to be drawn
    :param outputs: the outputs corresponding to the given states, used to colour them
    :param labels: the state labels
    :param scale: the extent of the x, y and z axes
    :param palette: the mapping from output values to colours
    :param top: whether the labels should be drawn above or below the states
    :return: the plot axes
    """
    check(ax, Axes).check(states, ndarray, lambda: states.ndim == 2 and states.shape[0] == 3 and states.size)
    check(outputs, ndarray, lambda: outputs.ndim == 1 and outputs.size and outputs.shape[0] == states.shape[-1])
    check(labels, Sequence, lambda: len(labels) > 0 and len(labels) == states.shape[-1])
    check(scale, float, lambda: scale > 0)
    check(palette, Fn)

    font_size = ax.xaxis.label.get_fontsize() + 2
    marker_size = scale * .01  # MAGIC NUMBER : looks good on my computer
    label_z_gap = font_size * scale * .003 * (-1.5 if top else 1)  # MAGIC NUMBER : looks good on my computer

    for state, output in zip(states.T, outputs):
        _draw_level_vertex(ax, pos=state, c=palette(output), s=marker_size)

    if labels:  # MAGIC NUMBER : looks good on my computer
        neighbourhoods = KDTree(states.T).query_ball_point(tuple(states.T), r=scale * .05, return_sorted=False)

        done = set()
        for neighbourhood in neighbourhoods:

            undone = sorted(set(neighbourhood) - done)

            if undone:
                done.update(undone)

                centre = mean(states[:2, asarray(undone)], axis=-1, keepdims=len(undone) == 1)
                for col, pos in zip(undone, _compute_positions(centre, scale * .03, len(undone))):  # MAGIC NUMBER
                    state = zeros(3)
                    state[:2] = pos
                    state[2] = states[2, col] - label_z_gap
                    _draw_level_vertex_label(ax, labels[col], state, s=font_size)

    return ax


def draw_level_links(ax: Axes, *levels: ndarray, scale: float = 2., colour: Color = EDGE_COLOUR, ls: str = ':') -> Axes:
    """
    Draw links between the points in the given levels.

    :param ax: the plot axes
    :param levels: the clouds of points to draw links between
    :param scale: the extent of the x, y and z axes
    :param colour: the colour of the links
    :param ls: the line style of the links
    :return: the plot axes
    """
    check(ax, Axes).check(levels, Tuple, lambda: len(levels) > 1).check(scale, float, lambda: scale > 0)

    for i, (level, next_level) in enumerate(pairwise(chain(levels, [dummy := None]))):

        if next_level is not dummy:
            links = starmap(lambda start, end: (tuple(start), tuple(end)), zip(level.T, next_level.T))
            for link_start, link_end in links:
                _draw_edge(ax, *map(array, [link_start, link_end]), scale=scale, c=colour, ls=ls)

    return ax


# ------------------------------------------ DELEGATE FUNCTIONS --------------------------------------------------------


def _compute_positions(centre: ndarray, radius: float, n: int) -> ndarray:
    """
    Generate n points in 2D that are equidistant from the center and subtend the same angle between them.

    :param centre: The center vector as a list or numpy array [cx, cy].
    :param radius: The radius of the circle.
    :param n: number of points.
    :return: array of shape (n, 2) with the coordinates of the points.
    """
    if n == 1:
        return centre.T

    x, y = centre
    angles = linspace(0, 2 * PI, n, endpoint=False)
    return array([[x + radius * cos(angle), y + radius * sin(angle)] for angle in angles])


def _draw_level_vertex(ax: Axes, pos: ndarray, c: Color, s: float):
    edge = to_rgba([1 - d for d in to_rgb(c)], alpha=1)
    draw_arc(ax, normal=Z_AXIS, radius=s, centre=pos, c=c, ec=edge, lw=.5, zorder=5)


def _draw_edge(ax: Axes, tail: ndarray, head: ndarray, scale: float, c: Color, ls: str):
    # trim = scale * .08  # MAGIC NUMBER : just happens to look good on my computer
    trim = scale * .01
    v = tail - head
    # draw_vec(ax, head=head + normed(v) * trim * .6, tail=tail - normed(v) * trim, c=c, ls=ls,
    #          ratio=scale * .02)  # MAGIC NUMBER
    draw_vec(ax, head=head + normed(v) * trim * 1, tail=tail - normed(v) * trim, c=c, ls=ls,
             ratio=scale * .02)  # MAGIC NUMBER


def _draw_level_vertex_label(ax: Axes, label: str, pos: ndarray, s: float):
    draw_text(ax, label, pos, weight='bold', s=s, c=K, va='top')
