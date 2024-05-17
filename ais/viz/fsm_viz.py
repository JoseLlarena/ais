"""
Functions to visualise the 3D state-transition diagram of an FSA
"""
from collections.abc import Sequence, Iterable
from math import degrees
from typing import Tuple

from matplotlib.axes import Axes
from matplotlib.colors import to_rgba, ListedColormap
from numpy import ndarray, ptp, empty, zeros, float32
from numpy.linalg import norm

from ais import Fn, check
from ais.viz import Color, vec
from ais.viz.drawing import draw_marker, draw_text, draw_loop, draw_vec, draw_link, draw_3d_background, \
    choose_output_palette, make_3d_plot, draw_guidelines, draw_level_text, draw_flat_floor, TRANSP, \
    FLOOR_COLOUR, AXIS_LABEL_COLOUR, K, W, draw_grid_floor, RGBA, KISH
from ais.viz.geometry import X_AXIS, Y_AXIS, find_normal, tangent_of, normed, compute_limits, Limits

EDGE_COLOUR = to_rgba(KISH, .5)
# created with https://www.vis4.net/palettes from a sequential interpolation between W and K
BINARY_PALETTE = ListedColormap(
    ['#ffffff', '#dbdbdb', '#b9b9b9', '#979797', '#777777', '#595959', '#3c3c3c', '#212121'], name='binary')
# created with https://www.vis4.net/palettes from a diverging interpolation between B,W and W,R
POLAR_PALETTE = ListedColormap(
    ['#01579b', '#5d7db4', '#95a6cd', '#cad2e6', '#ffffff', '#f7cac1', '#e89586', '#d2604f', '#b71c1c'], name='polar')
NO_PALETTE = lambda _: W


def plot_fsm(ax: Axes | None = None,
             vertices: ndarray = empty(0),
             edges: Tuple[Tuple[int, int], ...] = (),
             outputs: ndarray | None = None,
             vertex_labels: Iterable[str] = (),
             edge_labels: Iterable[str] = (),
             *,
             caption: str = '',
             x_label: str = 'X',
             y_label: str = 'Y',
             z_label: str = 'Z',
             limits: Limits = (),
             floor_z: float | None = None,
             grid: bool = True,
             background: bool = True,
             guidelines: bool = True) -> Axes:
    """
    Draws a Finite State Machine as a state-transition diagram in 3D space, optionally with a custom background plus
    a floor, drawn under the graph, to give a better sense of depth.

    :param ax: the plot axes, created automatically, along with the corresponding figure, if not provided
    :param vertices: the unique vertices of the graph as a DxK array
    :param edges: the edges of the graph as pairs of vertex indices/ids
    :param outputs: the outputs corresponding to each vertex, used for coloring the vertices
    :param vertex_labels: the labels to draw under the vertices
    :param edge_labels: the labels to draw above or on the side of the edges
    :param caption: the level annotation
    :param x_label: the label for the x axis
    :param y_label: the label for the y axis
    :param z_label: the label for the z axis
    :param limits: the lower and upper bounds for x, y and z axes
    :param floor_z: the z-level at which to draw the floor
    :param guidelines: whether to draw lines at x=0 and z=0 on the floor
    :param background: whether to draw the background
    :param grid: whether to draw a gridded or a flat floor
    :return: the plot axes
    """
    check(vertices, ndarray, lambda: vertices.ndim == 2 and vertices.size)
    L = vertices.shape[-1]
    check(edges, tuple, lambda: len(edges) <= L ** 2, f'{vertices.shape} {len(edges)}')
    check(outs := outputs, ndarray | None, lambda: outs is None or (outs.shape == (L,) and outs.size),
          lambda: f'{outs.shape} {vertices.shape}')
    check(vertex_labels, Iterable).check(edge_labels, Iterable)
    vertex_labels = tuple(vertex_labels)
    check(val=not vertex_labels or len(vertex_labels) == L)
    edge_labels = tuple(edge_labels)
    check(val=not edge_labels or len(edge_labels) == len(edges))
    check(caption, str).check(x_label, str).check(y_label, str).check(z_label, str)
    check(limits, Tuple | ndarray).check(floor_z, float | float32 | None)

    limits = (bottom, top) = limits if len(limits) else compute_limits(vertices)
    scale = ptp(limits).item() or 1.
    palette = choose_output_palette(empty(0) if outputs is None else outputs)
    floor_z = floor_z if floor_z is not None else bottom

    ax = ax or make_3d_plot()
    if background:
        ax = draw_3d_background(ax, limits, x_label, y_label, z_label, tick_colour=TRANSP)
    ax = draw_grid_floor(ax, limits, z=floor_z) if grid else draw_flat_floor(ax, limits, z=floor_z, colour=FLOOR_COLOUR)
    if guidelines:
        ax = draw_guidelines(ax, limits, z=floor_z)
    if caption:
        ax = draw_level_text(ax, text=caption, xy_limits=limits, z=floor_z, size=.1, mode='up',
                             colour=AXIS_LABEL_COLOUR, off=10.)
    return draw_fsm(ax, vertices, edges, outputs, vertex_labels, edge_labels, scale=scale, palette=palette)


# --------------------------------------------- DELEGATE FUNCTIONS -----------------------------------------------------

def draw_fsm(ax: Axes,
             vertices: ndarray,
             edges: Tuple[Tuple[int, int], ...],
             outputs: ndarray | None = None,
             vertex_labels: Sequence[str] = (),
             edge_labels: Sequence[str] = (),
             *,
             scale: float = 2,
             palette: Fn[[float], RGBA] = BINARY_PALETTE) -> Axes:
    """
    Draws a Finite State Machine as a state-transition diagram in 3D space, optionally with a custom background plus
    a floor, drawn under the graph, to give a better sense of depth.

    :param ax: the plot axes, created automatically, along with the corresponding figure, if not provided
    :param vertices: the unique vertices of the graph as a DxK array
    :param edges: the edges of the graph as pairs of vertex indices/ids
    :param outputs: the outputs corresponding to each vertex, used for coloring the vertices
    :param vertex_labels: the labels to draw under the vertices
    :param edge_labels: the labels to draw above or on the side of the edges
    :param scale: the extent of the plot to help size components; default is 2, corresponding to [-1, 1] bounds
    :param palette: the palette to colour the vertices based on the associated outputs; white when outputs not given
    :return: the plot axes
    """

    _draw_vertices(ax, vertices, outputs if outputs is not None else zeros(vertices.shape[-1]), palette=palette)

    if vertex_labels:
        _draw_vertex_labels(ax, vertices, vertex_labels, scale=scale)

    _draw_edges(ax, vertices, edges, scale=scale)

    if edge_labels:
        _draw_edge_labels(ax, vertices, edges, edge_labels, scale=scale)

    return ax


def _draw_vertices(ax: Axes, vertices: ndarray, outputs: ndarray, palette: Fn) -> Axes:
    marker_size = 6  # MAGIC NUMBER: just happens to look good on my computer
    for vertex, output in zip(vertices.T, outputs):
        _draw_vertex(ax, pos=vertex, s=marker_size, c=palette(output))

    return ax


def _draw_vertex_labels(ax: Axes, vertices: ndarray, labels: Iterable[str], scale: float) -> Axes:
    font_size = ax.xaxis.label.get_fontsize()
    label_gap = vec(0, 0, font_size * scale * .002)  # MAGIC NUMBER: just happens to look good on my computer

    for vertex, label in zip(vertices.T, labels):
        if label:
            _draw_vertex_label(ax, label, pos=vertex - label_gap, s=font_size)

    return ax


def _draw_edges(ax: Axes, vertices: ndarray, edges: Tuple[Tuple[int, int], ...], scale: float) -> Axes:
    font_size = ax.xaxis.label.get_fontsize() + 2  # MAGIC NUMBER: just happens to look good on my computer
    rad = scale * .02  # MAGIC NUMBER: just happens to look good on my computer
    rad_gap = vec(0, 0, rad)
    gap = font_size * scale * .003  # MAGIC NUMBER: just happens to look good on my computer
    tag_gap = vec(0, 0, gap)

    for start, end in edges:

        if start == end:
            _draw_cycle(ax, pos=vertices[:, start] + tag_gap * .5 + rad_gap, radius=rad, scale=scale)

        elif (end, start) in edges:
            _draw_switch(ax, *vertices[:, (start, end)].T, scale=scale)

        else:
            _draw_chain(ax, *vertices[:, (start, end)].T, scale=scale)

    return ax


def _draw_edge_labels(ax: Axes,
                      vertices: ndarray,
                      edges: Tuple[Tuple[int, int], ...],
                      labels: Iterable[str],
                      scale: float) -> Axes:
    font_size = ax.xaxis.label.get_fontsize() + 2  # MAGIC NUMBER: just happens to look good on my computer
    rad = scale * .03  # MAGIC NUMBER: just happens to look good on my computer
    gap = font_size * scale * .004  # MAGIC NUMBER: just happens to look good on my computer
    margin = scale * .04  # MAGIC NUMBER: just happens to look good on my computer
    marker_radius = scale * .04  # MAGIC NUMBER: just happens to look good on my computer

    for (start, end), label in zip(edges, labels):

        if start == end:  # vertex + vertex-marker radius + safety margin/2 + loop radius + safety margin/2
            label_pos = vertices[:, start] + vec(0, 0, margin) + vec(0, 0, marker_radius) + vec(0, 0, rad)

        elif (end, start) in edges:  # half-way along arc between endpoints + safety margin
            arc = _draw_switch(ax, *vertices[:, (start, end)].T, scale=scale, c=TRANSP)  # FIXME
            mid = arc.shape[-1] // 2
            label_pos = find_normal(tangent_of(arc, t=mid)) * gap + arc[:, mid]

        else:  # half-way between endpoints + safety margin (horizontal if vector's vertical-ish, vertical otherwise)
            halfway = vertices[:, (start, end)].mean(axis=-1)
            label_pos = halfway + (vec(0, 0, margin))

        if label:
            _draw_edge_label(ax, label, label_pos, font_size)

    return ax


def _draw_vertex(ax: Axes, pos: ndarray, s: float, c: Color):
    draw_marker(ax, pos, c=c, shape='o', s=s, e=K, zorder=2)


def _draw_cycle(ax: Axes, pos: ndarray, radius: float, scale: float, c: Color = EDGE_COLOUR):
    normal = X_AXIS - Y_AXIS
    trim = min(degrees(scale * .005 / radius), 359.99)  # MAGIC NUMBER: just happens to look good on my computer
    ratio = .01 * scale  # MAGIC NUMBER : just happens to look good on my computer
    arrow_length = .01 * scale  # MAGIC NUMBER : just happens to look good on my computer

    draw_loop(ax, pos, normal, radius=radius, trim=trim, ec=c, zorder=3, length=arrow_length, ratio=ratio)


def _draw_switch(ax: Axes, start: ndarray, end: ndarray, scale: float, c: Color = EDGE_COLOUR) -> ndarray:
    _radius = norm(end - start) * .5
    trim = min(degrees(scale * .05 / _radius), 359.99)  # MAGIC NUMBER : just happens to look good on my computer
    ratio = .02 * scale  # MAGIC NUMBER : just happens to look good on my computer
    length = .01  # MAGIC NUMBER : just happens to look good on my computer
    warp = .5  # MAGIC NUMBER : just happens to look good on my computer

    return draw_link(ax, start, end, trim=trim, warp=warp, c=c, zorder=3, length=length, ratio=ratio)


def _draw_chain(ax: Axes, tail: ndarray, head: ndarray, scale: float, c: Color = EDGE_COLOUR):
    v = tail - head
    ratio = scale * (.02 if norm(v) < scale * .1 else .03)  # MAGIC NUMBER : just happens to look good on my computer
    trim = scale * .025  # MAGIC NUMBER : just happens to look good on my computer

    draw_vec(ax, head=head + normed(v) * trim, tail=tail - normed(v) * trim, c=c, zorder=3, ratio=ratio)


def _draw_vertex_label(ax: Axes, label: str, pos: ndarray, s: float):
    draw_text(ax, label, pos, weight='bold', s=s, c=K, va='top', zorder=4)


def _draw_edge_label(ax: Axes, label: str, pos: ndarray, s: float):
    draw_text(ax, label, pos, weight='bold', s=s, c=K, zorder=4)
