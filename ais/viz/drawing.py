"""
Primitive functions for 3D drawing; plus constants used by various drawing modules.
"""
from typing import Tuple, TypeAlias

from matplotlib.axes._axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.backend_tools import Cursors
from matplotlib.colors import to_rgba, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.pyplot import figure, get_current_fig_manager, tight_layout
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import meshgrid, zeros_like, linspace, ndarray, stack, ptp, asarray, float32, ones
from numpy.linalg import norm

from ais import check, px, Fn
from ais.viz import Color, vec
from ais.viz.geometry import Y_AXIS, Z_AXIS, ORIG, make_arc, make_wedge, make_square, Limits

# colors taken from material design palette https://m2.material.io/design/color/the-color-system.html
B, R, K, G, W = (blue := '#01579B', red := '#B71C1C', black := '#212121', grey := '#FAFAFA', white := '#FFFFFF')
P, GN, KISH, OR = (purple := '#6A1B9A', green := '#43A047', blackish := '#616161', orange := '#FF6F00')
TRANSP = 0, 0, 0, 0
GRID_COLOUR = to_rgba(K, .115)
FLOOR_COLOUR = to_rgba(K, .02)
GUIDELINE_COLOUR = to_rgba(B, .1)
TICK_LABEL_COLOUR = to_rgba(K, .7)
AXIS_LABEL_COLOUR = to_rgba(K, .1)
BACK_COLOUR = G
BOTTOM_COLOUR = to_rgba(GRID_COLOUR, alpha=.01)
# created with https://www.vis4.net/palettes from a sequential interpolation between W and K
BINARY_PALETTE = ListedColormap(
    ['#ffffff', '#dbdbdb', '#b9b9b9', '#979797', '#777777', '#595959', '#3c3c3c', '#212121'], name='binary')
# created with https://www.vis4.net/palettes from a diverging interpolation between B,W and W,R
POLAR_PALETTE = ListedColormap(
    ['#01579b', '#5d7db4', '#95a6cd', '#cad2e6', '#ffffff', '#f7cac1', '#e89586', '#d2604f', '#b71c1c'], name='polar')
NO_PALETTE = lambda _: W
X_LABEL, Y_LABEL, Z_LABEL = 'XYZ'
FOCAL_LENGTH = .3
SUBSCRIPTS = '₀₁₂₃₄₅₆₇₈₉'
SUPERSCRIPTS = '⁰¹²³⁴⁵⁶⁷⁸⁹'
PPI = 72  # points per inch


def make_3d_plot(*args, **kwargs) -> Axes:
    # `computed_zorder=False` for programmatic ordering of complex 3d plots
    return make_figure(*args, **kwargs).add_subplot(1, 1, 1, projection='3d', computed_zorder=False)


def make_figure(title: str = '',
                colour: Color = BACK_COLOUR,
                width: float = .4,
                height: float = 1.,
                pos: float = .5,
                horiz: float = 0.,
                vert: float = 0.,
                left: float = 0.,
                right: float = .999,
                bottom: float = 0.,
                top: float = .95) -> Figure:
    """
    Convenience function to create a figure with the given parameters

    The `title` parameter sets the window's and not the figure's title, this is because
    setting the figure's title creates a gap at the top of the plots, to be avoided
    for aesthetic reasons and for interface faithfulness as the `top` argument's value wouldn't otherwise be honoured.

    The setting of the window title and the screen-relative sizing of the figure only works for the TkAgg backend

    :param title: the figure's window's title, different from the figure's title
    :param colour: the figure's background colour
    :param width: the width as a fraction of the screen width
    :param height: the height as a fraction of the screen height
    :param pos: the position as a fraction of the screen width
    :param horiz: horizontal margin between plots in the figure as a fraction of figure's width
    :param vert: vertical margin between plots in the figure  as a fraction of figure's height
    :param left: left margin between figure's border and plots as a fraction of figure's width
    :param right: right margin between figure's border and plots as a fraction of figure's width
    :param bottom: bottom margin between figure's border and plots as a fraction of figure's height
    :param top: top margin between figure's border and plots  as a fraction of figure's height
    :return: the newly created figure
    """
    fig = figure(facecolor=colour)
    fig.suptitle(title)

    try:  # FIXME SUPPORT OTHER BACKENDS
        fig.canvas.manager.set_window_title(title)

        mng = get_current_fig_manager()
        # resizes figure relative to screen size
        width_, height_ = mng.window.maxsize()
        mng.resize(width_ * width, height_ * height)
        # places figure relative to screen size
        mng.window.wm_geometry(f'+{int(width_ * pos)}+{0}')
    except Exception as e:
        print(e)

    fig.subplots_adjust(wspace=horiz, hspace=vert, left=left, right=right, bottom=bottom, top=top)
    # FIXME top and bottom gaps appear in the figure even after the call to tight_layout(); they can be removed
    # by resizing the figure manually
    tight_layout()

    try:
        # sets cursor to "hand" when dragging; only works for interactive backends
        for event in 'button_press', 'button_release', 'motion_notify':
            fig.canvas.mpl_connect(event + '_event', px(_on_click, fig=fig))
    except Exception as e:
        print(e)
        pass

    return fig


def draw_3d_background(ax: Axes,
                       limits: Limits,
                       x_label: str = 'X',
                       y_label: str = 'Y',
                       z_label: str = 'Z',
                       tick_colour: Color = TICK_LABEL_COLOUR) -> Axes:
    """
    Draw custom minimalist 3d background.

    :param ax: plot axes
    :param limits: the data limits for the x, y and z axes
    :param x_label: x axis label
    :param y_label: y axis label
    :param z_label: z axis label
    :param tick_colour: the colour or the axes' ticks
    :return: the plot axes
    """
    check(ax, Axes).check(limits, Tuple | ndarray, lambda: len(limits) == 2)
    check(x_label, str).check(y_label, str).check(z_label, str)
    check(tick_colour, str | float | Tuple)

    # Sets axis space limits to be maximum of all xs, ys, and zs to ensure that all points are shown and that
    # the space is symmetric along every dimension, ie, a cube
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_zlim(*limits)
    ax.grid(False)
    ax.set_facecolor(BACK_COLOUR)
    ax.set_box_aspect(ones(3))  # ensures aspect corresponds to data's scale though it affects responsiveness
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_alpha(0)  # hides axes' lines
        axis.set_pane_color(TRANSP)  # hides axes' space

    _draw_better_ticks(ax, limits, tick_colour)
    draw_axis_labels(ax, limits, x_label, y_label, z_label, TICK_LABEL_COLOUR)
    ax.set_proj_type('persp', focal_length=FOCAL_LENGTH)

    return ax


def draw_grid_floor(ax: Axes, xy_limits: Limits, z: float, grid_colour: Color = GRID_COLOUR) -> Axes:
    """
    Draws a grid at z, "the floor", to give a better sense of depth.

    :param ax: plot axes
    :param xylimits: limits of the x-y plane
    :param z: the z level of the text
    :param grid_colour: grid colour
    :return: the plot axes
    """
    check(ax, Axes).check(xy_limits, Tuple | ndarray, lambda: len(xy_limits) == 2)
    check(z, float | float32).check(grid_colour, str | float | Tuple)

    X, Y = meshgrid(linspace(*xy_limits, num=21), linspace(*xy_limits, num=21))  # MAGIC NUMBER
    ax.plot_wireframe(X, Y, Z=zeros_like(X) + z, color=grid_colour, ls='dashed', lw=.75, zorder=4)  # MAGIC NUMBER
    lo, hi = xy_limits

    scale = ptp(xy_limits)
    draw_square(ax,
                normal=Z_AXIS,
                side=scale,
                ec=grid_colour,
                c=BOTTOM_COLOUR,
                ls=':',
                centre=vec(lo + scale / 2, lo + scale / 2, z),
                zorder=4)

    return ax


def draw_flat_floor(ax: Axes,
                    xy_limits: Limits,
                    z: float,
                    colour: Color = BOTTOM_COLOUR,
                    edge_colour: Color = GRID_COLOUR) -> Axes:
    """
    Draws a square, "the floor", to give a better sense of depth.

    :param ax: plot axes
    :param xylimits: limits of the x-y plane
    :param z: the z level of the text
    :param colour: colour of the square
    :param edge_colour: colour of the edge of the square
    :return: the plot axes
    """
    check(ax, Axes).check(xy_limits, Tuple | ndarray, lambda: len(xy_limits) == 2).check(z, float | float32)
    check(colour, str | float | Tuple).check(edge_colour, str | float | Tuple)

    bottom, top = xy_limits
    scale = ptp(xy_limits)
    draw_square(ax,
                normal=Z_AXIS,
                side=scale,
                ec=edge_colour,
                c=colour,
                centre=vec(bottom + scale / 2, bottom + scale / 2, z),
                hatch='...',
                zorder=0)

    return ax


def draw_guidelines(ax: Axes, xy_limits: Limits, z: float, colour: Color = GUIDELINE_COLOUR) -> Axes:
    """
    Draw lines along y=0 and z=0

    :param ax: plot axes
    :param xylimits: limits of the x-y plane
    :param z: the z level of the text
    :param colour: grid colour
    :return: the plot axes
    """
    check(ax, Axes).check(xy_limits, Tuple | ndarray, lambda: len(xy_limits) == 2)
    check(z, float | float32).check(colour, str | float | Tuple)

    lo, hi = xy_limits
    if lo <= 0 <= hi:
        ax.plot([0] * 2, xy_limits, [z] * 2, c=colour, zorder=0, ls='--')
        ax.plot(xy_limits, [0] * 2, [z] * 2, c=colour, zorder=0, ls='--')

    return ax


def draw_level_text(ax: Axes,
                    text: str,
                    xy_limits: Limits,
                    z: float,
                    size: float = .05,
                    off: float = 1.,
                    mode: str = 'up',
                    colour: Color = K) -> Axes:
    """
    Draws flat 3D tex to serve as level text.

    :param ax: plot axes
    :param text: the text to draw
    :param xy_limits: limits of the x-y plane
    :param z: the z level of the text
    :param size: the font size of the text
    :param off: the gap between the text and the level
    :param mode: which axis to draw the text for: 'floor' -> flat on top of the xy plane, 'up' -> standing on zy plane
    :param colour: the colour of the text
    :return: the plot axes
    """
    check(ax, Axes).check(text, str)
    check(xy_limits, Tuple | ndarray, lambda: len(xy_limits) == 2)
    check(z, float | float32).check(size, float).check(off, float)
    check(val=mode in {'up', 'floor'})
    check(colour, str | float | Tuple)

    lo, hi = xy_limits
    extent = hi - lo
    offset = extent * off
    half_way = lo + extent / 2
    s = size * ax.figure.dpi / 72
    d = half_way - len(text) * s / 4
    xyz = (x := d, y := hi + .5 * s, z := z) if mode == 'floor' else (x := lo - offset, y := d, z + offset)

    return draw_flat_text(ax, text, xyz, s=s, c=colour, zdir='z' if mode == 'floor' else 'x')


def draw_flat_axis_label(ax: Axes,
                         text: str,
                         xy_limits: Limits,
                         z: float,
                         size: float = .05,
                         off: float = .075,
                         mode: str = 'x',
                         colour: Color = AXIS_LABEL_COLOUR) -> Axes:
    """
    Draws flat 3D text, ie, on one of the axis-aligned planes.

    :param ax: plot axes
    :param text: the text to draw
    :param xy_limits: limits of the x-y plane
    :param z: the z level of the text
    :param size: the font size of the text
    :param off: the gap between the text and the level
    :param mode: which axis to draw the text for: 'x' -> bottom of the xy plane, 'y' right of the xy plane,
        'z' -> zy plane
    :param colour: the colour of the text
    :return: the plot axes
    """
    check(ax, Axes).check(xy_limits, Tuple | ndarray).check(text, str).check(z, float).check(size, float,
                                                                                             lambda: size > 0)
    check(off, float).check(val=mode in 'xyz').check(colour, str | float | Tuple)

    lo, hi = xy_limits
    extent = hi - lo
    offset = extent * off
    half_way = lo + extent / 2
    f = extent * size
    s = f * ax.figure.dpi / PPI
    d = half_way - len(text) * s / 4  # MAGIC NUMBER

    if mode == 'x':
        xyz = (d, lo - offset, z)
        zdir = 'z'
    elif mode == 'y':
        xyz = (hi + offset, d, z)
        zdir = 'z'
    else:
        xyz = (half_way, d, z)
        zdir = 'x'

    return draw_flat_text(ax, text, xyz, s=s, c=colour, zdir=zdir, zorder=0)


def draw_axis_labels(ax: Axes,
                     limits: Limits,
                     x_label: str,
                     y_label: str,
                     z_label: str,
                     colour: Color = AXIS_LABEL_COLOUR) \
        -> Axes:
    """
    Draws axis labels as 3D horizontal text.

    :param ax: plot axes
    :param limits: the data limits for the x, y and z axes
    :param x_label: x axis label
    :param y_label: y axis label
    :param z_label: z axis label
    :param colour: the colour or the axes' ticks
    :return: the plot axes
    """
    check(ax, Axes).check(limits, Tuple | ndarray, lambda: len(limits) == 2)
    check(x_label, str).check(y_label, str).check(z_label, str).check(colour, str | float | Tuple)

    for mode, lab in zip('xy', (x_label, y_label)):
        if lab:
            draw_flat_axis_label(ax, lab, limits, z=limits[0], mode=mode, size=.04, off=.15, colour=colour)

    lo, hi = limits
    extent = hi - lo
    half_way = lo + extent / 2
    draw_text(ax, z_label, asarray([half_way, half_way, limits[-1] + extent * .15]), s=13, c=colour, weight='bold',
              zorder=0)

    return ax


def draw_text(ax, text: str, xyz: ndarray, **props) -> Axes:
    """
    Draws 3D text that always faces the viewing angle. The `zdir` property defines the orientation of the text, see
    `mpl_toolkits.mplot3d.art3d.get_dir_vector`.

    :param ax: plot axes
    :param text: the text to draw
    :param xyz: location of the text
    :param props: properties to be passed to `mpl_toolkits.mplot3d.axes3d.Axes3D.text`
    :return: the plot axes
    """

    check(ax, Axes).check(xyz, ndarray).check(text, str)
    if 'zdir' in props:
        check(val=props['zdir'] in tuple('xyz') + (None,))

    default_alpha = to_rgba(props.get('c', K))[-1]
    props = dict(zdir=None,
                 a=0,
                 s=10,
                 c=K,
                 alpha=default_alpha,
                 style='normal',
                 weight='normal',
                 ha='center',
                 va='top') | props

    ax.text(*xyz,
            text,
            zdir=props.pop('zdir'),
            size=props.pop('s'),
            style=props.pop('style'),
            weight=props.pop('weight'),
            rotation=props.pop('a'),
            color=props.pop('c'),
            alpha=props.pop('alpha'),
            ha=props.pop('ha'),
            va=props.pop('va'),
            **props)

    return ax


def draw_flat_text(ax, text: str, xyz: ndarray | Tuple[float, float, float], **props) -> Axes:
    """
    Draws 3D text flat on one of the planes. The 'zdir' property determines which ('x', 'y', 'z).

    :param ax: plot axes
    :param text: the text to draw
    :param xyz: location of the text
    :param props: properties: 's' -> size, 'zdir' -> orientation, 'a' -> angle, 'c' -> face colour, 'e' -> edge colour,
        'alpha' -> alpha, 'zorder' -> z-order, 'style' -> style, 'weight' -> 'weight
    :return: the plot axes
    """

    check(ax, Axes).check(text, str).check(xyz, ndarray | Tuple)

    default_alpha = to_rgba(props.get('c', K))[-1]
    props = dict(zdir='z', a=0, s=.05, c=K, alpha=default_alpha, style='normal', weight='normal', e=K, zorder=0) | props

    zdir = props['zdir']
    x, y, z = xyz

    _x, _y, _z = (x, z, y) if zdir == 'y' else (y, z, x) if zdir == 'x' else (x, y, z)

    text_path = TextPath((0, 0), text, size=props['s'])  # text's path
    trans = Affine2D().rotate(props['a']).translate(_x, _y)  # builds linear transform
    p1 = PathPatch(trans.transform_path(text_path), fc=props['c'], ec=props['e'],
                   alpha=props['alpha'], zorder=props['zorder'])  # transforms in 2D
    ax.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=_z, zdir=zdir)  # projects transformed 2D to 3D
    return ax


def draw_loop(ax: Axes,
              pos: ndarray,
              normal: ndarray = Y_AXIS,
              *,
              radius: float = 1.,
              trim: float = 0.,
              **props) -> ndarray:
    """

    Draws an arrow in the shape of a loop at the given position, meant for FSA-graph self-loops.

    :param ax: plot axes
    :param pos: position of the bottom of the loop
    :param normal: vector orthogonal to the plane the loop is drawn on
    :param radius: radius of the loop
    :param trim: gap between the beginning/arrow-tail and the end/arrow-head of the loop, in degrees
    :param props: 'ratio' -> arrow-head/arrow-length ratio, 'length' -> arrow-length, the rest are passed to
        `ais.drawing.draw_circle()` and `ais.drawing.draw_arrow()`
    :return: the coordinates of the loop, of shape 3x100
    """
    check(ax, Axes).check(pos, ndarray).check(normal, ndarray)
    check(radius, float, lambda: radius > 0).check(trim, float, lambda: 0 <= trim <= 360)

    ratio = props.pop('ratio', .3)
    length = props.pop('length', 1.)
    arc = draw_arc(ax, normal, radius=radius, centre=pos, start=-90 + trim, end=270 - trim, **dict(props))
    draw_arrow(ax, arc, ratio=ratio, length=length, **dict(props))

    return arc


def draw_link(ax: Axes, a: ndarray, b: ndarray, *, trim: float = 0., warp: float = 1., upright: bool = False,
              **props) -> \
        ndarray:
    """
    Draws a circular/elliptic arrow between two points, meant for bidirectional edges in FSA-graphs.

    :param ax: plot axes
    :param a: start position of link
    :param b: end position of link
    :param trim: gap between both ends of the link and the star and end positions, in degrees
    :param warp: the degree to which it's bent into an ellipse
    :param upright: whether the link should always face "up", ie, the positive z-axis
    :param props: passed to `ais.drawing.draw_wedge`
    :return: the coordinates of the link, of shape 3x100
    """
    check(ax, Axes)
    check(a, ndarray, lambda: a.shape == (3,)).check(b, ndarray, lambda: b.shape == (3,))

    centre = (a + b) * .5
    radius = norm(b - a) * .5
    # making b's direction the opposite of a's ensures angle is exactly 180 deg, avoiding opposite links being drawn
    # over each other on the same side
    a_on_wedge, b_on_wedge = (a - centre), -(a - centre)

    return draw_wedge(ax, a_on_wedge, b_on_wedge, radius=radius, orig=centre, trim=trim, warp=warp, upright=upright,
                      **props)


def draw_wedge(ax: Axes,
               a: ndarray,
               b: ndarray,
               *,
               radius: float = 1.,
               orig: ndarray = ORIG,
               trim: float = 0.,
               warp: float = 1.,
               upright: bool = False,
               **props) -> ndarray:
    """
    Draws a segment of a circle/ellipse.

    :param ax: plot axes
    :param a: start position of wedge
    :param b: end position of wedge
    :param radius: radius of the wedge
    :param orig: centre of the wedge
    :param trim: gap between both ends of the link and the star and end positions, in degrees
    :param warp: the degree to which it's bent into an ellipse
    :param upright: whether the link should always face "up", ie, the positive z-axis
    :param props: 'ratio' -> arrow-head/arrow-length ratio, 'length' -> arrow-length, the rest are passed to
        `mpl_toolkits.mplot3d.axes3d.Axes3D.plot` and `ais.drawing.draw_arrow()`
    :return: the coordinates of the wedge, of shape 3x100
    """
    check(ax, Axes).check(a, ndarray, lambda: a.shape == (3,)).check(b, ndarray, lambda: b.shape == (3,))
    check(radius, float | float32, lambda: radius > 0)
    check(orig, ndarray, lambda: orig.shape == (3,)).check(warp, float).check(trim, float, lambda: 0 <= trim <= 360)

    arc = make_wedge(a, b, radius=radius, trim=trim, warp=warp, upright=upright) + orig[:, None]
    ratio = props.pop('ratio', .3)
    length = props.pop('length', 1.)
    ax.plot(*arc, **props)
    draw_arrow(ax, arc, ratio=ratio, length=length, **props)

    return arc


def draw_arc(ax: Axes,
             normal: ndarray = Z_AXIS,
             *,
             radius: float = 1.,
             centre: ndarray = ORIG,
             start: float = 0.,
             end: float = 360.,
             **props) -> ndarray:
    """
    Draws 3D arc at the base of the given vector, taken as the normal. If the vector is the origin, the normal is
    taken to be collinear with the z-axis

    :param ax: plot axes
    :param normal: direction orthogonal to the plane the arc is drawn on
    :param radius: radius of the arc
    :param centre: centre of the arc
    :param start: the start of the arc in degrees with 0/-360 being the x of the standard basis, anti-clockwise
    :param end: the end of the arc in degrees with 360/-360 being the x of the standard basis, anti-clockwise
    :param props: 'c' -> face colour, 'ec' -> edge colour, the rest are passed to
        `mpl_toolkits.mplot3d.art3d.Poly3DCollection` and `mpl_toolkits.mplot3d.axes3d.Axes3D.plot`
    :return: the coordinates of the wedge, of shape 3x100
    """
    check(ax, Axes)
    check(centre, ndarray, lambda: centre.shape == (3,))

    arc = make_arc(normal=normal, radius=radius, start=start, end=end) + centre[:, None]  # 3xN (DxP)

    fc = props.pop('c', TRANSP)
    ec = props.pop('ec', KISH)

    ax.add_collection3d(Poly3DCollection(arc.T[None, :, :], color=fc, **props))  # 3xN -> Nx3 -> 1xNx3 (VxPxD)

    if ec:
        props.pop('hatch', None)
        ax.plot(*arc, color=ec, **props)

    return arc


def draw_square(ax: Axes, normal: ndarray, side: float, *, centre: ndarray = ORIG, **props) -> ndarray:
    """
    Draws 3D-square at the base of the given vector, taken as the normal. If the vector is the origin, the normal is
    taken to be collinear with the z-axis

    :param ax: plot axes
    :param normal: direction orthogonal to the plane the square is drawn on
    :param side: length of the side of the square
    :param centre: position of the centre of the square
    :param props: 'c' -> face colour, 'ec' -> edge colour, the rest are passed to
        `mpl_toolkits.mplot3d.art3d.Poly3DCollection` and `mpl_toolkits.mplot3d.axes3d.Axes3D.plot`
    :return: the coordinates of the square, of shape 3x5
    """
    check(ax, Axes)
    square = make_square(normal, side, centre=centre)  # 3x5

    fc = props.pop('c', TRANSP)
    ec = props.pop('ec', KISH)

    ax.add_collection3d(Poly3DCollection(square.T[None, :, :], color=fc, **props))  # 3x5 -> 5x3 -> 1x5x3 (VxPxD)
    if ec:
        props.pop('hatch', None)
        ax.plot(*square, color=ec, **props)

    return square


def draw_vec(ax: Axes, head: ndarray, *, tail: ndarray = ORIG, **props) -> ndarray:
    """
    Draws vector as arrow. If its length is 1e-3 or less, it draws a small pentagon at the origin instead.

    :param ax: plot axes
    :param head: position of the head of the vector
    :param tail: position of the tail of the vector
    :param props: 'ratio' -> arrow-head/arrow-length ratio, 'orig_marker' -> shape for marker for very short vectors,
        'orig_marker_size' -> size of the marker for very short vectors, the rest are passed to
        `ais.drawing.draw_marker()` and `mpl_toolkits.mplot3d.axes3d.Axes3D.quiver`
    :return: the coordinates of the vector, centred on the origin, of shape 3
    """
    check(ax, Axes).check(head, ndarray, lambda: head.shape == (3,)).check(tail, ndarray, lambda: tail.shape == (3,))

    length = norm(head - tail)

    ratio = props.pop('ratio', .1 / (length or 1) ** (1 / 1.25))  # gives a reasonably-sized arrow

    if length <= 1e-3 * max(norm(head), norm(tail)):  # TODO DOCUMENT THIS
        return draw_marker(ax,
                           tail,
                           shape=props.pop('orig_marker', 'P'),
                           s=props.pop('orig_marker_size', 3),
                           **props)

    props['color'] = props.pop('c', K)
    # translates between head-plus-tail-points to tail-point-plus-segment
    # the alr = arrow_length/length...
    ax.quiver(*tail, *(head - tail), arrow_length_ratio=ratio / length, **props)

    # block needed because the line style inappropriately applies to the head of the arrow
    if props.pop('ls', '-') not in {'-', 'solid'}:  # draws solid arrow on top of non-solid one
        arrow_length = ratio
        arrow = arrow_length * (head - tail) / length
        ax.quiver(*(head - arrow), *arrow, arrow_length_ratio=1, **props)

    return head - tail


def draw_marker(ax: Axes, xyz: ndarray, shape: str = 'P', **props) -> Axes:
    """
    Draws tip of the given vector with a glyph.

    :param ax: plot axes
    :param xyz: the position of the marker
    :param shape: shape of the marker
    :param props: 's' -> size, 'e' -> edge colour, 'mew' -> edge width, the rest are passed to
        `mpl_toolkits.mplot3d.axes3d.Axes3D.plot`
    :return: plot axes
    """
    check(ax, Axes).check(xyz, ndarray, lambda: xyz.shape == (3,))

    props = dict(s=6, e=K, mew=.5) | props
    ax.plot(*xyz, marker=shape, markersize=props.pop('s'), mec=props.pop('e'), mew=props.pop('mew'), **props)
    return ax


def draw_arrow(ax: Axes, curve: ndarray, **props) -> Axes:
    """
    Draws a curved arrow.

    :param ax: plot axes
    :param curve: arrow's coordinates, of shape 3xN
    :param props: 'ratio' -> arrow-head/arrow-length ratio, 'length' -> arrow-length, 'c' -> colour, the rest are
        passed to `mpl_toolkits.mplot3d.axes3d.Axes3D.quiver`
    :return: plot axes
    """
    check(ax, Axes).check(curve, ndarray, lambda: curve.shape[0] == 3 and curve.size)

    step = 9
    curvex, curvey, curvez = curve
    xs, ys, zs = curvex[-1], curvey[-1], curvez[-1]
    us, vs, ws = (c[-1] - c[-step] for c in curve)

    length = props.pop('length', norm(stack([us, vs, ws])))
    ratio = props.pop('ratio', .1 / (length or 1) ** (1 / 1.25))  # gives a reasonably-sized arrow
    ax.quiver(*[xs, ys, zs],
              *[us, vs, ws],
              pivot='tip',
              normalize=True,
              length=length,
              arrow_length_ratio=ratio / length,
              color=props.pop('c', KISH),
              **props)

    return ax


# ---------------------------------------- DELEGATE FUNCTIONS ----------------------------------------------------------


_button_pressed = False


def _on_click(event: MouseEvent, fig: Figure):
    if fig.canvas.widgetlock.locked():
        return  # Doesn't do anything if the zoom/pan tools have been enabled.
    global _button_pressed
    if event.name in {'button_press_event'}:
        fig.canvas.set_cursor(Cursors.MOVE)
        _button_pressed = True
    elif event.name in {'motion_notify_event'} and _button_pressed:
        fig.canvas.set_cursor(Cursors.MOVE)
    elif event.name in {'button_release_event'}:
        _button_pressed = False
        fig.canvas.set_cursor(Cursors.POINTER)


def _draw_better_ticks(ax: Axes, limits: Limits, colour: Color) -> Axes:
    """
    Draws uncluttered ticks

    :param ax: plot axes
    :param limits: limits of the x, y, z axes
    :param colour: the colour of the ticks
    :return: plot axes
    """
    # sets the number of tick labels to 3 equally-spaced ones
    for set_ticks in [ax.set_xticks, ax.set_yticks, ax.set_zticks]:
        ticks = [limits[0], limits[0] + (limits[-1] - limits[0]) * .5, limits[-1]] if set_ticks == ax.set_zticks else \
            [limits[0] + (limits[-1] - limits[0]) * .5]
        set_ticks(ticks, [f'{x:-5.2f}' for x in ticks])

    for a in 'xyz':  # hides ticks and draws tick labels
        ax.tick_params(axis=a, color=TRANSP, labelcolor=colour)

    return ax


def num_to_subs(num: int) -> str:
    """ converts a number into a subscript string representation """

    check(num, int, lambda: 0 <= 0 <= len(SUBSCRIPTS))
    return ''.join(SUBSCRIPTS[int(digit)] for digit in str(num))


def num_to_supers(num: int) -> str:
    """ converts a number into a superscript string representation """
    check(num, int, lambda: 0 <= 0 <= len(SUPERSCRIPTS))
    return ''.join(SUPERSCRIPTS[int(digit)] for digit in str(num))


def choose_output_palette(outputs: ndarray) -> Fn[[float], Tuple[float, float, float, float]]:
    """ chooses colour palette for model outputs based on values (binary, polar, none) """
    check(outputs, ndarray, lambda: outputs.ndim == 1)

    palette = (NO_PALETTE if not len(outputs) else
               (lambda v: POLAR_PALETTE(.5 * (v + 1))) if outputs.min() < 0 else
               BINARY_PALETTE)

    return palette


RGBA: TypeAlias = Tuple[float, float, float, float]
