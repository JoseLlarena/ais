"""
Primitive geometric operations and shape creation to be used by drawing modules
"""
from typing import Tuple, TypeAlias

from numpy import arccos, degrees, linspace, ndarray, cross, sin, cos, radians, outer, clip, zeros, eye, gradient, \
    asarray, stack, float32
from numpy.linalg import norm

from ais import check
from ais.viz import vec

Limits: TypeAlias = Tuple[float, float] | ndarray
Vertex: TypeAlias = Tuple[float, float, float]

[X_AXIS, Y_AXIS, Z_AXIS], ORIG = eye(3), zeros(3)
UP, FRONT, RIGHT = Z_AXIS, -Y_AXIS, X_AXIS


def make_arc(normal: ndarray, *, radius: float = 1., start: float = 0., end: float = 360.) -> ndarray:
    """
    Returns the coordinates of a 3D-arc

    :param normal: direction orthogonal to the plane the arc is in
    :param radius: the arc's radius
    :param start: the start of the arc in degrees with 0/-360 being the x of the standard basis, anti-clockwise
    :param end: the end of the arc in degrees with 360/-360 being the x of the standard basis, anti-clockwise
    :return: the arc's coordinates with shape 3x100
    """
    check(normal, ndarray, lambda: normal.shape == (3,)).check(radius, float, lambda: radius > 0, f'{radius}')
    check(start, float, lambda: -360 <= start <= 360, f'{start}')
    check(end, float, lambda: -360 <= end <= 360 and end >= start, f'{start} -> {end}')

    three_oclock = cross(Z_AXIS, normal)
    u, v = (u := normed(three_oclock) if not is_origin(three_oclock) else X_AXIS), find_ortho(u, normal)
    thetas = linspace(radians(start), radians(end), num=100)
    arc = radius * (outer(u, cos(thetas)) + outer(v, -sin(thetas)))

    return arc


def make_wedge(u: ndarray, v: ndarray, *, radius: float = 1., trim: float = 0., warp: float = 1., upright: bool = False) \
        -> ndarray:
    """
    Returns an arc's coordinates, part of a circle or ellipse

    :param u: the starting direction of the arc
    :param v: the end point of the arc
    :param radius: the arc's radius
    :param trim: the gap between the start and end directions, in degrees
    :param warp: the amount of warping in the arc, turning it into a segment of an ellipse, a negative value flips it
    :param upright: whether the arc should always face "up", ie, the positive z-axis
    :return: the arc's coordinates with shape 3x100
    """
    check(u, ndarray, lambda: u.size == 3 and not is_origin(u), f'{u}')
    check(v, ndarray, lambda: v.size == 3 and not is_origin(v), f'{v}')
    check(radius, float | float32, lambda: radius > 0).check(trim, float, lambda: 0 <= trim < 360, f'{trim}')
    check(warp, float)

    u = normed(u)
    v = normed(v)
    w = find_ortho(u, find_normal(u, v) if upright else find_ortho(u, v))

    a = angle_of(u, v) if upright else degrees(arccos(clip(u @ v, -1, 1))).item()
    thetas = linspace(radians(trim), radians(a - trim), num=100)

    return radius * (outer(u, cos(thetas)) + outer(w * warp, -sin(thetas)))


def make_square(normal: ndarray, side: float, *, centre: ndarray = ORIG) -> ndarray:
    """
    Returns the coordinates of a 3D-square

    :param normal: direction orthogonal to the plane the square is in
    :param side: size of the square's sides
    :param centre: the coordinates of the centre of the square
    :return: the square's coordinates with shape 3x5
    """
    check(normal, ndarray, lambda: normal.shape == (3,) and normal.size)
    check(side, float | float32, lambda: side > 0, f'{side}')
    check(centre, ndarray, lambda: centre.shape == (3,) and centre.size)

    u, v = (u := find_ortho(normal), find_ortho(normal, u))

    return asarray([centre + (u + v) * side * .5 for u, v in ([-u, -v], [u, -v], [u, v], [-u, v], [-u, -v])]).T


def find_ortho(u: ndarray, v: ndarray | None = None, *, unit: bool = True) -> ndarray:
    """
    Finds an orthogonal vector to u and, if given, also v. Algorithm:

    1. if v is not collinear to u, return the cross-product of u and v; else
    2. if u lies on the z-axis, return the negative y-axis; else
    3. return a vector that is orthogonal both to u and the positive z-axi

    This algorithm has a bias such that the orthogonal faces towards the "front" and "right" whenever possible,
    ie, towards Matplotlib's initial default viewing angle,  the one tended by the negative y-axis with the
    positive x-axis. The bias makes it consistent with the disambiguation rules in `angle_of()`.

    :param u: first 3D-vector to find an orthogonal vector for
    :param v: second 3D-vector, to find an orthogonal vector to; default: None
    :param unit: True if the returned orthogonal vector should have norm 1, False otherwise; default: True
    :return: a possibly normalised orthogonal vector to u, and if not None, also to v
    """

    check(u, ndarray, lambda: u.size == 3 and not is_origin(u), f'{u}')
    check(v, ndarray | None, lambda: v is None or (v.size == 3 and not is_origin(v)), f'{v}')
    v = u if v is None else v

    if is_origin(cross(u, v)):  # u x v = (0, 0, 0) means that u and v are collinear

        if is_origin(cross(u, Z_AXIS)):
            return -Y_AXIS

        v = cross(u, Z_AXIS)
        if v @ Z_AXIS <= 0:
            v *= -1.

    return normed(cross(u, v)) if unit else cross(u, v)


def find_normal(u: ndarray, v: ndarray | None = None, *, unit: bool = True) -> ndarray:
    """
    Finds the positive ("up") normal vector to the given u vector and if given also the v vector. Follows a similar
    logic as `find_ortho()`.

    :param u: first 3D-vector to find the normal for
    :param v: second 3D-vector, to find the normal for; default: None
    :param unit: True if the returned normal should have norm 1, False otherwise; default: True
    :return: a possibly normalised normal vector to u, and if not None, also to v
    """
    ortho = find_ortho(u, v, unit=unit)
    angle = angle_of(u, v) if v is not None and not is_origin(v) else 0.

    if angle > 180:
        return -ortho

    return ortho


def angle_of(u: ndarray, v: ndarray) -> float:
    """
    Calculates the counter-clockwise angle in degrees between two 3D-vectors in the [0, 360) interval.

    As the angle depends on which orientation we look at the uv-plane from, there's an inherent ambiguity,
    resolved here by adopting the convention that the plane is always facing "up", meaning the orientation of the
    positive z-axis. This is to make visualisation easier as planes facing "down" force the viewer to flip the 3D axes
    upside down from their conventional representation where the positive z-axis is pointing "up". This effectively
    flips the right-hand rule to the left-hand rule when the angle is greater than 180 degrees.

    There's a further ambiguity when the uv-plane is perpendicular to the xy-plane. In that case, the convention adopted
    here is that the uv-plane is facing "front", meaning the negative Y-axis, as that is the conventional direction from
    which it is viewed.

    Lastly, there's a final ambiguity when the plane spanned by u and v is exactly the yz-plane. In that case, the
    convention implemented is that the plane is facing "right", meaning the direction of the positive X-axis, as this is
    the default direction Matplotlib places the initial viewing angle from.

    :param u: the starting direction of the angle
    :param v: the end direction of the angle
    :return: the angle between `u` and `v` in degrees
    """
    check(u, ndarray, lambda: u.size == 3 and not is_origin(u), f'{u}')
    check(v, ndarray, lambda: v.size == 3 and not is_origin(v), f'{v}')

    u = normed(u)
    v = normed(v)
    # clipping because the normalisation of u/v can end up outside [-1, 1] where arccos is undefined
    deg = degrees(arccos(clip(u @ v, -1, 1))).item()

    normal = cross(u, v)
    if normal @ UP < 0:
        return 360 - deg
    elif normal @ UP > 0:
        return deg
    else:
        if normal @ FRONT < 0:
            return 360 - deg
        elif normal @ FRONT > 0:
            return deg
        else:
            if normal @ RIGHT < 0:
                return 360 - deg
            return deg


def is_origin(u: ndarray) -> bool:
    check(u, ndarray, lambda: u.size == 3)
    return not u.any()


def normed(u: ndarray, axis: int | None = None) -> ndarray:
    check(u, ndarray, lambda: u.size > 0)
    return u / (norm(u, axis=axis) or 1)


def tangent_of(curve: ndarray, t: int) -> ndarray:
    xx, yy, zz = (gradient(d) for d in curve)

    return vec(xx[t], yy[t], zz[t])


def compute_dim_limits(points: ndarray) -> ndarray:
    """ computes limits of point cloud for each dimension """
    check(points, ndarray, lambda: points.ndim == 2 and points.size, lambda: f'{points.shape}')  # assumes DxN

    return stack(tuple(map(compute_limits, points)))  # Dx2


def compute_limits(points: ndarray) -> ndarray:
    """ computes limits of point cloud across dimensions """
    check(points, ndarray, lambda: points.ndim and points.size, lambda: f'{points.shape}')

    bottom, top = limits = (points.min(), points.max())
    if bottom == top:
        limits = bottom - .5 * bottom, bottom + .5 * bottom

    return asarray(limits)  # 2
