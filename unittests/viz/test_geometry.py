from numpy import asarray

from ais.viz.geometry import Z_AXIS, make_arc, Y_AXIS, make_wedge, X_AXIS, make_square, find_ortho, find_normal, \
    compute_dim_limits, angle_of
from unittests import assert_close


def test_makes_arc():
    normal = Z_AXIS

    arc = make_arc(normal)

    assert arc.shape[-1] == 100
    for point in arc.T:
        assert normal @ point == 0


def test_makes_wedge():
    wedge = make_wedge(u=X_AXIS, v=Y_AXIS)

    assert wedge.shape[-1] == 100

    normal = Z_AXIS
    for point in wedge.T:
        assert normal @ point == 0


def test_makes_square():
    normal = Z_AXIS

    square = make_square(normal, side=1.)

    assert square.shape[-1] == 5
    for point in square.T:
        assert normal @ point == 0


def test_finds_orthogonal_vector():
    assert_close(find_ortho(u=X_AXIS, v=Y_AXIS), Z_AXIS)
    assert_close(find_ortho(u=Y_AXIS, v=X_AXIS), -Z_AXIS)


def test_finds_normal_vector():
    assert_close(find_normal(u=X_AXIS, v=Y_AXIS), Z_AXIS)
    assert_close(find_normal(u=Y_AXIS, v=X_AXIS), Z_AXIS)


def test_computes_angle():
    assert angle_of(u=X_AXIS, v=Y_AXIS) == 90
    assert angle_of(u=Y_AXIS, v=X_AXIS) == 270
    assert angle_of(u=X_AXIS, v=X_AXIS) == 0


def test_computes_dimensional_limits():
    assert_close(compute_dim_limits(asarray([[1, -1, 0],
                                             [0, 0., 2],
                                             [-2, -1, 1]])),
                 asarray([[-1, 1],
                          [0., 2.],
                          [-2, 1.]]))
