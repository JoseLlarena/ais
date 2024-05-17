from math import radians, cos, sin

from numpy import ndarray, cos, sin
from torch import Tensor, tensor, float32, from_numpy
from torch.testing import assert_close as torch_assert_close

from ais import check


def assert_close(actual: Tensor | ndarray, expected: Tensor | ndarray, rtol: float = 0, atol: float = 0):
    print('\nACTUAL  :', actual, 'EXPECTED:', expected, sep='\n')
    torch_assert_close(actual if isinstance(actual, Tensor) else from_numpy(actual),
                       expected if isinstance(expected, Tensor) else from_numpy(expected),
                       rtol=rtol,
                       atol=atol)


def rot3d(deg: float = 90., axis: str = 'z') -> Tensor:
    """
    Returns rotation matrix by an angle of `deg` degrees about the `axis` vector

    :param deg:
    :param axis:
    :return:
    """
    check(deg, float, lambda: 0 <= deg <= 360).check(val=axis in tuple('xyz'))

    theta = radians(deg)

    if axis == 'x':
        return tensor([[1, 0, 0],
                       [0, cos(theta), -sin(theta)],
                       [0, sin(theta), cos(theta)]], dtype=float32)  # anti/clockwise deg rotations about x axis

    if axis == 'y':
        return tensor([[cos(theta), 0, sin(theta)],
                       [0, 1, 0],
                       [-sin(theta), 0, cos(theta)]], dtype=float32)  # anti/clockwise deg rotations about y axis

    return tensor([[cos(theta), -sin(theta), 0],  # anti/clockwise deg rotations about z axis
                   [sin(theta), cos(theta), 0],
                   [0, 0, 1]], dtype=float32)
