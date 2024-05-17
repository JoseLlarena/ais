from torch import tensor, diag

from ais.spectral import svd_of
from unittests import assert_close, rot3d


def test_computes_full_SVD():
    mat = rot3d(deg=60., axis='z') @ diag(tensor([1., 2., 3.])) @ rot3d(deg=90., axis='x')
    mat[(mat < 1e-9) & (mat > -1e-9)] = 0

    U, S, Vt = svd_of(mat, dim=3)

    assert_close(U @ S @ Vt, mat, rtol=1e-6)


def test_computes_dimensionality_truncated_SVD():
    mat = rot3d(deg=60., axis='z') @ diag(tensor([1., 2., 0.])) @ rot3d(deg=90., axis='x')
    mat[(mat < 1e-9) & (mat > -1e-9)] = 0

    U, S, Vt = svd_of(mat, dim=2)

    assert_close(U @ S @ Vt, mat, rtol=1e-6)


def test_computes_singular_value_truncated_SVD():
    mat = rot3d(deg=60., axis='z') @ diag(tensor([1., 2., 0.])) @ rot3d(deg=90., axis='x')
    mat[(mat < 1e-9) & (mat > -1e-9)] = 0

    U, S, Vt = svd_of(mat, dim=-1, tol=1e-12)

    assert_close(U @ S @ Vt, mat, rtol=1e-6)
