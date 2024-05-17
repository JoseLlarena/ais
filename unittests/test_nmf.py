from torch import tensor

from ais.spectral import nmf_of
from unittests import assert_close


def test_computes_full_nmf():
    mat = tensor([1., 2., 3]).diag() @ tensor([4., 5., 6.]).diag().fliplr()

    E, D = nmf_of(mat, dim=3)

    assert_close(E @ D, mat, rtol=1e-6)


def test_computes_dimensionality_truncated_nmf():
    mat = tensor([1., 2., 0]).diag() @ tensor([4., 5., 6.]).diag()

    E, D = nmf_of(mat, dim=2)

    assert_close(E @ D, mat, rtol=1e-6)


def test_computes_svd_initialised_nmf():
    mat = tensor([1., 2., 3]).diag() @ tensor([4., 5., 6.]).diag().fliplr()

    E, D = nmf_of(mat, dim=3, init='svd')

    assert_close(E @ D, mat, rtol=1e-6)


def test_computes_singular_value_truncated_nmf():
    mat = tensor([1., 2., 0]).diag() @ tensor([4., 5., 6.]).diag()

    E, D = nmf_of(mat, dim=-1, tol=1e-9)

    assert_close(E @ D, mat, rtol=1e-6)
    assert tuple(E.shape) == (3, 2)
    assert tuple(D.shape) == (2, 3)


def test_computes_svd_initialised_singular_value_truncated_nmf():
    mat = tensor([1., 2., 0]).diag() @ tensor([4., 5., 6.]).diag()

    E, D = nmf_of(mat, dim=-1, tol=1e-9, init='svd')

    assert_close(E @ D, mat, rtol=1e-6)
    assert tuple(E.shape) == (3, 2)
    assert tuple(D.shape) == (2, 3)
