from collections.abc import Iterable

from torch import eye, tensor

from ais import px, outm
from ais.boolean_data import XOR, make_bool_data, BOOLS, OR
from ais.data import class_target, as_dataset
from ais.spectral import learn_wfsa
from ais.training import zero_one_loss, compute_loss


def test_learns_an_XOR_wfsa_with_svd():
    print()
    t_data = tuple(make_bool_data(XOR, max_len=3))
    wfsa = learn_wfsa(kind='binary', data=t_data, basis='freq-rank=1:1', algo='svd')

    v_data = make_bool_data(XOR, min_len=4, max_len=5)
    dataset = as_dataset(v_data, px(class_target, task=XOR), x_vocab=BOOLS, y_vocab=BOOLS, kind='class')

    assert compute_loss(wfsa, dataset, px(zero_one_loss, tol=1e-3)) == 0.


def test_learns_a_wfsa_with_svd():
    print()
    t_data = tuple(make_bool_data(OR, max_len=3))
    wfsa = learn_wfsa(kind='binary', data=t_data, basis='freq-rank=1:1', algo='svd')

    v_data = make_bool_data(OR, min_len=4, max_len=5)
    dataset = as_dataset(v_data, px(class_target, task=OR), x_vocab=BOOLS, y_vocab=BOOLS, kind='class')
    loss = compute_loss(wfsa, dataset, px(zero_one_loss, tol=1e-3))

    assert loss == 0.


def MAJ(booleans: Iterable[bool]) -> bool:
    bools = tuple(booleans)

    return bools.count(1) > len(bools) // 2


def test_learns_majority():
    print()
    t_data = tuple(make_bool_data(MAJ, max_len=10, n=500))
    wfsa = learn_wfsa(kind='binary', data=t_data, basis='freq-rank=7:6', algo='svd')
    print(wfsa.initial)
    v_data = make_bool_data(MAJ, max_len=10, n=100)
    dataset = as_dataset(v_data, px(class_target, task=MAJ), x_vocab=BOOLS, y_vocab=BOOLS, kind='class')
    loss = compute_loss(wfsa, dataset, px(zero_one_loss, tol=1e-3))

    assert loss == 0.


def MOD(booleans: Iterable[bool], mod: int = 2) -> bool:
    return (sum(booleans) % mod)


def test_modulo_4():
    print()
    MOD3 = px(MOD, mod=3)
    print(MOD3([True, True, True]))
    print(MOD3([True, True, True, False]))

    print(MOD3([True, True, False]))
    print(MOD3([True, True, False, False]))
    print(MOD3([True, True, True, True]))


def test_group():
    v = tensor([[1, 0],
                [0, 2],
                [-1, 0],
                [0, -2.]]).T

    outm(v, fr='vertices')

    a = eye(2)
    a[1] *= -1

    b = eye(2)
    b[0] *= -1

    c = eye(2)
    c *= -1

    for M in a, b, c:
        outm(M)
        outm(M @ v)
        outm(M @ (M @ v), sep='>>>')
