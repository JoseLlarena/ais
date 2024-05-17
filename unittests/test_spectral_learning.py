from ais import px, outm
from ais.boolean_data import XOR, make_bool_data, BOOLS
from ais.data import class_target, as_dataset
from ais.spectral import learn_wfsa
from ais.training import zero_one_loss, compute_loss


def test_learns_a_wfsa_with_svd():
    t_data = tuple(make_bool_data(XOR, max_len=3))
    wfsa = learn_wfsa(kind='binary', data=t_data, basis='freq-rank=1:1', algo='svd')

    v_data = make_bool_data(XOR, min_len=4, max_len=5)
    dataset = as_dataset(v_data, px(class_target, task=XOR), x_vocab=BOOLS, y_vocab=BOOLS, kind='class')
    loss = compute_loss(wfsa, dataset, px(zero_one_loss, tol=1e-3))

    assert loss == 0.


def test_learns_a_wfsa_with_nmf():
    t_data = tuple(make_bool_data(XOR, max_len=3))
    wfsa = learn_wfsa(kind='binary', data=t_data, basis='freq-rank=1:1', algo='nmf')

    v_data = make_bool_data(XOR, min_len=4, max_len=5)
    dataset = as_dataset(v_data, px(class_target, task=XOR), x_vocab=BOOLS, y_vocab=BOOLS, kind='class')
    loss = compute_loss(wfsa, dataset, px(zero_one_loss, tol=1e-3))

    assert loss == 0.
