from torch import eye, tensor, empty

from ais import make_deterministic
from ais.boolean_data import make_bool_data, AND, OR, make_bool_partitions, XOR, make_dataset_partitions
from ais.data import RaggedSupeDataset


def test_exhaustively_generates_boolean_data():
    data = tuple(make_bool_data(AND, max_len=3, n=-1))

    assert data == ((), (True,), (True, True), (True, True, True))


def test_samples_boolean_data():
    make_deterministic()
    data = tuple(make_bool_data(OR, min_len=5, max_len=7, n=3))

    assert data == ((False, False, False, False, True),
                    (False, False, False, True, False),
                    (False, False, False, True, True))


def test_makes_boolean_data_partitions():
    make_deterministic()
    training, validation, test = map(tuple, make_bool_partitions(XOR, max_lengths=(3, 4, 5), ns=(2, 2, 2), supe=False))

    assert training == ((True,), (False, True))
    assert validation == ((False, False, False, False), (False, False, False, True))
    assert test == ((False, False, False, False, False), (False, False, False, False, True))


def test_makes_boolean_dataset_partitions():
    make_deterministic()
    t, v, e, x_vocab, y_vocab = make_dataset_partitions(XOR, max_lengths=(3, 4, 5), ns=(2, 2, 2), scheme='class')

    F, T = eye(2)
    assert t == RaggedSupeDataset([(empty(0, 2), F), (tensor([[1, 0.]]), F)])
    assert v == RaggedSupeDataset([(tensor([[1, 0.]] * 4), F), (tensor([[1, 0.]] * 3 + [[0, 1]]), T)])
    assert e == RaggedSupeDataset([(tensor([[1, 0.]] * 5), F), (tensor([[1, 0.]] * 4 + [[0, 1]]), T)])
    assert x_vocab == (False, True)
    assert y_vocab == (False, True)
