from torch import eye, tensor, empty

from ais import make_deterministic, ONE, ZERO, BOS
from ais.boolean_data import make_bool_data, AND, BOOLS, TRUE, XOR
from ais.data import RaggedSupeDataset, as_dataset, one_hot_coder_from, id_coder_from, polar_coder_from, \
    binary_coder_from, format_input, format_target, bert_target, class_target, seq2seq_target, make_loader


def test_makes_dataset():
    make_deterministic()
    data = tuple(make_bool_data(TRUE, max_len=2, n=-1))
    dataset = as_dataset(data, lambda seq: [AND(seq)], BOOLS, BOOLS, kind='class-id')

    F, T = eye(2)
    assert dataset == RaggedSupeDataset([(empty(0, 2), T),

                                         (tensor([0.]), F),
                                         (tensor([1.]), T),

                                         (tensor([0, 0.]), F),
                                         (tensor([0, 1.]), F),
                                         (tensor([1, 0.]), F),
                                         (tensor([1, 1.]), T)])


def test_constructs_a_one_hot_coder():
    coder = one_hot_coder_from(BOOLS)

    seq = True, True, False
    assert coder.untensorise(coder.tensorise(seq)) == seq


def test_constructs_an_id_coder():
    coder = id_coder_from(BOOLS)

    seq = True, True, False
    assert coder.untensorise(coder.tensorise(seq)) == seq


def test_constructs_a_polar_coder():
    coder = polar_coder_from(BOOLS)

    seq = True, True, False
    assert coder.untensorise(coder.tensorise(seq)) == seq


def test_constructs_a_binary_coder():
    coder = binary_coder_from(zero=ZERO, one=ONE)

    seq = ZERO, ZERO, ONE
    assert coder.untensorise(coder.tensorise(seq)) == seq


def test_formats_input():
    assert format_input('abc') == tuple('abc')


def test_formats_padded_input():
    assert format_input('abc', length=4, pad='*') == tuple('abc*')


def test_formats_bos_input():
    assert format_input('abc', bos='^') == tuple('^abc')


def test_formats_bos_padded_input():
    assert format_input('abc', length=5, pad='*', bos='^') == tuple('^abc*')


def test_formats_eos_input():
    assert format_input('abc', eos='$') == tuple('abc$')


def test_formats_target():
    target = lambda seq: [XOR(seq)]

    assert format_target(target, kind='seq2seq-pad', length=3, pad='*')([True, False]) == (True, '*', '*')


def test_makes_bert_target_function():
    assert bert_target([BOS, True, False], XOR) == (True,)


def test_makes_classification_target_function():
    assert class_target([True, False], XOR) == (True,)


def test_makes_seq2seq_target_function():
    assert seq2seq_target([BOS, True, False], XOR) == (False, True, True)


def test_makes_loader():
    F, T = eye(2)
    dataset = RaggedSupeDataset([(tensor([[1, 0.]] * 4), F),
                                 (tensor([[1, 0.]] * 3 + [[0, 1]]), T),
                                 (tensor([[1, 0.]] * 5), F),
                                 (tensor([[1, 0.]] * 4 + [[0, 1]]), T)])

    batch_1, batch_2 = make_loader(dataset, bs=-1, shuffled=False)

    assert batch_1[0][0].shape[0] == 5
    assert batch_1[0][1].shape[0] == 5

    assert batch_2[0][0].shape[0] == 4
    assert batch_2[0][1].shape[0] == 4
