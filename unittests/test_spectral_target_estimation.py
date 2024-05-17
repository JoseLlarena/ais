from ais.boolean_data import XOR, make_bool_data
from ais.spectral import estimate_targets

data = tuple(make_bool_data(XOR, max_len=3))


def test_estimates_binary_targets():
    targets = estimate_targets(data, kind='binary')
    assert targets == {(True,): 1,
                       (False, True): 1,
                       (True, False): 1,
                       (False, False, True): 1,
                       (False, True, False): 1,
                       (True, False, False): 1,
                       (True, True, True): 1}

    assert targets[(False,)] == 0


def test_estimates_polar_targets():
    targets = estimate_targets(data, kind='polar')
    assert targets == {(True,): 1,
                       (False, True): 1,
                       (True, False): 1,
                       (False, False, True): 1,
                       (False, True, False): 1,
                       (True, False, False): 1,
                       (True, True, True): 1}

    assert targets[(False,)] == -1


def test_estimates_lm_targets():
    targets = estimate_targets(data, kind='lm')
    assert targets == {(True,): 1 / 7,
                       (False, True): 1 / 7,
                       (True, False): 1 / 7,
                       (False, False, True): 1 / 7,
                       (False, True, False): 1 / 7,
                       (True, False, False): 1 / 7,
                       (True, True, True): 1 / 7}

    assert targets[(False,)] == 0
