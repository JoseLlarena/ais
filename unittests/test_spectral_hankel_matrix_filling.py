from torch import tensor

from ais import outm, ZERO, EPS, ONE
from ais.boolean_data import make_bool_data, AND
from ais.spectral import fill_hankel
from unittests import assert_close


def test_fills_in_binary_hankel_matrix():
    print()
    prefixes = ((),
                (False,),
                (True,),

                (False,),
                (False, False),
                (True, False),

                (True,),
                (False, True),
                (True, True))

    suffixes = ((), (False,), (True,))
    targets = dict.fromkeys(make_bool_data(AND, max_len=4), 1)

    actual = fill_hankel(targets, prefixes, suffixes)

    outm(actual, tubes=(EPS, ZERO, ONE), rows=(EPS, ZERO, ONE), cols=(EPS, ZERO, ONE))
    expected = tensor([
        [[1, 0, 1],
         [0, 0, 0],
         [1, 0, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[1, 0, 1],
         [0, 0, 0],
         [1, 0, 1]]]) * 1.
    assert_close(actual, expected, rtol=0, atol=0)
