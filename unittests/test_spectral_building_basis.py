from ais.boolean_data import XOR, make_bool_data, AND
from ais.spectral import build_basis


def test_builds_basis_by_frequency_rank():
    data = make_bool_data(XOR, max_len=4)

    prefixes = ((),
                (False,),
                (True,),

                (False,),
                (False, False),
                (True, False),

                (True,),
                (False, True),
                (True, True))

    suffixes = (), (False,), (True,)

    assert build_basis(data, 'freq-rank=2:2') == (prefixes, suffixes)


def test_builds_basis_by_length():
    data = make_bool_data(XOR, max_len=3)

    prefixes = ((),
                (False,),
                (True,),

                (False,),
                (False, False),
                (True, False),

                (False, False,),
                (False, False, False),
                (True, False, False),

                (False, True,),
                (False, False, True),
                (True, False, True),

                (True,),
                (False, True),
                (True, True),

                (True, False,),
                (False, True, False),
                (True, True, False),

                (True, True,),
                (False, True, True),
                (True, True, True))

    suffixes = ((),
                (False,),
                (True,),
                (False, False),
                (False, True),
                (True, False),
                (True, True),
                (False, False, True),
                (False, True, False),
                (True, False, False),
                (True, True, True))

    assert build_basis(data, 'length=2:3') == (prefixes, suffixes)


def test_finds_basis_by_frequency_rank_with_base_vocab():
    data = make_bool_data(AND, max_len=4)

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

    assert build_basis(data, 'freq-rank=2:2', base_vocab={False}) == (prefixes, suffixes)


def test_finds_basis_by_length_with_base_vocab():
    data = make_bool_data(AND, max_len=3)

    prefixes = ((),
                (False,),
                (True,),

                (False,),
                (False, False),
                (True, False),

                (True,),
                (False, True),
                (True, True),

                (True, True),
                (False, True, True),
                (True, True, True))

    suffixes = ((), (False,), (True,), (True, True))

    assert build_basis(data, 'length=2:2', base_vocab={False}) == (prefixes, suffixes)
