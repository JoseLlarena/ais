"""
Functions and constants to generate boolean data
"""
from collections.abc import Sequence, Iterable
from functools import wraps, reduce
from itertools import product, chain
from operator import xor
from typing import Tuple, TypeAlias

from more_itertools import flatten

from ais import Fn, check, BOS, PAD, px, EOS
from ais.data import bert_target, seq2seq_target, as_dataset, DATA, class_target, format_input, format_target

Boolean: TypeAlias = Fn[[bool, bool], bool]
Varboolean: TypeAlias = Fn[Iterable[bool], bool]

BOOLS = False, True
ZERO, ONE = '𝟘', '𝟙'
bool_to_str = {False: ZERO, True: ONE}
str_to_bool = {ZERO: False, ONE: True}
SCHEMES = {'bert', 'bert-pad', 'bert-id', 'bert-id-pad', 'bert-pad-noextra',
           'class', 'class-pad', 'class-id', 'class-id-pad',
           'class-bos', 'class-bos-pad', 'class-bos-id', 'class-bos-id-pad', 'class-bos-noextra',
           'seq2seq-bos', 'seq2seq-bos-pad', 'seq2seq-bos-pad-noextra', 'seq2seq-bos-noextra'}


def _as_variadic(fn: Boolean, *, base_case: bool | None = False, name: str = '') -> Varboolean:
    """
    Converts a binary boolean function into a variadic one. `base_case` provides the output value for the nullary
    version of the function. 'None' supports the scenario where the boolean function does not lend itself to base-case
    recursion, in which case, the nullary version arbitrarily returns False and the unary version returns the only item
    in the input.

    :param fn: the binary function to convert
    :param base_case: the output value of the nullary version of the function
    :param name: the name to set the function for display purposes
    :return: the variadic function
    """

    @wraps(fn)
    def recurse(expression: Iterable[bool]):
        if base_case is None:

            expression = tuple(expression)

            if not expression:
                return False

            return reduce(fn, expression)

        return reduce(fn, expression, base_case)

    if name:
        recurse.__name__ = name

    return recurse


def _neg(fn: Varboolean, name: str = '') -> Varboolean:
    """ negates the given boolean function """

    def neg(expression):
        return not fn(expression)

    if name:
        neg.__name__ = name

    return neg


def _copy(fn: Varboolean, name: str) -> Varboolean:
    """ copies (by wrapping) the given boolean function and gives the provided name"""

    def clone(*args, **kwargs):
        return fn(*args, **kwargs)

    clone.__name__ = name

    return clone


TRUE = _copy(lambda *args, **kwargs: True, name='TRUE') # 1 dim
FALSE = _neg(TRUE, name='FALSE') # 1 dim
XOR = _as_variadic(xor, name='XOR')  # 2  dim
EQUIV = _neg(XOR, name='EQUIV')  # 2 dim
OR = _copy(any, name='OR')  # 2 dim
NOR = _neg(OR, name='NOR')  # 1 dim
AND = _copy(all, name='AND')  # 1 dim
NAND = _neg(AND, name='NAND')  # 2 dim
IF = _as_variadic(lambda left, right: (not left) or right, base_case=True, name='IF')  # 2 dim
NIF = _neg(IF, name='NIF')  # 2 dim
FIRST = _as_variadic(lambda left, right: left, base_case=None, name='FIRST')  # 2 dim
NFIRST = _neg(FIRST, name='NFIRST')  # 2 dim
LAST = _as_variadic(lambda left, right: right, name='LAST')  # 2 dim
NLAST = _neg(LAST, name='NLAST')  # 2 dim
INV = _as_variadic(lambda left, right: (not left) and right, name='INV')  # 2 dim
NINV = _neg(INV, name='NINV')  # 2 dim


def make_dataset_partitions(task: Varboolean,
                            max_lengths: Sequence[int],
                            ns: Sequence[int],
                            scheme: str,
                            min_len: int = 0) \
        -> Tuple[DATA, DATA, DATA, Tuple[bool | str, ...], Tuple[bool | str, ...]]:
    """
    Generates training, validation and testing partitions of a boolean dataset for the given task, plus the input and
    output vocabularies. The `scheme` parameter determines the input and target encodings:

        target type: single token ('class' and 'bert') vs sequence of tokens ('seq2seq')
        'bos': whether the input sequence should be preceded by a beginning-of-sequence token
        'pad': whether the input and possibly output sequence should be padded to a maximum length, given in max_lengths
        'id': whether the input should be encoded as integer-ids or as one-hot vectors
        'noextra': whether validation and test partitions should have the same maximum length as the training one

    :param task: the boolean function to train, validate and test
    :param max_lengths: the maximum lengths of each partition; assumes validation examples are longer than training ones
        and testing ones longer than both
    :param ns: the number of examples for each partition
    :param scheme: the dataset format corresponding to different input and target encodings
    :param min_len: the minimum length of the training examples
    :return: training, validation and test datasets plus input and output vocabularies
    """
    check(task, Fn)
    check(max_lengths, Sequence, lambda: len(max_lengths) == 3).check(ns, Sequence, lambda: len(ns) == 3)
    check(val=scheme in SCHEMES, msg=f'expected one of {SCHEMES} but found [{scheme}]')
    check(min_len, int, lambda: min_len >= 0)

    context_size, pad, bos, eos, kind, target, eps = -1, None, None, None, 'class', None, True
    x_vocab, y_vocab = BOOLS, BOOLS

    if scheme.startswith('seq2seq'):
        kind = 'seq2seq'

    if '-pad' in scheme:
        pad = PAD
        x_vocab = x_vocab + (PAD,)
        context_size = max(max_lengths)
        kind += '-pad'

    if '-bos' in scheme or scheme.startswith('bert'):
        bos = BOS
        x_vocab = (BOS,) + x_vocab
        max_lengths = tuple(max_len - 1 for max_len in max_lengths)

    if '-eos' in scheme:
        eos = EOS
        x_vocab = x_vocab + (EOS,)
        max_lengths = tuple(max_len - 1 for max_len in max_lengths)

    if '-id' in scheme:
        kind += '-id'

    if scheme.startswith('bert'):
        target = px(bert_target, task=task)

    elif scheme.startswith('class'):
        target = px(class_target, task=task, bos=bos)

    elif scheme.startswith('seq2seq'):
        target = px(seq2seq_target, task=task, bos=bos)

    formatting = px(format_input, length=context_size, bos=bos, eos=eos, pad=pad)
    target = format_target(target, kind=kind, length=context_size, pad=pad)

    if '-noextra' in scheme:
        check(val=-1 not in ns).check(val=len(set(max_lengths)) == 1)
        data, *_ = make_bool_partitions(task, [max_lengths[0], 0, 0], [sum(ns), 0, 0], supe=True, min_len=min_len)
        partitions = as_dataset(map(formatting, data), target, x_vocab, y_vocab, kind).chunk(3)

    else:
        t_data, v_data, e_data = make_bool_partitions(task, max_lengths, ns, supe=True, min_len=min_len)
        partitions = (as_dataset(map(formatting, data), target, x_vocab, y_vocab, kind)
                      for data in [t_data, v_data, e_data])

    return tuple(partitions) + (x_vocab, y_vocab)


def make_bool_partitions(task: Varboolean,
                         max_lengths: Sequence[int],
                         ns: Sequence[int],
                         supe: bool = False,
                         min_len: int = 0) \
        -> Tuple[Iterable[Tuple[bool, ...]], Iterable[Tuple[bool, ...]], Iterable[Tuple[bool, ...]]]:
    """
    Generates training, validation and testing partitions of boolean data for the given task. If `supe` is False, the
    training set will contain only inputs that are true according to the given `task`.

    :param task: the boolean function to train, validate and test
    :param max_lengths: the maximum lengths of each partition; assumes validation examples are longer than training ones
        and testing ones longer than both
    :param ns: the number of examples for each partition
    :param supe: whether the partitions are meant for supervised or unsupervised learning
    :param min_len: the minimum length of the training examples
    :return: the three partitions as collections of boolean sequences
    """
    t_len, v_len, e_len = max_lengths
    t_n, v_n, e_n = ns

    if supe:
        t_data = make_bool_data(TRUE, min_len=min_len, max_len=t_len, n=t_n)
    else:
        if task in {AND, NOR}:  # necessary because of uneven class distributions
            t_data = ((True if task == AND else False,) * n for n in range(min_len, t_len + 1))
        else:
            t_data = make_bool_data(task, min_len=min_len, max_len=t_len, n=t_n)

    if task in {OR, NOR, AND, NAND}:  # necessary because of uneven class distributions
        if v_n:
            v_data = tuple(make_bool_data(_neg(task), min_len=t_len + 1, max_len=v_len, n=v_n // 2))
            v_data = chain(v_data, make_bool_data(task, min_len=t_len + 1, max_len=v_len, n=v_n - len(v_data)))
        else:
            v_data = ()

        if e_n:
            e_data = tuple(make_bool_data(task, min_len=v_len + 1, max_len=e_len, n=e_n // 2))
            e_data = chain(e_data, make_bool_data(_neg(task), min_len=v_len + 1, max_len=e_len, n=e_n - len(e_data)))
        else:
            e_data = ()
    else:
        v_data = make_bool_data(TRUE, min_len=t_len + 1, max_len=v_len, n=v_n) if v_n else ()
        e_data = make_bool_data(TRUE, min_len=v_len + 1, max_len=e_len, n=e_n) if e_n else ()

    return t_data, v_data, e_data


def make_bool_data(is_true: Varboolean, max_len: int, *, min_len: int = 0, n: int = -1) -> Iterable[Tuple[bool, ...]]:
    """
    Generates a collection of boolean sequences, filtered by `is_true()`.

    :param is_true: function determining if a boolean sequence evaluates to 'True'
    :param max_len: maximum length of the generated sequences
    :param min_len: minimum length of the generated sequences; default: 1
    :param n: number of sequences; if n=-1, all sequences between 'min_len' and 'max_len' will be generated
    :return: an iterator of tuples of boolean values
    """
    check(is_true, Fn)
    check(max_len, int, lambda: max_len > 0)
    check(min_len, int, lambda: 0 <= min_len <= max_len, f'min length [{min_len}] should be >= max length [{max_len}]')
    check(n, int, lambda: n == -1 or n > 0, f'[{n}] should be -1 or greater than 0')

    if n != -1 and n >= sum(2 ** L for L in range(min_len, max_len + 1)):
        n = -1  # handles scenario where the provided n >= the number of sequences in the provided interval

    if n != -1:
        k = 0
        for idx, seq in enumerate(make_bool_data(is_true, min_len=min_len, max_len=max_len, n=-1)):
            if k < n:  # TODO ADD SAMPLING, OTHERWISE MOST STRINGS WILL HAVE THE SAME SHORTEST LENGTH
                k += 1
                yield seq
            else:
                return
    else:
        yield from flatten(map(lambda ln: filter(is_true, product(BOOLS, repeat=ln)), range(min_len, max_len + 1)))
