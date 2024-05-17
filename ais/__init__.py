from __future__ import annotations

import random
import sys
from collections import OrderedDict
from collections.abc import Sequence, Iterable, Iterator
from functools import partial, update_wrapper
from itertools import zip_longest
from logging import getLogger, INFO, StreamHandler, Formatter, Logger, FileHandler
from types import UnionType
from typing import Callable, Tuple, Type, Any, TypeVar, runtime_checkable, Protocol, List, Generic, TypeAlias

import numpy as np
from numpy import ndarray, newaxis
from torch import set_deterministic_debug_mode, use_deterministic_algorithms

Fn: TypeAlias = Callable
T, K, V, X, Y = (TypeVar(name) for name in 'TKVXY')

EPS = '𝝺'
UNK = '⊡'
BOS: str = '∙'
EOS: str = '⊙'
PAD: str = '★'
ZERO, ONE = '𝟘', '𝟙'


def config_logging(logger: Logger = getLogger(__name__), path: str = '', sparse: bool | None = True) -> Logger:
    """
    Configures given logger or creates and configures a new one

    :param logger: The logger to configure
    :param path: if not empty, the file to log to, in addition to the console
    :param sparse: wether the logging format should be short (True), long (False) or minimal (None)
    :return: the configured logger
    """
    if sparse is None:
        formatter = Formatter('[%(asctime)s] %(message)s')
        formatter.datefmt = '%y%m%d %H%M:%S'
    elif sparse:
        formatter = Formatter('[%(asctime)s][%(levelname)1s] %(message)s')
        formatter.datefmt = '%Y%m%d %H:%M:%S'
    else:
        formatter = Formatter('[%(asctime)s][%(levelname)s][%(name)s.%(funcName)s:%(lineno)3d] %(message)s')
        formatter.datefmt = '%Y-%m-%d %H:%M:%S'

    logger.handlers.clear()
    logger.setLevel(INFO)

    handler = StreamHandler(stream=sys.stdout)
    handler.setLevel(INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if path:
        handler = FileHandler(path)
        handler.setLevel(INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


LOG = config_logging(getLogger(__package__))


def px(fn: Fn, *args, **kwargs):
    """
    Replacement for `functools.partial` that's concise and inherits the properties of the partialled out function

    :param fn: function to partial out
    :param args: position parameters
    :param kwargs: keyword parameters
    :return: partialled out function
    """
    args = (fn,) + args
    return update_wrapper(partial(*args, **kwargs), fn)


def ident(arg: T) -> T:
    """ Identity function """
    return arg


def last(pair: Tuple[Any, Y]) -> Y:
    """ Function that returns the last of a 2-tuple"""
    return pair[-1]


def outm(M: TensorLike | ndarray | Sequence,
         *,
         fracs: int = 3,
         ints: int = 1,
         tol: float = 1e-9,
         rows: Sequence[str] = '',
         cols: Sequence[str] = '',
         tubes: Sequence[str] = '',
         slabs: Sequence[str] = '',
         fr: str = '',
         row_gap: int = 3,
         sep: str = '') -> TensorLike | ndarray | Sequence:
    """
    Pretty-prints a matrix

    :param M: the matrix to be printed to console
    :param fracs: the number of fractional digits to print
    :param ints: the number of integer digits to print
    :param tol: the minimum number that switches the display of -1's, 0's and 1's to a symbol
    :param rows: the labels to print before each row
    :param cols: the labels to print above each column
    :param tubes: the labels to print before each tube
    :param slabs: the labels to print before each slab
    :param fr: the label to print before the first row
    :param row_gap: the gap between the left margin and the beginning of each row
    :param sep: the separator symbol to print repeatedly for a total of 120 characters after the matrix
    :return: the input matrix
    """
    check(M, TensorLike | ndarray | Sequence)
    check(fracs, int, lambda: fracs >= 0)
    check(ints, int, lambda: ints >= 0)
    check(tol, float, lambda: tol >= 0)
    check(rows, Sequence).check(cols, Sequence).check(tubes, Sequence).check(tubes, Sequence)
    check(fr, str)
    check(row_gap, int, lambda: row_gap >= 1)
    check(sep, str)

    A = to_ndarray(M.unsqueeze(0) if hasattr(M, 'ndim') and M.ndim == 0 else M)
    if not len(A.tolist()):
        print(A)

    if A.ndim == 0:
        A = A[newaxis, :]

    if A.ndim == 1:
        A = A[newaxis, :] if not rows else A[: newaxis]

    if A.ndim == 2:
        print(f'{"":{row_gap + 1}} '
              f'{"".join(f"{c:^{fracs + ints + 2 + 1}.{fracs}}" for c in cols[:len(A[0])])}  '
              f'{"".join(cols[len(A[0]):])}')

        if fr:
            if not rows:
                rows = [fr]
            else:
                rows[0] = fr

        for head, row in zip_longest(rows, A, fillvalue=' ' * (0 if not rows else len(rows[0]))):
            print(f'{head:{row_gap}} ' + pretty_row(row, fracs, ints, tol))

    elif A.ndim == 3:
        tubes = tuple(tubes or range(1, A.shape[0] + 1))

        for tube, matrix in zip(tubes, A):
            print(tube)
            tube_sep = '' if tube == tubes[-1] else '. '
            outm(matrix, fracs=fracs, ints=ints, tol=tol, rows=rows, cols=cols, row_gap=row_gap, sep=tube_sep,
                 fr=fr)

    elif A.ndim == 4:
        slabs = tuple(slabs or range(1, A.shape[0] + 1))
        for slab, matrix in zip(slabs, A):
            print(slab, ':')
            slab_sep = '' if slab == slabs[-1] else '. '
            outm(matrix, fracs=fracs, ints=ints, tol=tol, rows=rows, cols=cols, row_gap=row_gap, sep=slab_sep)
    else:
        print(A)

    if sep:
        print(sep * (120 // len(sep)))

    return M


def pretty_row(row: TensorLike | ndarray | Sequence, fracs: int = 3, ints: int = 1, tol: float = 1e-9) -> str:
    """
    Format row for easier visualisation of matrix rows and row vectors

    :param row: the vector to format
    :param fracs: the number of fractional digits to print
    :param ints: the number of integer digits to print
    :param tol: the minimum number that switches the display of -1's, 0's and 1's to a symbol
    :return: the formatted vector
    """
    nums = (num.item() if hasattr(row, 'item') else num for num in row)
    return f'{" ".join(pretty_num(num, fracs, ints, tol) for num in nums)}'


def pretty_num(num: float, fracs: int = 3, ints: int = 1, tol: float = 1e-9) -> str:
    """
    Formats number for easier visualisation of matrices

    :param num: the number to format
    :param fracs: how many characters wide the decimal part should take up
    :param ints: how many characters wide the integral part should take up
    :param tol: how close a number has to be to 0, 1 or -1 to be displayed with special symbols
    :return: the formatted number
    """
    if abs(num - 0) < tol:
        symbol = '∘'
    elif abs(num - 1) < tol:
        symbol = '■'
    elif abs(num - -1) < tol:
        symbol = '□'
    else:  # allows for negative signs
        return f'{num: {ints + fracs + 2}.{fracs}f}'.replace('0.', ' .' if abs(num) < 1 else '0.').rstrip()

    return f'{"   " + symbol:^{ints + fracs + 2}.{ints + fracs + 2}s}'


def check(obj: Any = object(),
          atype: Type | UnionType = object,
          val: bool | Fn[[], bool] = partial(bool, True),
          msg: str | Fn[[], str] = '') -> Check:
    """
    Provides pre-condition checks for type and value.

    :param obj: the object to check
    :param atype: the type(s) the object should be an instance of
    :param val: whether the objects' values are valid
    :param msg: the message to display in the ValueError if the value is invalid
    :return: an instance of `Check` to allow for method chaining, with a view to conciseness
    """
    if not isinstance(obj, atype):
        raise TypeError(f'argument should be of type [{atype}] but was [{type(obj)}]')

    if (isinstance(val, bool) and not val) or (isinstance(val, Fn) and not val()):
        raise ValueError(msg() if isinstance(msg, Fn) else msg)

    return Check()


class Check:
    """
    Enables chaining of `check` calls
    """

    def check(self,
              obj: Any = object(),
              atype: Type | UnionType = object,
              val: bool | Fn[[], bool] = partial(bool, True),
              msg: str | Fn[[], str] = '') -> Check:
        check(obj, atype, val, msg)
        """ see check() """
        return self


def make_deterministic(seed: int = 42):
    """
    Makes execution deterministic by setting pytorch, cuda, numpy and python randon number generators to the given seed;
    plus it makes pytorch and cuda algorithms deterministic.

    This will still not make the code deterministic across python runs unless PYTHONHASHSEED has been set before running
    the Python interpreter

    :param seed: the seed supplied to random number generators
    :return: nothing but mutates the application state
    """
    check(seed, int)
    # Pytorch imported locally because it loads very slowly
    import torch as t
    from torch import cuda
    from torch.backends import cudnn
    from torch.utils import deterministic

    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    set_deterministic_debug_mode(0)
    use_deterministic_algorithms(True, warn_only=True)
    deterministic.fill_uninitialized_memory = True


def to_ndarray(matrix: ndarray | TensorLike | Sequence) -> ndarray:
    """
    Converts input matrix to ndarray if it's not already

    :param matrix: the input matrix
    :return: an ndarray
    """
    check(matrix, ndarray | TensorLike | Sequence)

    return (matrix if isinstance(matrix, ndarray) else
            matrix.detach().cpu().numpy() if isinstance(matrix, TensorLike) else
            np.array(matrix))


@runtime_checkable
class TensorLike(Protocol):
    """
    Enables use of Tensor type checks in function signatures without having to import Pytorch as it's slow
    """

    def item(self) -> float:
        pass

    def tolist(self) -> List:
        pass

    def __len__(self) -> int:
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        return ()

    @property
    def ndim(self) -> int:
        return -1

    def __iter__(self) -> Iterator:
        pass

    def __getitem__(self, item):
        pass

    def numpy(self) -> ndarray:
        pass

    def numel(self) -> int:
        pass

    def detach(self) -> TensorLike:
        pass

    def cpu(self) -> TensorLike:
        pass

    def unsqueeze(self, dim: int) -> TensorLike:
        pass


class OrderedSet(OrderedDict, Generic[K]):
    def __init__(self, keys: Iterable[K] = ()):
        super().__init__(dict.fromkeys(keys))

    def add(self, item: K) -> OrderedSet:
        super().__setitem__(item, None)
        return self
