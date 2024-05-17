"""
Functions to learn Weighted Finite State Automata using the Spectral method
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import product, starmap
from logging import getLogger
from operator import concat
from typing import Tuple, TypeVar, Sequence, Mapping, Iterable, Any, Literal
from warnings import filterwarnings

from more_itertools import flatten
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from torch import FloatTensor, Tensor, from_numpy, stack, inference_mode, full
from torch.linalg import pinv, svd, matrix_rank

from ais import check, px
from ais.model_creation import wfsa_from, WFSA_MODES
from ais.models import WFSA

T = TypeVar('T')

LOG = getLogger(__package__)
BASIS_REGEX = re.compile(r'(freq-rank|length)=\d+:\d+')

filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.decomposition._nmf')


@inference_mode()
def learn_wfsa(kind: Literal['binary', 'polar', 'lm', 'log-lm'],
               data: Sequence[Sequence[T]],
               targets: Mapping | None = None,
               basis: str = 'freq-rank=1:1',
               base_vocab: Iterable[T] = (),
               **kwargs: Any) -> WFSA:
    """
    Learns a WFSA (aka WFA) using the Spectral method.

    :param kind: the type of WFSA to learn, one of 'binary', 'polar', 'lm', 'log-lm'; 'log-lm' is essentiall y the same
        as 'lm' but returned a WFSA that outputs log probabilities instead of probabilities
    :param data: training data
    :param targets: map from every sequence in the training data to a target value, used to fill the Hankel matrix; if
        not provided, it will be calculated from the training data statistics
    :param basis: basis choice algorithm in '<NAME>=<PREFIX-PARAMETER>:<SUFFIX-PARAMETER>' format,
        one of 'freq-rank', 'length'
    :param base_vocab: the vocabulary to add to the one found in the training data
    :param kwargs: arguments to be passed to `ais.spectral.estimate_parameters()`
    :return: the learned WFSA as a Pytorch module
    """
    check(targets, Mapping | None)

    targets = targets or estimate_targets(data, kind=kind)
    prefs, suffs = build_basis(data, basis, base_vocab=base_vocab)
    hankel = fill_hankel(targets, prefs, suffs, default=-1. if kind == 'polar' else 0.)
    params = estimate_parameters(hankel, **kwargs)
    return wfsa_from(*params, mode=kind)


def estimate_targets(data: Iterable[Sequence[T]], kind: str) -> Mapping[Sequence[T], float]:
    """
    Estimates the cells of the Hankel matrix, which contain the target values for the task to be learnt by the WFSA.
    Supported kinds of targets:
        `binary`    -> binary two-class classification {0, 1}
        `polar`     -> polar two-class classification {-1, 1}
        `lm`        -> language modelling [0,1]
        'lm-log'    -> has the same effect as 'lm', but supported for convenience

    :param data: a collection of sequences
    :param kind: type of targets, one of `binary`, `polar`, 'lm`, 'lm-log`
    :return: a mapping from each sequence in `data` to a target value
    """
    check(data, Iterable).check(val=kind in WFSA_MODES)

    default = px(float, -1 if kind == 'polar' else 0.)
    counts = Counter(data)
    match kind:
        case 'lm' | 'log-lm':
            n = counts.total()
            seq_to_value = defaultdict(default, {seq: val / n for seq, val in counts.items()})
        case _:
            seq_to_value = defaultdict(default, {seq: min(count, 1) for seq, count in counts.items()})

    return seq_to_value


def build_basis(sequences: Iterable[Sequence[T]], basis: str, base_vocab: Iterable[T] = ()) \
        -> Tuple[Tuple[T, ...], Tuple[T, ...]]:
    """
    Builds prefix-closed Hankel basis for the given sequences according to the provided algorithm. The supported algos
    are:
        freq-rank   -> picks the k-most frequent prefixes/suffixes
        length      -> picks all prefixes/suffixes whose length is less than or equal than the threshold

    :param sequences: a collection of sequences
    :param basis: the basis-finding algorithm, with syntax `<KIND>=<PREFIX-PARAMETER>:<SUFFIX-PARAMETER>`
    :param base_vocab: the base vocabulary, possibly empty
    :return: a 2-tuple with the prefixes and suffixes found
    """

    check(sequences, Iterable)
    check(basis, str, lambda: BASIS_REGEX.match(basis), basis)
    check(base_vocab, Iterable)

    p_val, s_val = map(int, basis.split('=')[-1].split(':'))

    if basis.startswith('freq-rank'):
        return _by_freq_rank(sequences, p_val, s_val, base_vocab)

    if basis.startswith('length'):
        return _by_length(sequences, p_val, s_val, base_vocab)


def fill_hankel(seq_to_value: Mapping[Sequence[T], float],
                prefs: Sequence[Sequence[T]],
                suffs: Sequence[Sequence[T]],
                default: float = 0.) -> Tensor:
    """
    Fills a Hankel matrix with basis given by `prefs` and `suffs` and with cell values given by `seq_to_value` if the
    string corresponding to a cell is in `seq_to_value` else they are set to `default`. The data is assumed to represent
    the empty string with an empty sequence.

    The returned tensor is rank 3, where the first dimension (tube) is the subblock, of size |V|+1, the second
    dimension (row) stands for prefixes, of size |P'|, and the third dimension (column) stands for suffixes, of size |S|.
    |V| is the vocabulary size, |P'| is the number of root prefixes and |S| the number of suffixes

    :param seq_to_value: a mapping from sequence to target value, to go in the hankel cell
    :param prefs: the prefix-closed basis, assumed to be sorted by the last token
    :param suffs: the suffix basis
    :param default:
    :return: a rank-3 tensor of size |V|+1x|P'|x|S|
    """
    check(seq_to_value, Mapping, lambda: len(seq_to_value) > 0)
    check(prefs, Sequence, lambda: len(prefs) > 0).check(suffs, Sequence, lambda: len(suffs) > 0).check(default, float)

    hankel = full((len(prefs), len(suffs)), default)  # PxS
    for (i, prefix), (j, suffix) in product(enumerate(prefs), enumerate(suffs)):
        hankel[i, j] = seq_to_value.get(tuple(prefix + suffix), default)

    vocab_n = len(set(flatten(prefs))) + 1  # V, assumes prefixes contain full vocabulary, then adds the empty string
    return hankel.view((vocab_n, -1, len(suffs)))  # PxS -> V+1xP'xS


def estimate_parameters(hankel: Tensor, dim: int = -1, tol: float = 1e-1, algo: str = 'svd', **kwargs) \
        -> Tuple[Tensor, Tensor, Tensor]:
    """
    Estimates initial weights, final weights and transition matrices for Linear WFSA. The first row in `hankel` is
    assumed to correspond to the empty-string (epsilon) subblock. Extra parameters are passed to ais.spectral.nmf_of().

    When `dim` = `-1`, the dimensionality of the WFSA, ie, the number of minimal states, is found by keeping the singular
    values that are at least `tol` times the largest singular value of the Hankel complete subblock.

    When `dim` is not `-1`, the dimensionality is set to that value, in which case `tol` is ignored.

    :param hankel: hankel block as a |V|x|P'|x|S| rank-3 tensor, where V is vocab, P' root prefixes and S suffixes
    :param dim: dimension of WFSA, if `dim` =-1, the dimension found by SVD, in conjunction with `tol`, is used
    :param tol: minimum fraction of the largest singular value singular values need to have to be kept;
        ignored when `dim` != -1
    :param algo: specifies which matrix factorisation algorithm to use, one of {nmf, svd}; default: nmf
    :return: a 3-tuple with initial and final weights and transition tensor of the induced WFSA; sizes D, D and DxDxV
    """
    check(hankel, Tensor, lambda: hankel.ndim == 3 and hankel.numel())
    check(dim, int, lambda: dim == -1 or dim >= 1)
    check(tol, float, lambda: 0 <= tol <= 1)
    check(val=algo in {'svd', 'nmf'})
    check(val=(algo == 'svd' and not kwargs) or (algo == 'nmf'), msg=f'[{algo}] and {kwargs}')

    complete_block = hankel[0]  # VxP'xS -> P'xS

    P, S = _low_rank_factorise(complete_block, dim, tol, algo, **kwargs)  # D = dim; P'S -> P'xD, DxS

    pinv_P = pinv(P)  # P'xD -> DxP'
    pinv_S = pinv(S)  # DxS -> SxD
    init = pinv_S.T @ complete_block[0, :]  # [[SxD -> DxS] @ S] -> D
    final = complete_block[:, 0] @ pinv_P.T  # [P' @ [DxP' -> P'xD]] -> D
    trans = stack([(pinv_P @ (block @ pinv_S)).T for block in hankel[1:]], dim=-1)
    # ^ V: [[DxP' @ P'xS @ SxD] -> DxD -> DxD] -> DxDxV

    return init, trans, final,  # D, DxDxV, D


# --------------------------------------------- DELEGATE FUNCTIONS -----------------------------------------------------


def _by_length(sequences: Iterable[Sequence[T]], topp: int, tops: int, base_vocab: Iterable[T] = ( )) \
        -> Tuple[Tuple[T, ...], Tuple[T, ...]]:
    """
    Makes prefix-closed Hankel basis for the given sequences, choosing prefixes whose length is >= `topp` and suffixes
    whose length >= `tops`

    The number of prefixes returned is augmented with the tokens in `base_vocab`:
    (|topp| + |`base_vocab`|) x |vocab U `base_vocab`|
    The number of suffixes returned is augmented with the tokens in `base_vocab`: |tops| + |`base_vocab`|

    The prefixes are sorted by then by decreasing length, then by last token, then by content
    The suffixes are sorted by decreasing length, then by content

    :param sequences: a collection of sequences
    :param topp: the shortest prefixes to include in the basis
    :param tops: the shortest suffixes to include in the basis
    :param base_vocab: the base vocabulary to include in the basis, possibly empty
    :return: a 2-tuple with the prefixes and suffixes making up the basis
    """
    check(sequences, Iterable)
    check(topp, int, lambda: topp >= 1).check(tops, int, lambda: tops >= 1).check(base_vocab, Iterable)

    sequences = tuple(map(tuple, sequences))
    vocab = set(base_vocab).union(flatten(sequences))
    infixes = {(token,) for token in vocab} | {()}

    suffixes = {suffix for suffix in _affixes_of(sequences, prefix=False) if len(suffix) <= tops}
    suffixes = sorted(infixes.union(suffixes), key=_epsilon_first)

    prefixes = {prefix for prefix in _affixes_of(sequences) if len(prefix) <= topp}
    prefixes = sorted(infixes.union(prefixes), key=_epsilon_first)

    closed_prefixes = starmap(concat, sorted(product(sorted(infixes), prefixes), key=lambda seq: seq[::-1]))

    return tuple(closed_prefixes), tuple(suffixes)


def _by_freq_rank(sequences: Iterable[Sequence[T]], topp: int, tops: int, base_vocab: Iterable[T] = ( )) \
        -> Tuple[Tuple[T, ...], Tuple[T, ...]]:
    """
    Makes prefix-closed Hankel basis for the given sequences, choosing the `topp` most frequent prefixes and the `tops`
    most frequent suffixes.

    The number of prefixes returned is augmented with the tokens in `base_vocab`:
    (|topp| + |`base_vocab`|) x |vocab U `base_vocab`|
    The number of suffixes returned is augmented with the tokens in `base_vocab`: |tops| + |`base_vocab`|

    The prefixes are sorted by then by decreasing length, then by last token, then by content
    The suffixes are sorted by decreasing length, then by content

    :param sequences: a collection of sequences
    :param topp: the most frequent prefixes to include in the basis
    :param tops: the most frequent suffixes to include in the basis
    :param base_vocab: the base vocabulary to include in the basis, possibly empty
    :return: a 2-tuple with the prefixes and suffixes making up the basis
    """
    check(sequences, Iterable)
    check(topp, int, lambda: topp >= 1).check(tops, int, lambda: tops >= 1).check(base_vocab, Iterable)

    sequences = tuple(map(tuple, sequences))
    vocab = set(base_vocab).union(flatten(sequences))
    infixes = {(token,) for token in vocab} | {()}

    _suffixes, _ = zip(*Counter(_affixes_of(sequences, prefix=False)).most_common(tops))
    suffixes = sorted(infixes.union(_suffixes), key=_epsilon_first)

    _prefixes, _ = zip(*Counter(_affixes_of(sequences)).most_common(topp))
    prefixes = sorted(infixes.union(_prefixes), key=_epsilon_first)

    closed_prefixes = starmap(concat, sorted(product(sorted(infixes), prefixes), key=lambda seq: seq[::-1]))

    return tuple(closed_prefixes), tuple(suffixes)


def _low_rank_factorise(complete_block: Tensor, dim: int, tol: float, algo: str, **kwargs) -> Tuple[Tensor, Tensor]:
    if algo == 'nmf':
        return nmf_of(complete_block, dim=dim, tol=tol, **kwargs)  # P'xS -> P'xD, DxS

    U, Z, Vt = svd_of(complete_block, dim=dim, tol=tol)  # P'xS -> P'xD, DxD, DxS
    Z = Z ** .5
    return U @ Z, Z @ Vt  # [P'xD @ DxD] -> P'xD, [DxD @ DxS] -> DxS (nicer weights than U @ Z, Vt)


def nmf_of(matrix: Tensor, dim: int = -1, tol: float = 0., init: str = 'nndsvd', shuffle: bool = False, seed: int = 42) \
        -> Tuple[Tensor, Tensor]:
    """
    Finds two matrices, Encoder and Decoder, that when multiplied together approximate the given non-negative matrix,
    using Non-negative Matrix Factorisation (NMF). Supports 2 scenarios:

    a) if `dim` is not `-1`, an NMF is computed on `matrix` truncated to the number of dimensions specified in `dim`, in
        which case, the parameter `tol` is ignored

    b) if `dim=-1`, a NMF is computed with the number of dimensions decided by how many singular values are
        `tol` times the largest singular value.

    The current implementation is a thin Pytorch wrapper around `sklearn.decomposition.NMF`

    :param matrix: the matrix to be factorised into encoder and decoder. Must contain only non-negative values
    :param dim: inner dimension of factor matrices; default: -1
    :param tol: the minimum fraction of the largest singular value that a singular value must have to be kept;
        ignored if dim!= -1; default: 0
    :param init: initialisation strategy for NMF, one of `svd, random, nndsvd, nndsvda, nndsvdar`,
        see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html for details; default:
        nndsvd
    :param shuffle: whether the coordinates should be shuffled.
        See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html for details; default: False
    :param seed: random seed, for reproducibility; default: 42
    :return: encoder and decoder matrices such that matrix ~ encoder @ decoder
    """
    M = matrix
    check(M, Tensor, lambda: M.ndim == 2 and M.numel() > 0, lambda: f'matrix should be non-empty, found {M}')
    L = max(M.shape)
    check(dim, int, lambda: ((1 <= dim <= L) or dim == -1), f'dim should be in [1,{L}] or -1, found [{dim}]')
    check(tol, float, lambda: 0 <= tol <= 1, f'tol should be in [0, 1], found [{tol}]')
    check(val=init in {'svd', 'random', 'nndsvd', 'nndsvda', 'nndsvdar'})

    if init == 'svd':

        U, S, Vt = svd_of(M, dim=dim, tol=tol)  # PxS -> PxD, DxD, DxS
        S = S ** .5
        E, D = (U @ S).abs().numpy(), (S @ Vt).abs().numpy()  # [P'xD @ DxD] -> P'xD, [DxD @ DxS] -> DxS
        nmf = NMF(E.shape[-1], random_state=seed, max_iter=1000, init='custom', shuffle=shuffle)
        dec = from_numpy(nmf.fit_transform(M.numpy(), W=E, H=D)).type(FloatTensor)  # PxS, PxD, DxS -> DxS

    else:
        dim = matrix_rank(M, atol=0, rtol=tol).item() if dim == -1 else dim
        nmf = NMF(dim, random_state=seed, max_iter=1000, init=init, shuffle=shuffle)
        dec = from_numpy(nmf.fit_transform(M.numpy())).type(FloatTensor)  # PxS -> DxS

    enc = from_numpy(nmf.components_).type(FloatTensor)  # DxS
    return dec, enc  # PxD, DxS


def svd_of(matrix: Tensor, dim: int = -1, tol: float = 0.) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Performs a possibly truncated Singular Value Decomposition of the given PxS `data` tensor. Supports two scenarios:

    a) if `dim` >= 1, a truncated SVD will be performed such that only the top `dim` singular values are preserved, ie,
        matrix[PxS] ~ U[PxD] S[DxD] Vt[DxS]. In this case, the `tol` parameter will be ignored. Use when
        you know exactly how many dimensions you want

    b) if `dim` = -1, a truncated SVD will be computed such that matrix[PxS] ~ U[PxD] S[DxD] Vt[DxS], where D is the
        number of singular values that are greater or equal than `tol` times the largest singular value. Use wheb
        you want only the dimensions that explain the most variance. When `tol=0`, the number of
        dimensions will be the rank of the matrix.

    The current implementation is a thin Pytorch wrapper around `torch.linalg.svd`

    :param matrix: The matrix to compute the SVD on, a rank-2 PxS tensor
    :param dim: the number of dimensions/singular values to keep, default: -1
    :param tol: the minimum fraction of the largest singular value that other singular values must have to be kept;
        ignored if dim!= -1; default: 0
    :return: a 3-tuple containing the U, S and Vt factors of the SVD decomposition, of sizes PxD, DxD, DxS
    """
    M = matrix
    check(M, Tensor, lambda: M.ndim == 2 and M.numel() > 0, lambda: f'matrix should be non-empty, found {M}')
    L = max(M.shape)
    check(dim, int, lambda: ((1 <= dim <= L) or dim == -1), f'dim should be in [1,{L}] or -1, found [{dim}]')
    check(tol, float, lambda: 0 <= tol <= 1, f'tol should be in [0, 1], found [{tol}]')

    U, s, Vt = svd(M, full_matrices=False)  # PxS -> PxR, R, RxS    TODO use sparse tensors

    if dim == -1:
        dim = (s >= s[0] * tol).count_nonzero()

    return U[:, :dim], s[:dim].diag(), Vt[:dim, :]  # PxR, R, RxS -> PxD, DxD, DxS


def _affixes_of(sequences: Iterable[Sequence[T]], prefix: bool = True) -> Iterable[T]:
    return flatten(map(lambda i: seq[i:] if prefix else seq[:i], range(len(seq) + 1)) for seq in sequences)


def _epsilon_first(item: Sequence[T]) -> Tuple[int, Sequence[T]]:
    return len(item), item
