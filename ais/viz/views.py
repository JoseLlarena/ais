"""
Collection of functions to translate raw data to a format fit for visualisation
"""
from collections import defaultdict, namedtuple
from collections.abc import Sequence, Iterable
from functools import wraps
from itertools import pairwise, zip_longest
from logging import getLogger
from types import NoneType
from typing import Tuple, TypeVar
from warnings import filterwarnings

import numpy as np
import torch as t
from numpy import ndarray, float32
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, FastICA, SparsePCA, NMF, TruncatedSVD
from sklearn.manifold import LocallyLinearEmbedding as LLE, SpectralEmbedding, Isomap, MDS, TSNE
from torch import Tensor

from ais import check, Fn, to_ndarray, OrderedSet

A = TypeVar('A', Tensor, ndarray)
Graph = namedtuple('Graph', ['vertices', 'edges', 'outputs', 'vertex_labels', 'edge_labels'])

LOG = getLogger(__package__)
SEED = 42
NAME_TO_CLASS = {'fa': FactorAnalysis,
                 'ica': FastICA,
                 'isomap': Isomap,
                 'kpca': KernelPCA,
                 'lle': LLE,
                 'mds': MDS,
                 'nmf': NMF,
                 'pca': PCA,
                 'sparsepca': SparsePCA,
                 'spectral': SpectralEmbedding,
                 'svd': TruncatedSVD,
                 'tsne': TSNE,
                 'umap': None}
NAME_TO_KERNEL = {'cpca': 'cosine', 'ppca': 'poly', 'rpca': 'rbf', 'spca': 'sigmoid'}

filterwarnings('ignore', category=UserWarning)
filterwarnings('ignore', category=RuntimeWarning)  # TODO ADD MODULE LEVEL FILTER


def reduce_level_dim(states: A, nonlin: str = 'spca', dim: int = 3) -> ndarray:
    """
    Reduces the dimensionality of the states in the given matrix if greater than `dim`. Dimensionality is reduced
    by as many principal components as needed to keep 99.9% of the variance. If this results in more than 3 components
    then `nonlin` is applied to reduce it to 3. These steps are done to minimise warping of the original space.

    :param states: a dimension by token (DxN) matrix
    :param nonlin: the nonlinear dimensionality reduction algorithm to apply after PCA
    :param dim: the maximum dimensionality the states must have to avoid dimensionality reduction
    :return: a `dim`xN ndarray
    """
    check(states, Tensor | ndarray, lambda: states.ndim == 2, lambda: f'[{states.shape}]')  # assumes DxN
    check(val=nonlin in NAME_TO_CLASS | NAME_TO_KERNEL, msg=f'[{nonlin}]')
    check(dim, int, lambda: dim > 0, f'[{dim}]')

    states = np.nan_to_num(to_ndarray(states), neginf=0.)  # DxN -> DxN -> DxN; handles masked attention matrices
    _dim = states.shape[0]

    if _dim > dim:
        LOG.info(f'[{_dim}] dims before pca')
        states = dim_reduce(states, dim=-1, algo='pca', min_var=.999)  # DxN -> SxN
        _dim = states.shape[0]
        (LOG.info if _dim <= dim else LOG.warning)(f'[{_dim}] dims after pca with 99.9% explained variance')

        if _dim > 3:
            states = dim_reduce(states, dim=dim, algo=nonlin)  # SxN -> KxN
            LOG.info(f'{states.shape[0]} dims after pca -> [{nonlin}]')

    return states  # KxN


def quantise_level(states: A, radius: float) -> ndarray:
    """
    Quantises states by clustering those that are within the given radius of each other. The larger the radius, the
    coarser the quantisation and thus the smaller the number of clusters (K).

    :param states: a dimension by token (DxN) matrix
    :param radius: the radius within which two points are considered to be the same
    :return: a DxK ndarray
    """
    check(states, Tensor | ndarray, lambda: states.ndim == 2, lambda: f'[{states.shape}]')  # assumes DxN
    check(radius, float, lambda: radius > 0)

    states = to_ndarray(states)
    if np.all(states == states[0][0]):
        return states

    d_indices, seen = [], {}
    for idx, c in enumerate(fclusterdata(states.T, radius, criterion='distance')):
        if c not in seen:
            seen[c] = idx
        d_indices.append(seen[c])

    return states[:, d_indices]


def normalise_level(level: A, emph: str = 'v') -> ndarray:
    """
    Normalises every dimension to [-1, 1], previous adding a 3rd (z) and 2nd (y) dimensions if needed.

    :param level: a dimension by token (DxN) matrix
    :return: a DxN ndarray
    :param emph: which direction should be emphasised, one of `v`(-ertical), `h`(-orizontal)
    """
    check(level, Tensor | ndarray, lambda: level.shape[0] and level.shape[-1], lambda: f'[{level.shape}]')
    check(val=emph in {'v', 'h'}, msg=f'[{emph}]')

    level = to_3d(to_ndarray(level))  # assumes DxN -> 3xN

    mins, maxes = level.min(axis=-1, keepdims=True), level.max(axis=-1, keepdims=True)  # 3x1, 3x1
    if emph == 'h':
        mins, maxes = np.tile(mins.min(), (3, 1)), np.tile(maxes.max(), (3, 1))
    extent = maxes - mins  # 3x1
    mask = np.sign(np.abs(extent))  # 3x1; ensures constant-value dimensions are set to zero, which will centre them
    extent[extent == 0] = 1  # avoids division by zero
    new_min, new_extent = -1, 2

    return (new_min + (level - mins) * new_extent / extent) * mask  # [3xN/3x1 -> 3xN] * 3x1 -> 3xN


def stack_levels(levels: Sequence[A],
                 *,
                 pad: float = 0.,
                 lower: float | None = None,
                 upper: float | None = None,
                 uniform: bool = False) -> Tuple[ndarray, ...]:  # assumes 3xN
    """
    Stacks up levels by shifting the zs so then don't overlap. Optional padding can be added between levels. If
    `uniform` is True, each level will be rescaled to the same z-extent keeping the original total height. If `lower`
    and `upper` are both not None, the levels will be rescaled to stretch between `lower` and `upper`, maintaining the
    same relative z-extent. This function doesn't change any other dimension other than z.

    :param levels: a sequence of dimension by token (DxN) matrices of size at least 2 (L)
    :param pad: padding to add between levels
    :param lower: lower bound for rescaling
    :param upper: upper bound for rescaling
    :param uniform: whether to change the extent levels to be the same while keeping the original total height
    :return: a sequence of L ndarrays of size DxN
    """
    check(levels, Sequence, lambda: len(levels) > 0)
    check(pad, float, lambda: pad >= 0, f'[{pad}]')
    check(lower, float | None, f'[{lower}]')
    check(upper, float | None, lambda: (lower is None and upper is None) or lower < upper, f'[{upper}]')

    stacked = _pile_up(levels, pad)

    if uniform:
        stacked = _equalise(stacked)

    if lower is not None:
        stacked = _rescale(stacked, lower, upper)

    return tuple(stacked)


def align_levels(levels: Sequence[A]) -> Tuple[ndarray, ...]:
    """
    Recursively aligns levels to the previous one, starting with the second level in the sequence. The alignment is done
    only on the x-y plane by finding the rotation that minimises distances between the points in each level, using the
    Kabsch algorithm.

    :param levels: a sequence of dimension by token (DxN) matrices of size at least 2 (L)
    :return: a sequence of L DxN ndarrays
    """
    check(levels, Sequence, lambda: len(levels) > 1)

    basis = to_ndarray(levels[0])
    aligned = [basis]

    for i in range(len(levels) - 1):  # L:3xN...
        check(levels[i], Tensor | ndarray, levels[i].shape[0] == 3 and levels[i].shape[-1])

        to_be_rotated = to_ndarray(levels[i + 1])  # [3xN -> Nx3, 3xN -> Nx3] -> 3x3
        to_be_rotated_no_z = to_3d(to_be_rotated[:2])
        basis_no_z = to_3d(basis[:2])

        if np.all(basis_no_z == 0):  # necessary because align_vectors() doesn't support the zero matrix
            rot = Rotation.from_matrix(np.eye(3)).as_matrix()
        else:
            rot = Rotation.align_vectors(basis_no_z.T, to_be_rotated_no_z.T)[0].as_matrix()
        # [3x3 @ 3xN -> 3xN -> 2xN];1xN -> 3XN
        basis = np.concatenate([(rot @ to_be_rotated_no_z)[:2], to_be_rotated[2:]])

        aligned.append(basis.astype(float32))

    return tuple(aligned)


def as_multi_input(fn: Fn[[A, ...], A]) -> Fn[[Sequence[A], ...], Tuple[A, ...]]:
    """
    Decorates the passed in function so that it can take a sequence of matrices instead of a single one and return them
    as a sequence too. Convenient for operations that require the processing of different sets of points together.

    :param fn: the function to be decorated
    :return: the decorated function
    """
    check(fn, Fn)

    @wraps(fn)
    def concat_fn(matrices: Sequence[A], *args, **kwargs) -> Tuple[A, ...]:
        check(matrices, Sequence, lambda: len(matrices) > 0 and len(matrices[0]) > 0)

        concat = t.concat if isinstance(matrices[0], Tensor) else np.concatenate
        concat_in = concat(tuple(matrices), axis=-1)  # L:[DxN,DxM,...] -> DxN+M...
        concat_out = fn(concat_in, *args, **kwargs)  # DxN+M... -> KxN+M...

        split = (t.split if isinstance(concat_out, Tensor) else
                 lambda x, sizes, axis: np.split(x, np.cumsum(sizes[:-1]), axis))
        # KxN+M... ->  L:[KxN,KxM,...]
        return tuple(m for m in split(concat_out, [m.shape[-1] for m in matrices], axis=-1))

    return concat_fn


def to_graph(path: ndarray,
             outputs: ndarray | None = None,
             state_labels: Iterable[str] = (),
             trans_labels: Iterable[str] = (),
             state_label_fn: Fn[[Iterable[str]], str] = '\n'.join,
             trans_label_fn: Fn[[Iterable[str]], str] = ','.join) -> Graph:
    """
    Converts a sequence of possibly repeated states into a unique sequence, preserving the order of the first instance
    of each state, plus a sequence of transitions in the form of pairs of state ids. If outputs are provided, the
    unique instances of them are also returned, each corresponding to a state. If state labels are provided, they are
    aggregated into the same number of corresponding states. If transition labels are provided, they are aggregated into
    the same number of corresponding edges.

    :param path: a dimension by token (DxN) Tensor/ndarray
    :param outputs: a vector of size D
    :param state_labels: a collection of strings of length D
    :param trans_labels: a collection of string sof length D-1
    :param state_label_fn: a function to aggregate the state labels
    :param trans_label_fn: a function to aggregate the transition labels
    :return: a Graph object with K vertices, upto K^2 edges, 0/K outputs, 0/K vertex labels and upto K^2 edge labels
    """
    check(path, ndarray, lambda: path.ndim == 2 and path.size)
    L = path.shape[-1]
    check(state_labels, Iterable).check(trans_labels, Iterable)
    state_labels = tuple(state_labels)
    trans_labels = tuple(trans_labels)

    check(outs := outputs, ndarray | NoneType, lambda: outs is None or (outs.shape == (L,) and outs.size),
          lambda: f'{outs.shape}  vs {path.shape}')
    check(val=not state_labels or len(state_labels) == path.shape[-1])
    check(val=not trans_labels or len(trans_labels) == path.shape[-1] - 1, msg=f'{len(trans_labels)} vs {path.shape}')
    check(state_label_fn, Fn).check(trans_label_fn, Fn)
    # TODO SIMPLIFY
    step_to_indexes = defaultdict(list)
    for idx, step in enumerate(path.T):
        step_to_indexes[tuple(step.tolist())].append(idx)

    v_indexes = [0] * path.shape[-1]
    for indexes in step_to_indexes.values():
        for index in indexes:
            v_indexes[index] = indexes[0]

    vertex_to_label = defaultdict(list)
    for idx, label in zip(v_indexes, state_labels):
        vertex_to_label[idx].append(label)

    new_indexes = sorted(set(v_indexes))

    orig_to_new = dict(zip(new_indexes, range(len(new_indexes))))

    edge_to_label = defaultdict(list)
    edges = OrderedSet()
    for trans, label in zip_longest(pairwise(v_indexes), trans_labels):
        edges.add((orig_to_new[trans[0]], orig_to_new[trans[-1]]))
        if label:
            edge_to_label[orig_to_new[trans[0]], orig_to_new[trans[-1]]].append(label)

    return Graph(path[:, new_indexes],
                 tuple(edges),
                 outputs[new_indexes] if outputs is not None else np.empty(0),
                 tuple(map(state_label_fn, vertex_to_label.values())),
                 tuple(map(trans_label_fn, edge_to_label.values())))


# ------------------------------------------ DELEGATE FUNCTIONS --------------------------------------------------------

def dim_reduce(data: A, dim: int, algo: str, seed: int = SEED, **params) -> ndarray:
    """
    Reduces dimensionality of DxN tensor. Supported algos:
        cpca, fa, ica, isomap, hlle, kpca, lle, ltsa, mds, mlle, nmf, pca, ppca, rpca, sparsepca, spca, spectral, svd,
        tsne, umap

    This function is a thin facade over scikit-learn's PCA, KernelPCA, FactorAnalysis, FastICA, SparsePCA, NMF,
    TruncatedSVD, LocallyLinearEmbedding, SpectralEmbedding, Isomap, MDS and TSNE, plus umap.UMAP.

    :param data: a dimension by token tensor (DxN)
    :param dim: dimensionality to reduce to (K); use -1 for automatically guessed
    :param algo: the name of the algorithm, see above
    :param seed: the random seed
    :return: a KxN ndarray
    """
    check(data, ndarray | Tensor)
    data = to_ndarray(data)
    check(val=data.ndim == 2 and data.size > 0, msg=f'{data.shape}')
    check(dim, int, lambda: dim == -1 or 0 < dim <= data.shape[0], f'{dim} {data.shape}')
    check(algo, str, algo in NAME_TO_CLASS or algo in NAME_TO_KERNEL)
    check(seed, int)

    reducer = None

    if algo in {'mlle', 'hlle', 'ltsa'}:
        method = 'modified' if algo == 'mlle' else 'hessian' if algo == 'hlle' else algo
        reducer = NAME_TO_CLASS['lle'](n_components=dim, random_state=seed, method=method, **params)
    elif algo in {'isomap'}:
        reducer = NAME_TO_CLASS[algo](n_components=dim, n_neighbors=min(dim, len(data)), **params)
    elif algo in {'tsne'}:
        reducer = NAME_TO_CLASS[algo](n_components=min(3, dim), random_state=seed, init='random', **params)
    elif algo in NAME_TO_KERNEL:
        reducer = NAME_TO_CLASS['kpca'](n_components=dim, random_state=seed, kernel=NAME_TO_KERNEL[algo], **params)
    elif algo == 'pca':
        dim = dim if dim != -1 else params.pop('min_var', None)
        reducer = PCA(n_components=dim, random_state=seed, svd_solver='full', **params)
    else:
        from umap import UMAP  # imported locally because it loads very slowly
        NAME_TO_CLASS['umap'] = UMAP
        reducer = NAME_TO_CLASS[algo](n_components=dim, random_state=seed, **params)

    return reducer.fit_transform(data.T).T  # DxN -> NxD -> Nxdim -> dimxN


def to_3d(A: ndarray) -> ndarray:
    """
    Converts the dimensionality of the input D or DxN array to 3 or 3xN. For dimensions <3 a row of zeros is added.
    For dimensions > 3 the higher dimensions are cut out.

    :param A: input D(xN) array
    :return: output 3(xN) array
    """
    check(A, ndarray, lambda: 1 <= A.ndim <= 2 and A.size > 0)  # expects dimensions D or DxN

    D = A.shape[0]

    if D == 3:  # 3 -> 3 or 3xN -> 3xN
        return A

    if A.ndim == 1:
        _zeros = np.zeros(3)
        _zeros[:min(D, 3)] = A[:3]  # D -> 3
        return _zeros

    N = A.shape[1]
    _zeros = np.zeros((3, N))
    _zeros[:min(D, 3), :] = A[:3, :]  # DxN -> 3xN
    return _zeros


def _pile_up(levels: Sequence[A], pad: float = .1) -> Tuple[ndarray, ...]:
    below, z = to_ndarray(levels[0]), -1
    stacked = [below]

    for i, level in enumerate(levels[1:]):
        check(level, Tensor | ndarray, lambda: level.shape[0] == 3 and level.shape[-1])
        level = to_ndarray(level)
        gap = below[z].max() - level[z].min() + pad  # adds gap only if it's larger than the existing one
        below = level.copy()
        below[z] = below[z] + gap
        stacked.append(below)

    return tuple(stacked)


def _equalise(levels: Tuple[ndarray, ...]) -> Tuple[ndarray, ...]:
    bottom, top, z = levels[0], levels[-1], -1
    old_min = bottom[z].min()
    uniheight = (top[z].max() - bottom[z].min()) / len(levels)

    for level in levels:
        old_height = (level[z].max() - level[z].min())
        ratio = 1 if not old_height else uniheight / old_height
        level[z] = level[z] * ratio + old_min * (1 - ratio)
    return levels


def _rescale(levels: Tuple[ndarray, ...], lower: float, upper: float) -> Tuple[ndarray, ...]:
    bottom, top, z = levels[0], levels[-1], -1
    old_min = bottom[z].min()
    ratio = (upper - lower) / ((top[z].max() - old_min) or 1)

    for level in levels:
        level[z] = lower + (level[z] - old_min) * ratio  # val' = min' + (val-min)*(max'-min')/(max-min)

    return levels
