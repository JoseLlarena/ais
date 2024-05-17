"""
Functions to search the hyper-parameter space for both Stochastic Gradient Descent and Spectral learning
"""
from __future__ import annotations

from itertools import product
from logging import getLogger
from random import randrange
from typing import Tuple, Mapping, Sequence, Iterable, Dict, TypeVar, Literal

from torch import Tensor, inference_mode, tensor
from torch.utils.data import Dataset

from ais import check, make_deterministic, Fn
from ais.data import EMPTY_DATASET, as_dataset
from ais.model_io import register_model, make_metadata
from ais.models import WFSA
from ais.sgd import M, train_express
from ais.spectral import estimate_targets, learn_wfsa
from ais.training import compute_loss, evaluate

T, V = TypeVar('T'), TypeVar('V')

LOG = getLogger(__package__)
DEFAULT_KWARGS = dict(dim=-1, tol=1e-1, algo='svd')
MAX_LOSS = tensor(1e22)


def sgd_hp_search(model_specs: Mapping[str, Sequence[V]],
                  sgd_specs: Mapping[str, Sequence[V]],
                  t_data: Dataset,
                  v_data: Dataset,
                  e_data: Dataset = EMPTY_DATASET,
                  *,
                  shortcut_loss: float = 1e-16,
                  tracker: str = 'basic',
                  period: float = .5,
                  device: str = 'cuda:0',
                  seed: int = 42,
                  **metadata) -> Tuple[M, Tensor, Tensor]:
    """
    Exhaustively searches hyper-parameter space for SGD learning. Every time the validation loss decreases the model
    is saved, along with metadata, the latter to a registry file.

    The SGD hyper-parameters are:
        'epochss':      epochs
        'bss':          batch sizes
        'inits':        initialisation functions
        't_loss_fns':   training loss functions
        'v_loss_fns':   validation loss functions
        'lrs':          learning rates for optimisers
        'max_lrs':      maximum learning rate for schedulers
        'scheds':       schedulers
        'opts':         optimisers

    The model parameters should contain at least 'archs', the models' architectures.

    :param model_specs: specifications for model hyper-parameters, containing all the values for all the model features
    :param sgd_specs: specifications for sgd hyper-parameters, containing all values for all learning parameters
    :param t_data: the training dataset
    :param v_data: the validation dataset
    :param e_data: the test (evaluation) dataset, possibly empty
    :param shortcut_loss: the minimum loss to consider the task learned; shortcuts the learning loop
    :param tracker: the type of function to track the search process
    :param period: the logging period as percentage of the total search runs
    :param device: the device to run the search on
    :param seed: the random generator seed for reproducibility
    :param metadata: extra data to save along with the models
    :return: the best model found, along with the training and validation losses
    """
    check(model_specs, Mapping).check(sgd_specs, Mapping)
    check(t_data, Dataset).check(v_data, Dataset).check(e_data, Dataset)
    check(shortcut_loss, float, lambda: 0 <= shortcut_loss)

    runs = tuple(product(_combine(**model_specs), _combine(**sgd_specs)))

    info = _search_info(
        model_specs | sgd_specs | dict(t_data=len(t_data), v_data=len(v_data), runs=len(runs)) | metadata)
    LOG.info(info)

    top_model, top_t_loss, top_v_loss, top_ms, top_hps = None, MAX_LOSS, MAX_LOSS, {}, {}

    make_deterministic(seed)
    for run, (model_spec, hyper_spec) in enumerate(runs, start=1):
        try:
            LOG.info(_run_info(model_spec | hyper_spec, len(runs), run))

            model, t_loss, v_loss, e_loss = train_express(model_spec,
                                                          hyper_spec,
                                                          t_data,
                                                          v_data,
                                                          EMPTY_DATASET,
                                                          tracker,
                                                          period,
                                                          device,
                                                          seed := randrange(2 ** 32 - 1))
            if v_loss < top_v_loss:
                LOG.info('Found better model...')
                top_model, top_t_loss, top_v_loss, top_ms, top_hps = model, t_loss, v_loss, model_spec, hyper_spec

                LOG.info(f'Saving better model with seed [{seed}]...')
                reg = make_metadata(t_loss=top_t_loss,
                                    v_loss=top_v_loss,
                                    v_error=hyper_spec['v_loss_fn'],
                                    **({'seed': seed} | metadata),
                                    **top_ms,
                                    **top_hps)
                register_model(top_model.adapted, reg)

                if v_loss <= shortcut_loss:
                    LOG.info(f'Minimum loss achieved [{v_loss.item():.1e}]; stopping search...')
                    break

        except Exception as e:
            LOG.exception(e)

    if e_data != EMPTY_DATASET:
        e_loss = evaluate(top_model, e_data, top_hps['bs'], top_hps['v_loss_fn'])
        LOG.info(f'test loss [{e_loss:.3e}]')

    return top_model, top_t_loss, top_v_loss


@inference_mode()
def spectral_hp_search(kind: Literal['binary', 'polar', 'lm', 'log-lm'],
                       x_vocab: Sequence[T],
                       y_vocab: Sequence[T],
                       target: Fn[[Sequence[T]], float],
                       spectral_specs: Mapping[str, Sequence[V]],
                       t_data: Sequence[Sequence[T]],
                       v_data: Sequence[Sequence[T]],
                       e_data: Sequence[Sequence[T]] = (),
                       *,
                       shortcut_loss: float = 1e-6,
                       shortcut_tern: float = .99,
                       period: int = .1,
                       device: str = 'cpu:0',
                       seed: int = 42,
                       **metadata) -> Tuple[WFSA, Tensor]:
    """

    Exhaustively searches hyper-parameter space for Spectral learning. The best model, as per the validation loss
    is saved to disk, along with metadata, the latter to a registry file.

    The spectral parameters are:

        'basis_algos':  basis choice algorithms, one of 'freq-rank', 'length'
        'factor_algos': factorisation algorithms, one of 'svd', 'nmf'
        'k_prefs':      maximum numbers of most common prefixes ('freq-rank') or maximum lengths of prefixes ('length')
        'k_suffs':      maximum numbers of most common suffixes ('freq-rank') or maximum lengths of suffixes ('length')
        'inits':        initialisation modes for nmf, one of 'svd', 'random', 'nndsvd', 'nndsvda', 'nndsvdar'
        'shuffles':     shuffling flags for nmf, one of True, False
        'tern_tols':     the minimum distances from +-1 and 0 to consider a parameter binary
        't_loss_fns':   training loss functions
        'v_loss_fns':   validation loss functions

    :param kind: the type of WFSA to learn, one of 'binary', 'polar', 'lm', 'log-lm'
    :param x_vocab: input vocabulary
    :param y_vocab: output vocabulary
    :param target: the function that provides a target for each sequence in the data partitions
    :param spectral_specs: the values for all the parameters to use in the spectral learning search
    :param t_data: the training data
    :param v_data: the validation data
    :param e_data: the test (evaluation) data
    :param shortcut_loss: the minimum loss to consider the task learned; shortcuts the learning loop
    :param shortcut_tern: the minimum ternariness to consider good enough, once the shortcut loss is achieved
    :param period: the logging period as percentage of the total search runs
    :param device: the device to run the search on
    :param seed: the random generator seed for reproducibility
    :param metadata: extra data to save along with the models
    :return: the best WFSA along with the validation loss
    """
    check(target, Fn)
    check(spectral_specs, Mapping, lambda: len(spectral_specs))
    check(t_data, Sequence, lambda: len(t_data)).check(v_data, Sequence, lambda: len(v_data)).check(e_data, Sequence)
    check(shortcut_loss, float, lambda: 0 <= shortcut_loss)
    check(shortcut_tern, float, lambda: 0 <= shortcut_tern <= 1)
    check(period, float, lambda: 0 <= period <= 1)
    check(device, str)

    make_deterministic(seed)
    runs = tuple(_combine(**spectral_specs))
    period = max(1, int(period * len(runs)))

    LOG.info(_search_info(spectral_specs | dict(t_data=len(t_data), v_data=len(v_data), runs=len(runs)) | metadata))

    best_wfsa, best_hps, best_loss, best_ternariness = None, {}, MAX_LOSS, -1

    data_kind = kind.replace('log-', '') if 'lm' in kind else ('class' + ('-polar' if kind == 'polar' else ''))
    t_dataset, v_dataset = (as_dataset(data, target, x_vocab, y_vocab, kind=data_kind) for data in [t_data, v_data])

    targets = estimate_targets(t_data, kind=kind)

    for run, hyper_spec in enumerate(runs, start=1):

        if not run % period:
            LOG.info(_run_info(hyper_spec, len(runs), run))

        basis_algo, factor_algo, k_pref, k_suff, init, shuffle, tern_tol, t_loss_fn, v_loss_fn = hyper_spec.values()
        check(tern_tol, float, lambda: 0 <= tern_tol <= .5)

        kwargs = DEFAULT_KWARGS | (dict(init=init, shuffle=shuffle, algo='nmf') if factor_algo == 'nmf' else {})
        wfsa = learn_wfsa(kind, t_data, targets, f'{basis_algo}={k_pref}:{k_suff}', x_vocab, **kwargs).to(device)

        t_loss = compute_loss(wfsa, t_dataset, t_loss_fn)
        v_loss = compute_loss(wfsa, v_dataset, v_loss_fn)
        ternariness = _avg_ternariness_of(wfsa.parameters(), tern_tol)

        if (v_loss < best_loss) or (v_loss == best_loss and ternariness > best_ternariness):
            if ternariness > best_ternariness:
                best_ternariness = ternariness

            LOG.info(
                _run_info(dict(t_loss=t_loss, v_loss=v_loss, bin=ternariness, dim=len(wfsa.initial)), len(runs), run))

            best_wfsa, best_hps, best_loss = wfsa, hyper_spec, v_loss

            if best_loss <= shortcut_loss and best_ternariness >= shortcut_tern:
                LOG.info('short-circuiting...')
                break

    LOG.info(_search_info(dict(v_loss=best_loss, bin=best_ternariness) | best_hps | metadata))

    reg = make_metadata(loss=best_loss,
                        bin=best_ternariness,
                        algo=best_hps['factor_algo'],
                        basis=best_hps['basis_algo'],
                        prefs=best_hps['k_pref'],
                        suffs=best_hps['k_suff'],
                        init_=best_hps['init'],
                        shuffle=best_hps['shuffle'],
                        v=len(x_vocab),
                        dim=best_wfsa.init.shape[0],
                        **metadata)
    register_model(best_wfsa, reg)

    if e_data:
        e_dataset = as_dataset(e_data, target, x_vocab, y_vocab, kind=data_kind)
        e_loss = compute_loss(best_wfsa, e_dataset, best_hps['v_loss_fn'])
        LOG.info(f'test loss [{e_loss:.3e}]')

    return best_wfsa, best_loss


def _avg_ternariness_of(params: Iterable[Tensor], tol: float) -> float:
    params = tuple(params)
    return sum(_ternariness_of(p, tol, normed=False) for p in params) / sum(p.numel() for p in params)


def _ternariness_of(weights: Tensor, tol: float = 1e-2, normed: bool = True) -> float:
    """ computes the (possibly normalised) number of parameters that are close-to-binary in the given tensor"""
    s = (weights.abs() <= 0 + tol).count_nonzero() + (
            (weights.abs() > 1 - tol) & (weights.abs() < 1 + tol)).count_nonzero()

    return s / (weights.numel() if normed else 1)


def _combine(**specs: Sequence[V]) -> Iterable[Dict[str, V]]:
    hp_keys = [key[:-1] for key in specs.keys()]

    return map(lambda vals: dict(zip(hp_keys, vals)), product(*(vals or [None] for vals in specs.values())))


def _search_info(params: Dict[str, V]) -> str:
    msg = ''

    for k, v in params.items():
        _format = ('4,d' if isinstance(v, int) else
                   ' .3e' if isinstance(v, float | Tensor) else
                   '10.10s' if isinstance(v, str) else
                   '')
        msg += f'{k}=[{v:{_format}}] '.replace('[[', '[(').replace(']]', ')]')

    return msg


def _run_info(specs: Dict[str, V], num_runs: int, run: int) -> str:
    return f'[{run:4,d} / {num_runs:4,d}] ' + _search_info(specs)
