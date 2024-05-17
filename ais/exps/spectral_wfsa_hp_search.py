"""
Searches parameter space for the best WFSA with Spectral Learning, for binary classification and LM tasks
"""
import os
from itertools import product
from logging import getLogger
from typing import Tuple

from ais import Fn, make_deterministic, outm, px, config_logging
from ais.boolean_data import make_bool_partitions, TRUE, XOR, EQUIV, OR, NOR, AND, NAND, IF, NIF, FIRST, NFIRST, LAST, \
    NLAST, INV, NINV, BOOLS
from ais.data import as_dataset, class_target
from ais.hp_search import spectral_hp_search
from ais.spectral import estimate_targets
from ais.training import compute_loss

LOG = config_logging(getLogger(__package__))


def classifier_wfsa(task: Fn, loss_fn: str, kind: str = 'binary'):
    make_deterministic(seed=42)

    t_data, _, e_data = make_bool_partitions(task, max_lengths=[12, 0, 24], ns=[-1, 0, 100])
    v_data, *_ = make_bool_partitions(task, min_len=13, max_lengths=[18, 0, 0], ns=[100, 0, 0])
    t_data, v_data = tuple(t_data), tuple(v_data)

    targets = estimate_targets(t_data + v_data, kind=kind)

    def target(x) -> Tuple[float]:
        return targets[x],

    wfsa, *_ = spectral_hp_search(kind=kind,
                                  x_vocab=BOOLS,
                                  y_vocab=BOOLS,
                                  target=target,
                                  spectral_specs=dict(basis_algos=['freq-rank', 'length'][:1],
                                                      factor_algos=['nmf', 'svd'],
                                                      k_prefs=range(1, 11),
                                                      k_suffs=range(1, 11),
                                                      inits=['nndsvd', 'nndsvda', 'svd', 'nndsvdar', 'random'],
                                                      shuffles=[False, True],
                                                      tern_tols=[1e-2],
                                                      t_loss_fns=[loss_fn],
                                                      v_loss_fns=['zero-one-3']),
                                  t_data=t_data,
                                  v_data=v_data,
                                  shortcut_loss=1e-6,
                                  shortcut_tern=.99,
                                  fn=task.__name__.upper())

    print('\nBEST WFSA')
    outm(wfsa.initial, sep='>>> ', fracs=2, tol=1e-4)
    outm(wfsa.transitions, sep='>>> ', fracs=2, tol=1e-4)
    outm(wfsa.final, sep='>>> ', fracs=2, tol=1e-4)

    e_dataset = as_dataset(e_data, px(class_target, task=task), x_vocab=BOOLS, y_vocab=BOOLS, kind='class')
    e_loss = compute_loss(wfsa, e_dataset, 'zero-one-3')
    LOG.info(f'test loss [{e_loss:.3e}]\n')


def lm_wfsa(task: Fn):
    make_deterministic(seed=42)

    data = make_bool_partitions(task, max_lengths=[12, 18, 24], ns=[-1, 100, 100])
    t_data, v_data, e_data = map(tuple, data)
    targets = estimate_targets(t_data + v_data, kind='lm')

    wfsa, *_ = spectral_hp_search(kind='log-lm',
                                  x_vocab=BOOLS,
                                  y_vocab=BOOLS,
                                  target=lambda x: [1 - targets[x], targets[x]],
                                  spectral_specs=dict(basis_algos=['freq-rank', 'length'][:1],
                                                      factor_algos=['nmf', 'svd'],
                                                      k_prefs=range(1, 11),
                                                      k_suffs=range(1, 11),
                                                      inits=['nndsvd', 'nndsvda', 'svd', 'nndsvdar', 'random'],
                                                      shuffles=[False, True],
                                                      tern_tols=[1e-2],
                                                      t_loss_fns=['kl'],
                                                      v_loss_fns=['kl']),
                                  t_data=t_data,
                                  v_data=v_data,
                                  e_data=e_data,
                                  shortcut_loss=1e-8,
                                  shortcut_tern=.99,
                                  fn=task.__name__.upper())

    print('\nBEST WFSA')
    outm(wfsa.initial, sep='>>> ', fracs=2, tol=1e-4)
    outm(wfsa.transitions, sep='>>> ', fracs=2, tol=1e-4)
    outm(wfsa.final, sep='>>> ', fracs=2, tol=1e-4)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for fn, loss in product(
            [TRUE, IF, XOR, OR, AND, EQUIV, NOR, NAND, NIF, FIRST, NFIRST, LAST, NLAST, INV, NINV],
            ['l2', 'xent', 'zero-one-3']):
        classifier_wfsa(fn, loss)

    for fn in [TRUE, IF, XOR, OR, AND, EQUIV, NOR, NAND, NIF, FIRST, NFIRST, LAST, NLAST, INV, NINV]:
        lm_wfsa(fn)
