"""
Trains, if needed, and visualises AND, OR and XOR  SGD and Spectral WFSAs.
"""
import os

from torch import inference_mode, load, save

from ais import make_deterministic, outm, px, Fn
from ais.boolean_data import XOR, make_bool_partitions, BOOLS, AND, OR
from ais.data import as_dataset, class_target
from ais.exps import make_standard_partitions
from ais.models import WFSA
from ais.sgd import train_express
from ais.spectral import learn_wfsa
from ais.tracing import as_traced_wfsa
from ais.training import zero_one_loss, compute_loss
from ais.viz.display import viz_recurrent_model

SETUP_TO_HYPS = {(XOR, 'spectral'): dict(prefs=1, suffs=1, init='nndsvd', shuffle=True),
                 (OR, 'spectral'): dict(prefs=1, suffs=8, init='svd', shuffle=False),
                 (AND, 'spectral'): dict(prefs=1, suffs=1, init='nndsvd', shuffle=False),
                 (XOR, 'sgd'): dict(seed=2748778024, hid_dim=2, bs=-64, lr=1e-5, max_lr=1e-1, init='xavieru'),
                 (OR, 'sgd'): dict(seed=3863813067, hid_dim=2, bs=-32, lr=1e-1, max_lr=1, init='xavieru'),
                 (AND, 'sgd'): dict(seed=3831882064, hid_dim=1, bs=-128, lr=1e-5, max_lr=1e-1, init='ortho')}

SETUP_TO_VIZ = {(XOR, 'spectral'): dict(elev=9, azim=-47),
                (OR, 'spectral'): dict(elev=9, azim=-178),
                (AND, 'spectral'): dict(elev=22, azim=123),
                (XOR, 'sgd'): dict(elev=20, azim=-97),
                (OR, 'sgd'): dict(elev=41, azim=150),
                (AND, 'sgd'): dict(elev=23, azim=120)}


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for task in (OR, AND, XOR):

        for learn_fn, method in zip([learn_spectral_wfsa, learn_sgd_wfsa], ['spectral', 'sgd']):

            try:
                wfsa = load(f'{task.__name__}-{method}-WFSA.pt', weights_only=True)
            except Exception as e:
                print(e)
                wfsa = learn_fn(task, **SETUP_TO_HYPS[task, method])
                try:
                    save(wfsa, f'{task.__name__}-{method}-WFSA.pt')
                except Exception as e:
                    print(e)

            viz_recurrent_model(as_traced_wfsa(wfsa), f'{task.__name__}-{method}-WFSA', **SETUP_TO_VIZ[task, method])


@inference_mode()
def main_figures():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for task in (OR, AND, XOR):
        for method in ('spectral', 'sgd'):
            wfsa = load(f'{task.__name__}-{method}-WFSA.pt', weights_only=True)
            viz_recurrent_model(as_traced_wfsa(wfsa),
                                f'{task.__name__}-{method}-WFSA',
                                save=True,
                                **SETUP_TO_VIZ[task, method])


@inference_mode()
def learn_spectral_wfsa(task: Fn, prefs: int, suffs: int, init: str, shuffle: bool, seed: int = 42) -> WFSA:
    make_deterministic(seed=seed)

    t_data, _, e_data = map(tuple, make_bool_partitions(task, max_lengths=[12, 18, 24], ns=[-1, 0, 100]))

    wfsa = learn_wfsa(kind='binary',
                      data=t_data,
                      basis=f'freq-rank={prefs}:{suffs}',
                      algo='nmf',
                      init=init,
                      shuffle=shuffle,
                      base_vocab=BOOLS,
                      tol=.1)

    outm(wfsa.initial, sep='>>> ', fracs=3, tol=1e-6)
    outm(wfsa.transitions, sep='>>> ', fracs=3, tol=1e-6)
    outm(wfsa.final, sep='>>> ', fracs=3, tol=1e-6)

    print()
    for which, data in zip(['training', 'test'], [t_data, e_data]):
        dataset = as_dataset(data, px(class_target, task=task), x_vocab=BOOLS, y_vocab=BOOLS, kind='class')
        loss = compute_loss(wfsa, dataset, px(zero_one_loss, tol=1e-3))
        print(f'{which:8s} loss [{loss:.3f}]')

    return wfsa


def learn_sgd_wfsa(task: Fn, hid_dim: int, bs: int, lr: float, max_lr: float, init: str, seed: int = 42) -> WFSA:
    make_deterministic(seed=42)

    t_data, v_data, e_data, x_vocab, y_vocab = make_standard_partitions(task)

    wfsa, *_ = train_express(model_spec=dict(arch='wfsa',
                                             v=len(x_vocab),
                                             hid_dim=hid_dim,
                                             mode='binary'),
                             sgd_spec=dict(epochs=100,
                                           bs=bs,
                                           optimiser='adam',
                                           lr=lr,
                                           scheduler='onecycle',
                                           max_lr=max_lr,
                                           init=init,
                                           t_loss_fn='l2',
                                           v_loss_fn='zero-one-3'),
                             t_data=t_data,
                             v_data=v_data.cat(t_data),
                             e_data=e_data,
                             seed=seed)

    wfsa = wfsa.train(False).cpu().adapted
    outm(wfsa.initial, sep='>>> ', fracs=3, tol=1e-6)
    outm(wfsa.transitions, sep='>>> ', fracs=3, tol=1e-6)
    outm(wfsa.final, sep='>>> ', fracs=3, tol=1e-6)

    return wfsa


if __name__ == '__main__':
    main()
    main_figures()
