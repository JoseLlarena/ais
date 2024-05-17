"""
Trains, if needed, and visualises AND, OR and XOR, SRNNs, GRUs and LSTMS
"""
import os
from itertools import product

from torch import load, save
from torch.nn import ReLU

from ais import make_deterministic, Fn
from ais.boolean_data import XOR, AND, OR
from ais.exps import make_standard_partitions
from ais.models import OLSRNN, OLLSTM, OLGRU
from ais.sgd import train_express
from ais.tracing import as_traced_rnn
from ais.viz.display import viz_recurrent_model

SETUP_TO_HYPS = \
    {
        (XOR, 'softmax-srnn'): dict(seed=3729774825, hid_dim=3, eps=750, bs=-32, lr=1e-0, max_lr=1e-1, init='xavieru'),
        (OR, 'softmax-srnn'): dict(seed=4076983999, hid_dim=2, eps=750, bs=-32, lr=1e-3, max_lr=1e-0, init='ortho'),
        (AND, 'softmax-srnn'): dict(seed=81698624, hid_dim=2, eps=250, bs=-16, lr=1e-5, max_lr=1e-0, init='xaviern'),

        (XOR, 'lin-srnn'): dict(seed=546335398, hid_dim=3, eps=750, bs=-16, lr=1e-1, max_lr=1e-1, init='ortho'),
        (OR, 'lin-srnn'): dict(seed=215967068, hid_dim=2, eps=750, bs=-32, lr=1e-1, max_lr=1e-1, init='ortho'),
        (AND, 'lin-srnn'): dict(seed=2613969982, hid_dim=2, eps=750, bs=-16, lr=1e-5, max_lr=1e-1, init='ortho'),

        (XOR, 'softmax-gru'): dict(seed=2536146025, hid_dim=2, eps=200, bs=-128, lr=1e-5, max_lr=1e-0, init='ortho'),
        (OR, 'softmax-gru'): dict(seed=2536146025, hid_dim=2, eps=200, bs=-128, lr=1e-5, max_lr=1e-0, init='ortho'),
        (AND, 'softmax-gru'): dict(seed=398340369, hid_dim=1, eps=200, bs=-128, lr=1e-3, max_lr=1e-0, init='xaviern'),

        (XOR, 'lin-gru'): dict(seed=3687729441, hid_dim=2, eps=1000, bs=-16, lr=1e-3, max_lr=1e-0, init='ortho'),
        (OR, 'lin-gru'): dict(seed=2596989704, hid_dim=2, eps=1000, bs=-32, lr=1e-5, max_lr=1e-1, init='xavieru'),
        (AND, 'lin-gru'): dict(seed=2096246684, hid_dim=2, eps=1000, bs=-4, lr=1e-4, max_lr=1e-0, init='xaviern'),

        (XOR, 'softmax-lstm'): dict(seed=1929338154, hid_dim=2, eps=200, bs=-128, lr=1e-4, max_lr=1e-0, init='ortho'),
        (OR, 'softmax-lstm'): dict(seed=4124013886, hid_dim=2, eps=200, bs=-64, lr=1e-5, max_lr=1e-0, init='xavieru'),
        (AND, 'softmax-lstm'): dict(seed=3614942327, hid_dim=1, eps=200, bs=-32, lr=1e-5, max_lr=1e-0, init='xavieru'),

        (XOR, 'lin-lstm'): dict(seed=1285052179, hid_dim=2, eps=1000, bs=-8, lr=1e-4, max_lr=1e-0, init='xavieru'),
        (OR, 'lin-lstm'): dict(seed=4018738574, hid_dim=2, eps=200, bs=-16, lr=1e-0, max_lr=1e-0, init='xaviern'),
        (AND, 'lin-lstm'): dict(seed=1194211864, hid_dim=2, eps=500, bs=-32, lr=1e-3, max_lr=1e-0, init='xaviern')}

SETUP_TO_VIZ = {(XOR, 'softmax-srnn'): dict(elev=33, azim=-40),
                (OR, 'softmax-srnn'): dict(elev=27, azim=34, quant_rad=.2),
                (AND, 'softmax-srnn'): dict(elev=22, azim=29),
                (XOR, 'lin-srnn'): dict(elev=9, azim=-58),
                (OR, 'lin-srnn'): dict(elev=15, azim=-32),
                (AND, 'lin-srnn'): dict(elev=19, azim=-40),

                (XOR, 'softmax-gru'): dict(elev=33, azim=46, quant_rad=.1),
                (OR, 'softmax-gru'): dict(elev=32, azim=131, quant_rad=.1),
                (AND, 'softmax-gru'): dict(elev=26, azim=128, quant_rad=.1),
                (XOR, 'lin-gru'): dict(elev=11, azim=-53),
                (OR, 'lin-gru'): dict(elev=22, azim=106),
                (AND, 'lin-gru'): dict(elev=14, azim=-180),

                (XOR, 'softmax-lstm'): dict(elev=25, azim=-171),
                (OR, 'softmax-lstm'): dict(elev=20, azim=27),
                (AND, 'softmax-lstm'): dict(elev=20, azim=52),
                (XOR, 'lin-lstm'): dict(elev=18, azim=-177),
                (OR, 'lin-lstm'): dict(elev=11, azim=-28),
                (AND, 'lin-lstm'): dict(elev=32, azim=59)}


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for task in (OR, AND, XOR):

        for (fn, arch), out in product(zip([train_srnn, train_gru, train_lstm], ['srnn', 'gru', 'lstm']),
                                       ['softmax', 'lin']):
            try:
                rnn = load(f'{task.__name__}-{out}-{arch.upper()}.pt', weights_only=True)
            except Exception as e:
                print(e)
                rnn = fn(task, out, **SETUP_TO_HYPS[task, f'{out}-{arch}'])
                try:
                    save(rnn, f'{task.__name__}-{out}-{arch.upper()}.pt')
                except Exception as e:
                    print(e)
            rnn = as_traced_rnn(rnn)
            viz_recurrent_model(rnn, f'{task.__name__}-{out}-{arch.upper()}',
                                **SETUP_TO_VIZ[task, f'{out}-{arch.upper()}'.lower()])


def main_figures():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for task in (OR, AND, XOR):

        for arch, out in product(['srnn', 'gru', 'lstm'], ['softmax', 'lin']):
            rnn = load(f'{task.__name__}-{out}-{arch.upper()}.pt', weights_only=True)
            rnn = as_traced_rnn(rnn)
            viz_recurrent_model(rnn,
                                f'{task.__name__}-{out}-{arch.upper()}',
                                save=True,
                                **SETUP_TO_VIZ[task, f'{out}-{arch.upper()}'.lower()])


def train_srnn(task: Fn, out: str, hid_dim: int, eps: int, bs: int, lr: float, max_lr: float, init: str, seed: int) \
        -> OLSRNN:
    make_deterministic(seed=42)

    t_data, v_data, e_data, x_vocab, y_vocab = make_standard_partitions(task)

    srnn, *_ = train_express(model_spec=dict(arch='srnn',
                                             hid_dim=hid_dim,
                                             nonlin=ReLU,
                                             mode='class' + ('' if out == 'lin' else '-xent'),
                                             v=len(x_vocab),
                                             out_dim=len(y_vocab)),

                             sgd_spec=dict(epochs=eps,
                                           bs=bs,
                                           optimiser='adam',
                                           lr=lr,
                                           scheduler='onecycle',
                                           max_lr=max_lr,
                                           init=init,
                                           t_loss_fn='l2' if out == 'lin' else 'xent',
                                           v_loss_fn='zero-one-3'),
                             t_data=t_data,
                             v_data=v_data.cat(t_data),
                             e_data=e_data,
                             seed=seed)

    return srnn.train(False).cpu().adapted


def train_gru(task: Fn, out: str, hid_dim: int, eps: int, bs: int, lr: float, max_lr: float, init: str, seed: int) \
        -> OLGRU:
    make_deterministic(seed=42)

    t_data, v_data, e_data, x_vocab, y_vocab = make_standard_partitions(task)

    gru, *_ = train_express(model_spec=dict(arch='gru',
                                            hid_dim=hid_dim,
                                            mode='class' + ('' if out == 'lin' else '-xent'),
                                            v=len(x_vocab),
                                            out_dim=len(y_vocab)),

                            sgd_spec=dict(epochs=eps,
                                          bs=bs,
                                          optimiser='adam',
                                          lr=lr,
                                          scheduler='onecycle',
                                          max_lr=max_lr,
                                          init=init,
                                          t_loss_fn='l2' if out == 'lin' else 'xent',
                                          v_loss_fn='zero-one-3'),
                            t_data=t_data,
                            v_data=v_data.cat(t_data),
                            e_data=e_data,
                            seed=seed)

    return gru.train(False).cpu().adapted


def train_lstm(task: Fn, out: str, hid_dim: int, eps: int, bs: int, lr: float, max_lr: float, init: str, seed: int) \
        -> OLLSTM:
    make_deterministic(seed=42)

    t_data, v_data, e_data, x_vocab, y_vocab = make_standard_partitions(task)

    lstm, *_ = train_express(model_spec=dict(arch='lstm',
                                             hid_dim=hid_dim,
                                             mode='class' + ('' if out == 'lin' else '-xent'),
                                             v=len(x_vocab),
                                             out_dim=len(y_vocab)),

                             sgd_spec=dict(epochs=eps,
                                           bs=bs,
                                           optimiser='adam',
                                           lr=lr,
                                           scheduler='onecycle',
                                           max_lr=max_lr,
                                           init=init,
                                           t_loss_fn='l2' if out == 'lin' else 'xent',
                                           v_loss_fn='zero-one-3'),
                             t_data=t_data,
                             v_data=v_data.cat(t_data),
                             e_data=e_data,
                             seed=seed)

    return lstm.train(False).cpu().adapted


if __name__ == '__main__':
    main()
    main_figures()
