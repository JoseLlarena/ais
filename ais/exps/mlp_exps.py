"""
Trains, if needed, and visualises AND, OR, XOR and Rumelhart-XOR MLPs
"""
import os

from torch import load, save
from torch.nn.functional import relu

from ais import make_deterministic, Fn
from ais.boolean_data import OR, AND, XOR
from ais.exps import make_standard_partitions
from ais.model_creation import make_rumelhart_mlp
from ais.models import OLMLP
from ais.sgd import train_express
from ais.tracing import as_traced_mlp
from ais.viz.display import viz_ffw_model

SETUP_TO_HYPS = \
    {
        XOR: dict(seed=7263971, hid_dim=16, eps=200, bs=-2, lr=1e-5, max_lr=1e-0, init='xaviern'),
        OR: dict(seed=1881625505, hid_dim=2, eps=1000, bs=-32, lr=1e-1, max_lr=1e-1, init='xaviern'),
        AND: dict(seed=2281194061, hid_dim=2, eps=1000, bs=-128, lr=1e-5, max_lr=1e-1, init='xavieru')
    }

SETUP_TO_VIZ = \
    {
        XOR: dict(elev=19, azim=-86, steps=27, quant_rad=.01),
        OR: dict(elev=24, azim=146, steps=27, quant_rad=.01),
        AND: dict(elev=34, azim=79, steps=27, quant_rad=.01),
        'rumelhart-mlp': dict(elev=19, azim=-55, quant_rad=.01, seq_length=3)
    }


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for task in (OR, AND, XOR)[:2]:
        try:
            model = load(f'{task.__name__}-MLP.pt', weights_only=True)
        except Exception as e:
            print(e)
            model = train_mlp(task, **SETUP_TO_HYPS[task])
            try:
                save(model, f'{task.__name__}-MLP.pt')
            except Exception as e:
                print(e)

        viz_ffw_model(as_traced_mlp(model, kind='learned'), f'{task.__name__}-MLP', **SETUP_TO_VIZ[task])

    model = as_traced_mlp(make_rumelhart_mlp(hid_dim=3), kind='rumelhart')
    viz_ffw_model(model, f'XOR-rumelhart-MLP', kind='rumelhart-mlp', **SETUP_TO_VIZ['rumelhart-mlp'])


def main_figures():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for task in (OR, AND, XOR)[:2]:
        mlp = load(f'{task.__name__}-MLP.pt', weights_only=True)
        mlp = as_traced_mlp(mlp, kind='learned')
        viz_ffw_model(mlp, f'{task.__name__}-MLP', save=True, **SETUP_TO_VIZ[task])

    mlp = as_traced_mlp(make_rumelhart_mlp(hid_dim=3), kind='rumelhart')
    viz_ffw_model(mlp, f'XOR-rumelhart-MLP', kind='rumelhart-mlp', save=True, **SETUP_TO_VIZ['rumelhart-mlp'])


def train_mlp(task: Fn, hid_dim: int, eps: int, bs: int, lr: float, max_lr: float, init: str, seed: int) -> OLMLP:
    make_deterministic(seed=42)

    t_data, v_data, e_data, x_vocab, y_vocab = make_standard_partitions(task, scheme='class-bos')

    mlp, *_ = train_express(model_spec=dict(arch='mlp',
                                            hid_dim=hid_dim,
                                            nonlin=relu,
                                            decoder='sum-xent',
                                            v=len(x_vocab),
                                            out_dim=len(y_vocab)),
                            sgd_spec=dict(epochs=eps,
                                          bs=bs,
                                          optimiser='adam',
                                          lr=lr,
                                          scheduler='onecycle',
                                          max_lr=max_lr,
                                          init=init,
                                          t_loss_fn='xent',
                                          v_loss_fn='zero-one-3'),
                            t_data=t_data,
                            v_data=v_data.cat(t_data),
                            e_data=e_data,
                            seed=seed)

    return mlp.train(False).cpu().adapted


if __name__ == '__main__':
    main()
    main_figures()
