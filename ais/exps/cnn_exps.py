"""
Trains, if needed, and visualises AND, OR and XOR CNNs
"""
import os

from torch import load, save

from ais import make_deterministic, Fn
from ais.boolean_data import OR, AND, XOR
from ais.exps import make_standard_partitions
from ais.models import OLCNN
from ais.sgd import train_express
from ais.tracing import as_traced_cnn
from ais.viz.display import viz_ffw_model

SETUP_TO_HYPS = \
    {
        XOR: dict(seed=7263971, hid_dim=16, eps=200, bs=-2, lr=1e-5, max_lr=1e-0, init='xaviern'),
        OR: dict(seed=2536146025, hid_dim=2, eps=200, bs=-128, lr=1e-5, max_lr=1e-0, init='ortho'),
        AND: dict(seed=1812140441, hid_dim=1, eps=200, bs=-128, lr=1e-5, max_lr=1e-0, init='xaviern')
    }

SETUP_TO_VIZ = \
    {
        XOR: dict(elev=19, azim=-86, quant_rad=.05),
        OR: dict(elev=18, azim=-177, quant_rad=.9),
        AND: dict(elev=18, azim=70, quant_rad=.1)
    }


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for task in (OR, AND, XOR):
        try:
            cnn = load(f'{task.__name__}-CNN.pt', weights_only=True)
        except Exception as e:
            print(e)
            cnn = train_cnn(task, **SETUP_TO_HYPS[task])
            try:
                save(cnn, f'{task.__name__}-CNN.pt')
            except Exception as e2:
                print(e2)

        cnn = as_traced_cnn(cnn)
        viz_ffw_model(cnn, f'{task.__name__}-CNN', **SETUP_TO_VIZ[task])


def main_figures():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for task in (OR, AND, XOR):
        cnn = load(f'{task.__name__}-CNN.pt', weights_only=True)
        cnn = as_traced_cnn(cnn)
        viz_ffw_model(cnn, f'{task.__name__}-CNN', save=True, **SETUP_TO_VIZ[task])


def train_cnn(task: Fn, hid_dim: int, eps: int, bs: int, lr: float, max_lr: float, init: str, seed: int) -> OLCNN:
    make_deterministic(seed=42)

    t_data, v_data, e_data, x_vocab, y_vocab = make_standard_partitions(task, scheme='class-bos')

    cnn, *_ = train_express(model_spec=dict(arch='cnn',
                                            hid_dim=hid_dim,
                                            k=2,
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
                                          v_loss_fn='zero-one-1'),
                            t_data=t_data,
                            v_data=v_data.cat(t_data),
                            e_data=e_data,
                            seed=seed)

    return cnn.train(False).cpu().adapted


if __name__ == '__main__':
    main()
    main_figures()
