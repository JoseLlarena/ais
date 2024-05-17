"""
Trains, if needed, and visualises AND, OR, XOR and Chiang-XOR encoder-Transformers.
"""
import os
from logging import getLogger

from torch import load, save
from torch.nn import ReLU

from ais import make_deterministic, Fn
from ais.boolean_data import OR, AND, XOR
from ais.exps import make_standard_partitions
from ais.model_creation import make_chiang_transformer
from ais.models import GPT
from ais.sgd import train_express
from ais.tracing import as_traced_transformer
from ais.viz.display import viz_ffw_model

LOG = getLogger(__package__)

SETUP_TO_HYPS = \
    {
        XOR: dict(seed=34780500, hid_dim=64, eps=200, bs=-128, lr=1e-1, max_lr=1e-2, init='ortho'),
        OR: dict(seed=3301639171, hid_dim=2, eps=201, bs=-128, lr=1e-3, max_lr=1e-0, init='ortho'),
        AND: dict(seed=2584025260, hid_dim=2, eps=201, bs=-128, lr=1e-4, max_lr=1e-1, init='xaviern')
    }

SETUP_TO_VIZ = \
    {
        XOR: dict(elev=19, azim=-86, quant_rad=.01),
        OR: dict(elev=24, azim=146, quant_rad=.01),
        AND: dict(elev=23, azim=132, quant_rad=.01),
        'chiang-tfm': dict(elev=28, azim=-21, quant_rad=.01, seq_length=8)
    }


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for task in (OR, AND, XOR):
        mode = 'bert'
        try:
            model = load(f'{task.__name__}-{mode}-TFM.pt', weights_only=True)
        except Exception as e:
            LOG.exception(e)
            model = train_transformer(task, mode, **SETUP_TO_HYPS[task])
            try:
                save(model, f'{task.__name__}-{mode}-TFM.pt')
            except Exception as e:
                LOG.exception(e)

        model = as_traced_transformer(model)
        viz_ffw_model(model, f'{task.__name__}-{mode}-TFM', kind='tfm', **SETUP_TO_VIZ[task])

        print('=== ' * 40)

    model = as_traced_transformer(make_chiang_transformer(factor=40.).train(False), kind='chiang')
    viz_ffw_model(model, f'XOR-chiang-TFM', kind='chiang-tfm', **SETUP_TO_VIZ['chiang-tfm'])


def main_figures():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for task in (OR, AND, XOR):
        print(task.__name__)
        tfm = load(f'{task.__name__}-bert-TFM.pt', weights_only=True)
        tfm = as_traced_transformer(tfm)
        viz_ffw_model(tfm, f'{task.__name__}-bert-TFM', kind='tfm', save=True, **SETUP_TO_VIZ[task])

    model = as_traced_transformer(make_chiang_transformer(factor=40.).train(False), kind='chiang')
    viz_ffw_model(model, f'XOR-chiang-TFM', kind='chiang-tfm', save=True, **SETUP_TO_VIZ['chiang-tfm'])


def train_transformer(task: Fn, kind: str, hid_dim: int, eps: int, bs: int, lr: float, max_lr: float, init: str,
                      seed: int) -> GPT:
    make_deterministic(seed=42)

    scheme = 'class-bos-id' if kind == 'sum' else 'bert-id'
    t_data, v_data, e_data, x_vocab, y_vocab = make_standard_partitions(task, scheme=scheme)

    kind = 'class-sum-xent' if kind == 'sum' else 'class-first-xent'

    transformer, *_ = train_express(model_spec=dict(arch='gpt',
                                                    hid_dim=hid_dim,
                                                    nonlin=ReLU(),
                                                    mode=kind,
                                                    length=24,
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

    return transformer.train(False).cpu().adapted


if __name__ == '__main__':
    main()
    main_figures()
