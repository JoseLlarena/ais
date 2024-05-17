"""
Searches parameter space for the best encoder-Transformer
"""
import os
from itertools import product

from torch.nn import ReLU

from ais import Fn, make_deterministic
from ais.boolean_data import XOR, OR, AND
from ais.exps import make_standard_partitions
from ais.hp_search import sgd_hp_search


def encoder_transformer(task: Fn, loss_fn: str, reduction: str):
    make_deterministic(seed=42)

    t_data, v_data, e_data, x_vocab, y_vocab = make_standard_partitions(task,
                                                                        scheme='bert-id' if reduction == 'first' else
                                                                        'class-bos-id')

    sgd_hp_search(model_specs=dict(archs=['gpt'],
                                   hid_dims=[1, 2, 3, 4, 8, 16, 32, 64, 128, 256],
                                   factors=[4],
                                   drops=[0],
                                   lengths=[24],
                                   vs=[len(x_vocab)],
                                   out_dims=[len(y_vocab)],
                                   modes=['class' + '-' + reduction + ('-xent' if loss_fn == 'xent' else '')],
                                   nonlins=[ReLU()]),

                  sgd_specs=dict(epochss=[201],
                                 bss=[2048, 1024, 512, 256, -128, -64, -32, -16, -8, -4, -2, -1][-8:],
                                 optimisers=['adam'],
                                 lrs=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                                 schedulers=['onecycle'],
                                 max_lrs=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                                 inits=['ortho', 'xaviern', 'xavieru'],
                                 t_loss_fns=[loss_fn],
                                 v_loss_fns=['zero-one-3']),
                  t_data=t_data,
                  v_data=v_data.cat(t_data),
                  e_data=e_data,
                  task=task)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for fn, loss, reduction in product([OR, AND, XOR], ['l2', 'xent'], ['sum', 'first']):

        if not reduction:
            pass
        else:
            encoder_transformer(fn, loss, reduction)
