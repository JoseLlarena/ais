"""
Searches parameter space for the best CNN
"""
import os
from itertools import product

from ais import Fn, make_deterministic
from ais.boolean_data import OR, AND, XOR
from ais.exps import make_standard_partitions
from ais.hp_search import sgd_hp_search


def classifier_cnn(task: Fn, loss_fn: str):
    make_deterministic(seed=42)

    kernel = 2
    t_data, v_data, e_data, x_vocab, y_vocab = make_standard_partitions(task, scheme='class-bos')

    sgd_hp_search(model_specs=dict(archs=['cnn'],
                                   vs=[len(x_vocab)],
                                   hid_dims=[1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64],
                                   ks=[kernel],
                                   out_dims=[len(y_vocab)]),

                  sgd_specs=dict(epochss=[10],
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

    for fn, loss in product([OR, AND, XOR], ['l2', 'xent']):
        classifier_cnn(fn, loss)
