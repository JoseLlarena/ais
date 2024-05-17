"""
Classes and functions to implement model training by Stochastic Gradient Descent
"""
from __future__ import annotations

from logging import getLogger
from typing import Tuple, TypeVar, Any, Iterable, Mapping, Dict
from warnings import filterwarnings

from torch import Tensor, tensor, inference_mode
from torch.nn import Parameter, Module
from torch.nn.init import uniform_
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import Dataset

from ais import check, Fn, make_deterministic
from ais.data import make_loader, EMPTY_DATASET
from ais.models import AdapterModel
from ais.training import NAME_TO_CONSTRUCTOR, NAME_TO_INIT, NAME_TO_LOSS_FN, NAME_TO_TRACKER, NAME_TO_OPTIMISER, \
    NAME_TO_SCHEDULER, HP_KEYS, evaluate, Optimiser, epoch_step, Batch

M, V = TypeVar('M', bound=AdapterModel), TypeVar('V', str, int, float)

LOG = getLogger(__package__)

filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.linear')
filterwarnings('ignore', category=UserWarning, module='torch.autograd.graph')


class Optim:
    """
    Encapsulates optimisation hyper-parameters to decouple them from the model's parameters.
    """

    def __init__(self, sched: Dict, opt: Dict):
        self._opt_props = opt
        self._sched_props = sched  # TODO USE SIMPLENAMESPACE
        self._optimiser = None

    def sched(self, **kwargs) -> Optim:
        _sched = {**dict(self._sched_props), **kwargs}

        return Optim(_sched, dict(self._opt_props))

    def opt(self, **kwargs) -> Optim:
        _opt = {**dict(self._opt_props), **kwargs}

        return Optim(dict(self._sched_props), _opt)

    @property
    def optimiser(self) -> Optimiser | None:
        return self._optimiser

    def set_epochs(self, epochs: int) -> Optim:
        if self._sched_props['kind'] == OneCycleLR:
            self._sched_props['epochs'] = epochs
        elif self._sched_props['kind'] == CosineAnnealingLR:
            self._sched_props['T_max'] = epochs

        return self

    def set_num_batches(self, num_batches: int) -> Optim:
        if self._sched_props['kind'] == OneCycleLR:
            self._sched_props['steps_per_epoch'] = num_batches

        return self

    def set_total_steps(self, num_steps: int) -> Optim:
        if self._sched_props['kind'] == OneCycleLR:
            self._sched_props['total_steps'] = num_steps

        return self

    def set_params(self, params: Iterable[Parameter]) -> Optim:
        opt_class = self._opt_props.pop('kind')
        sched_class = self._sched_props.pop('kind')

        opt = opt_class(params, **self._opt_props)

        self._optimiser = Optimiser(params, opt, sched_class(opt, **self._sched_props))

        self._opt_props['kind'] = opt_class
        self._sched_props['kind'] = sched_class

        return self

    def __str__(self):
        _str = '' + self._opt_props['kind'].__name__.ljust(6, ' ')[:6]
        for key, value in self._opt_props.items():
            if key != 'kind':
                f = '1.1e' if isinstance(value, float) else ''
                _str += f':{value:{f}}'

        _str += '; ' + self._sched_props['kind'].__name__.ljust(10, ' ')[:10]
        for key, value in self._sched_props.items():
            if key not in {'kind', 'epochs', 'steps_per_epoch'}:
                f = '1.1e' if isinstance(value, float) else ''
                _str += f' {key}:{value:{f}}'

        return _str.replace('0.', '.')


def train_express(model_spec: Mapping[str, V],
                  sgd_spec: Mapping[str, V],
                  t_data: Dataset,
                  v_data: Dataset,
                  e_data: Dataset = EMPTY_DATASET,
                  tracker: str = 'basic',
                  period: float = .5,
                  device: str = 'cuda:0',
                  seed: int = 42) -> Tuple[M, Tensor, Tensor, Tensor | None]:
    """
    Trains models with SGD using a simple string-based interface.

    The SGD hyper-parameters are:
        'epochs':      epochs
        'bs':          batch size
        'init':        initialisation function
        't_loss_fn':   training loss function
        'v_loss_fn':   validation loss function
        'lr':          learning rate for optimiser
        'max_lr':      maximum learning rate for scheduler
        'sched':       scheduler
        'opt':         optimiser

    :param model_spec: specification for model hyper-parameters, containing all the values for all the model features
    :param sgd_spec: specification for sgd hyper-parameters, containing all values for all learning parameters
    :param t_data: the training dataset
    :param v_data: the validation dataset
    :param e_data: the test (evaluation) dataset, possibly empty
    :param tracker: the type of function to track the search process
    :param period: the logging period as percentage of the total search runs
    :param device: the device to run the search on
    :param seed: the random generator seed for reproducibility
    :return: the trained model, along with the training, validation and test losses, the latter will be None if no test
        dataset is provided
    """
    check(model_spec, Mapping).check(sgd_spec, Mapping)
    check(t_data, Dataset).check(v_data, Dataset)
    check(val=tracker in NAME_TO_TRACKER).check(period, float, lambda: 0 < period <= 1)
    check(device, str).check(seed, int)
    make_deterministic(seed)
    epochs, bs, init, t_loss_fn, v_loss_fn, lr, max_lr, sched, opt = (sgd_spec[k] for k in HP_KEYS)

    model_spec = dict(model_spec)
    constructor = NAME_TO_CONSTRUCTOR[model_spec.pop('arch')]
    model = AdapterModel(initialise(constructor(**model_spec), NAME_TO_INIT[init]).to(device))

    t_loader = make_loader(t_data, bs, shuffled=True)

    optim = (Optim(sched=dict(kind=NAME_TO_SCHEDULER[sched], max_lr=max_lr),
                   opt=dict(kind=NAME_TO_OPTIMISER[opt], lr=lr))
             .set_epochs(epochs)
             .set_num_batches(len(t_loader))
             .set_params(model.parameters()))

    model, t_loss, v_loss = train(model,
                                  t_loader,
                                  make_loader(v_data, len(v_data) if bs > 0 else bs, shuffled=False),
                                  epochs,
                                  NAME_TO_LOSS_FN[t_loss_fn.replace('-', '_')],
                                  optim.optimiser,
                                  NAME_TO_LOSS_FN[v_loss_fn.replace('-', '_')],
                                  NAME_TO_TRACKER[tracker](),
                                  max(1, int(period * epochs)))

    return model, t_loss, v_loss, evaluate(model, e_data, bs, v_loss_fn)


def train(model: M,
          t_data: Iterable[Batch],
          v_data: Iterable[Batch],
          epochs: int,
          t_loss_fn: Fn,
          optimiser: Optimiser,
          v_loss_fn: Fn,
          tracker: Fn[[int, Tensor, Tensor], Any],
          track_period: int) -> Tuple[M, Tensor, Tensor]:
    """
    Trains a model for the given number of epochs and computes the average per-batch training and validation losses. If
    the validation data is empty, the loss will be returned as zero. The training loop will be tracked as per the
    given tracker function every ``track_period`` epochs (i.e., 0, 1, track_period*n..., t-1)

    An extra non-learning 0-epoch is run first to enable tracking of pre-training loss.

    The breakdown of this function's functionality is as follows:

        - (indirectly) samples data

        - feeds training data to model (delegated)
        - calculates training loss  (delegated)
        - calculates gradients  (delegated)
        - updates weights  (delegated)
        - updates optimiser  (delegated)

        - feeds validation data to model  (delegated)
        - calculates validation loss  (delegated)

        - tracks progress (delegated)
        - returns trained model, training loss and validation loss

    :param model: the model to be trained, assumed to live in a single device, accessible through a `dev` property
    :param t_data: the data to train the model, as a collection of input-output pairs
    :param v_data: the data to evaluate the model, as a collection of input-output pairs
    :param epochs: the number of epochs to train the model for
    :param t_loss_fn: training loss function, must do sum-reduction so it can be summed over batches then averaged
    :param v_loss_fn: validation loss function, must do sum-reduction so it can be summed over batches then averaged
    :param optimiser: the optimiser for training of the model
    :param tracker: the function to track per-epoch progress
    :param track_period: the period to track an epoch at the end of
    :return: a trained model, example-averaged training loss and example-averaged validation loss
    """
    check(model, AdapterModel)
    check(t_data, Iterable).check(v_data, Iterable).check(epochs, int, lambda: epochs > 0)
    check(t_loss_fn, Fn).check(v_loss_fn, Fn).check(optimiser, Optimiser)
    check(tracker, Fn).check(track_period, int, lambda: track_period >= 0)

    train_loss = tensor(0., device=model.dev)
    val_loss = tensor(0., device=model.dev)
    last_epoch = -1

    for epoch in range(epochs + 1):
        model, train_loss = epoch_step(model.train(epoch != 0), t_data, optimiser, t_loss_fn)

        if epoch == 1 or (not epoch % track_period):
            model, val_loss = epoch_step(model.train(False), v_data, optimiser, v_loss_fn)
            tracker(epoch, train_loss, val_loss)

        last_epoch = epoch

    if last_epoch % track_period:  # FIXME EXPLAIN THIS
        tracker(epochs, train_loss, val_loss)

    return model, train_loss, val_loss


@inference_mode()
def initialise(net: M, init: Fn | None = None) -> M:
    check(net, Module).check(init, Fn | None)

    for param in (net.parameters() if init else ()):
        (init if (init.__name__ == 'sparse_' and param.ndim == 2) else
         init if (init.__name__ != 'sparse_' and param.ndim > 1) else
         uniform_)(param)

    return net
