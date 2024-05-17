"""
Functions, classes and constants used by various modules to train models
"""
from __future__ import annotations

from logging import getLogger
from typing import TypeVar, Generic, Sequence, Iterable, Tuple, TypeAlias

from torch import Tensor, inference_mode, empty, tensor, set_grad_enabled
from torch.nn import Module, Parameter
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss, huber_loss, cross_entropy, kl_div, \
    binary_cross_entropy
from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, orthogonal_, uniform_
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, NAdam, RAdam, Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ConstantLR, StepLR, LinearLR, LRScheduler
from torch.utils.data import Dataset

from ais import px, check, Fn
from ais.data import RaggedSupeDataset, make_loader, Coder, EMPTY_DATASET
from ais.model_creation import make_wfsa, make_olsrnn, make_olgru, make_ollstm, make_mlp, make_cnn, make_transformer
from ais.models import AdapterModel

T, X, Y = (TypeVar(t) for t in 'TXY')
M = TypeVar('M', bound=AdapterModel)
Batch: TypeAlias = Tuple[Tensor, Tensor]

NAME_TO_CONSTRUCTOR = dict(wfsa=make_wfsa,
                           srnn=make_olsrnn,
                           gru=make_olgru,
                           lstm=make_ollstm,
                           mlp=make_mlp,
                           cnn=make_cnn,
                           gpt=make_transformer)
NAME_TO_INIT = dict(xavieru=xavier_uniform_,
                    xaviern=xavier_normal_,
                    kaimingu=kaiming_uniform_,
                    kaimingn=kaiming_normal_,
                    ortho=orthogonal_,
                    uniform=px(uniform_, a=-1, b=-1),
                    none=None)
NAME_TO_TRACKER = dict(basic=lambda: _loss_tracker)
NAME_TO_OPTIMISER = dict(adam=Adam, nadam=NAdam, radam=RAdam)
NAME_TO_SCHEDULER = dict(onecycle=OneCycleLR, constant=ConstantLR, step=StepLR, linear=LinearLR)
HP_KEYS = 'epochs', 'bs', 'init', 't_loss_fn', 'v_loss_fn', 'lr', 'max_lr', 'scheduler', 'optimiser'
MAX_BATCH_SIZE = 2048
LOG = getLogger(__package__)


def zero_one_loss(y_hats: Tensor, ys: Tensor, tol: float = 1e-5, reduction: str = 'mean') -> Tensor:
    """
    Computes the 0-1 loss between targets and outputs, defined by per-row equality comparison.

    :param y_hats: an Nxk 2nd-order tensor with N k-dimensional one-hot-encoded targets, one per row
    :param ys: an Nxk 2nd-order tensor with N k-dimensional one-hot-encoded outputs, one per row
    :param reduction: type of loss aggregation, one of 'sum', 'mean', 'none'
    :return: a N-sized boolean tensor or a 1-sized float tensor with the losses
    """
    check(y_hats, Tensor, lambda: y_hats.ndim in {2, 3} and y_hats.numel())
    check(ys, Tensor, lambda: ys.shape == y_hats.shape and ys.numel())
    check(tol, float, lambda: tol >= 0)
    check(val=reduction in {'none', 'mean', 'sum'})

    losses = ((y_hats - ys).abs() > tol).max(dim=-1).values.float()

    if reduction == 'none':
        return losses

    return losses.sum() if reduction == 'sum' else losses.mean()


NAME_TO_LOSS_FN = dict(l2=mse_loss,
                       l1=l1_loss,
                       sl1=smooth_l1_loss,
                       huber=huber_loss,
                       xent=cross_entropy,
                       kl=kl_div,
                       binxent=binary_cross_entropy,
                       zero_one_5=px(zero_one_loss, tol=1e-5),
                       zero_one_4=px(zero_one_loss, tol=1e-4),
                       zero_one_3=px(zero_one_loss, tol=1e-3),
                       zero_one_2=px(zero_one_loss, tol=1e-2),
                       zero_one_1=px(zero_one_loss, tol=1e-1))


class Optimiser:
    """
    Encapsulates all parameter optimisation logic, including Pytorch Optimizer, Scheduler and gradient-clipping
    """

    def __init__(self, params: Iterable[Parameter], optimiser: Optimizer | None, scheduler: LRScheduler | None,
                 clip: float = 1):
        self.params = params
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.clip_max = clip

    def step(self) -> Optimiser:
        """
        Runs an optimisation step, consisting of optimiser and scheduler updating. Gradient clipping is also done.

        :return: self
        """
        clip_grad_norm_(self.params, self.clip_max, error_if_nonfinite=True)
        if self.optimiser:
            self.optimiser.step()
        if self.scheduler:
            self.scheduler.step()
        return self

    def zero_grad(self) -> Optimiser:
        """
        Resets the optimiser's gradient

        :return: self
        """
        self.optimiser.zero_grad()
        return self

    def __str__(self) -> str:
        EOL = '\n'
        return f'{len(tuple(self.params))}\n' \
               f'{self.optimiser}\n' \
               f'{self.scheduler.__class__.__name__} ' \
               f'{EOL.join(f"{(k, v)}" for k, v in self.scheduler.state_dict().items())}\n' \
               f'{self.clip_max}'


EMPTY_OPTIMISER = Optimiser((), None, None)


@inference_mode()
def compute_loss(model: Module, dataset: Dataset, loss_fn: Fn[[Tensor, Tensor], Tensor] | str) -> Tensor:
    """
    Convenience function for evaluating a model.

    :param model: the model to test
    :param dataset: the test dataset
    :param loss_fn: the loss function to compute
    :return: the test loss
    """
    bs = -1 if isinstance(dataset, RaggedSupeDataset) else min(len(dataset), MAX_BATCH_SIZE)
    loss_fn = NAME_TO_LOSS_FN[loss_fn.replace('-', '_')] if isinstance(loss_fn, str) else loss_fn

    _, test_loss = epoch_step(AdapterModel(model).train(False),
                              make_loader(dataset, bs=bs, shuffled=False),
                              EMPTY_OPTIMISER,
                              loss_fn)

    return test_loss


@inference_mode()
def evaluate(model: AdapterModel, e_data: Dataset, bs: int, loss_fn: str) -> Tensor | None:
    """
     Convenience function for evaluating an adapted model.

    :param model: model to be evaluated
    :param e_data: test dataset
    :param bs: batch size
    :param loss_fn: loss function
    :return: loss
    """
    e_loss = None
    if e_data != EMPTY_DATASET:
        e_loader = make_loader(e_data, len(e_data) if bs > 0 else bs, shuffled=False)
        _, e_loss = epoch_step(model.train(False), e_loader, None, NAME_TO_LOSS_FN[loss_fn.replace('-', '_')])
    return e_loss


class WrapperModel(AdapterModel, Generic[T]):
    """
    Decorator for models to provide a convenient raw input and output interface instead of Module's native tensor one.
    """

    def __init__(self, delegate: Module, xcoder: Coder[T], ycoder: Coder[T] | None = None):
        super().__init__(delegate)
        self.xcoder = xcoder
        self.ycoder = ycoder

    @property
    def wrapped(self) -> Module:
        return super().adapted

    @AdapterModel.adapted.getter
    def adapted(self) -> Module:
        return super().adapted.adapter if hasattr(super().adapted, 'adapted') else super().adapted

    def forward(self, x: T, *args, **kwargs) -> Tensor | T:

        if not x or not isinstance(x[0], Sequence) or isinstance(x[0], str):  # TODO ADD SUPPORT FOR BATCH SIZE > 1

            tens = empty(0, self.xcoder.n) if not x else self.xcoder.tensorise(x)
            out = self.delegate(tens.unsqueeze(0), *args, **kwargs).squeeze(0)

        else:
            raise NotImplementedError()

        if self.ycoder is not None:
            if not x:
                return ()
            out = self.ycoder.untensorise(out)

        return out


def epoch_step(model: M, batches: Iterable[Batch], optim: Optimiser | None, loss_fn: Fn[[Tensor, Tensor], Tensor]) \
        -> Tuple[M, Tensor]:
    """
    Runs batches through model, doing a forward pass, and optionally - if the model is in training mode - doing a
    backward pass and updating the optimiser. It computes the average loss over the batch examples.

    This method is meant to be used as the inner loop of both training and validation outer loops

    :param model: the model to step through
    :param batches: collection of input-target 2-tuples, each a 2nd/3rd-order tensor BxD/BxTxD
    :param optim: optimiser to update model weights
    :param loss_fn: a loss function; must do a sum-reduction so that it can be summed over batches then averaged
    :return: possibly updated model and the example-averaged loss
    """
    total_loss = tensor(0., device=model.dev)
    n = 0
    with set_grad_enabled(model.training):

        for xs, ys in batches:
            xs, ys = xs.to(model.dev), ys.to(model.dev)

            y_hats = model(xs)
            losses = loss_fn(y_hats, ys, reduction='sum')
            if model.training:
                losses.backward()
                optim.step().zero_grad()  # scheduler update will be wrong if it needs to be per-epoch

            total_loss += losses
            n += (ys.shape[0] * (ys.shape[1] if ys.ndim == 3 else 1))  # assumes BxK or BxTxK shapes

        return model, total_loss / n


@inference_mode()
def _loss_tracker(epoch: int, train_loss: float, val_loss: float):
    """
    Prints epoch, training loss and validation loss

    :param epoch: the epoch to track
    :param train_loss: the training loss to track
    :param val_loss: the validation loss to track
    :return: nothing but prints to console
    """

    LOG.info(f'[{epoch:5,d}] [{train_loss:8.2e}] [{val_loss:8.2e}]')
