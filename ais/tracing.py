"""
Classes and functions to trace embeddings through model layers
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from math import sqrt
from typing import Tuple, TypeVar

from torch import Tensor, empty, einsum, stack, cat, inference_mode
from torch.nn import Module, Sequential, Identity, Linear
from torch.utils.hooks import RemovableHandle

from ais import check, outm
from ais.models import GPT, OLMLP, OLCNN, OLLSTM, WFSA, OLGRU, OLSRNN, FuncModule

M = TypeVar('M', bound=Module)

REDUCED_TO_CUMULATIVE = dict(sum=FuncModule(lambda xs: xs.cumsum(dim=-2)),
                             mean=FuncModule(lambda xs: xs.cummean(dim=-2)),
                             prod=FuncModule(lambda xs: xs.cumprod(dim=-2)),
                             last=Identity(),
                             first=Identity())


class TracingModule(Module):
    """
    Decorates modules to capture the inputs and outputs, with a view to probing and visualising them.
    """

    def __init__(self, target: Module, name: str = '', dump: bool = False, capture: int | None = None):
        super().__init__()
        self._target = target
        self._name = name or target.__class__.__name__
        self._xs = ()
        self._ys = ()
        self._dump = dump
        self.capture = capture

    @property
    def target(self) -> Module:
        return self._target

    @property
    def name(self) -> str:
        return self._name

    @property
    def xs(self) -> Tuple[Tensor, ...]:
        return self._xs

    @property
    def ys(self) -> Tuple[Tensor, ...]:
        return self._ys[self.capture:self.capture + 1] if self.capture is not None else self._ys

    @property
    def dump(self) -> bool:
        return self._dump

    @dump.setter
    def dump(self, val: bool):
        self._dump = val

    @property  # TODO should this be in this class?
    def weight(self) -> Tensor:
        return (self._target.weight if hasattr(self._target, 'weight') else
                self._target.proj_in.weight if hasattr(self._target, 'proj_in') else
                empty(0))

    @property  # TODO should this be in this class?
    def H(self) -> int | None:
        return self._target.H if hasattr(self._target, 'H') else None

    def forward(self, x: Tensor | Tuple[Tensor, ...], *other: Tensor) -> Tensor | Tuple[Tensor, ...]:
        if self.dump:
            outm(x[0] if isinstance(x, Tuple) else x)
        _x = x if isinstance(x, tuple) else (x,)
        self._xs = tuple(_for_viewing(tens) for tens in _x + other)

        y = self._target(x, *other)
        _ys = y if isinstance(y, Tuple) else (y,)
        self._ys = tuple(_for_viewing(_y) for _y in _ys)

        return y


class PassThrough(Module):
    """
    Decorates modules to enable serialisation of data flow in models that have parallel channels, like transformers'
    residual streams. This makes tracing easier.
    """

    def __init__(self, delegate: Module, swap: bool = False):
        super().__init__()
        self.delegate = delegate
        self.swap = swap

    def register_forward_hook(self, *args, **kwargs) -> RemovableHandle:
        return self.delegate.register_forward_hook(*args, **kwargs)

    def register_forward_pre_hook(self, *args, **kwargs) -> RemovableHandle:
        return self.delegate.register_forward_pre_hook(*args, **kwargs)

    def forward(self, x: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, passthrough = x if isinstance(x, Tuple) else (x, x)
        return (passthrough, self.delegate(x)) if self.swap else (self.delegate(x), passthrough)


@inference_mode()
def as_traced_wfsa(wfsa: WFSA) -> Sequential:
    """
    Builds a traced serialised WFSA for ease of tracing.

    :param wfsa: original WFSA
    :return: traced WFSA
    """
    check(wfsa, WFSA)

    wfsa = deepcopy(wfsa)

    def _cumulative_hook(xs: Tensor) -> Tensor:
        # concatenates initial state with input-matched states to give full state sequence
        state = wfsa.init.expand((-1, xs.shape[0]))  # expands init vector to match batch size: Dx1 -> DxB
        states = [state]
        for x in xs.permute(1, 2, 0):  # BxTxV -> TxVxB -> T:VxB... puts time dimension first to ease iteration

            trans = einsum('ikv,vb->bik', wfsa.transits, x)  # DxDxV @ VxB -> DxDxB -> BxDxD
            state = einsum('bik,kb->ib', trans, state)  # BxDxD @ DxB -> BxDxB -> DxB
            states.append(state)  #

        return stack(states).permute(-1, -3, -2)  # T+1:DxB...-> T+1xDxB -> BxT+1xD

    return Sequential(OrderedDict(hidden=_trace(FuncModule(_cumulative_hook, 'hidden'), 'hidden'),
                                  logits=Identity(),  # dummy layer added as expected by viz code
                                  y=_trace(wfsa.decoder, 'y')))


@inference_mode()
def as_traced_rnn(rnn: OLSRNN | OLGRU | OLLSTM) -> Sequential:  # TODO UNLIKE FULL MODULE, IT CAN'T HANDLE EMPTY INPUTS
    """
    Builds a traced serialised RNN for ease of tracing.
    :param rnn: original RNN
    :return: traced RNN
    """
    check(rnn, OLSRNN | OLGRU | OLLSTM)

    rnn = deepcopy(rnn)

    traced = Sequential(OrderedDict(hidden=_trace(rnn.rnn, 'hidden'),
                                    logits=rnn.decoder.logits,
                                    y=_trace(rnn.decoder.y, 'y')))

    def _init_hook(_, x: Tuple[Tensor]) -> Tuple[Tensor, Tensor | Tuple[Tensor, Tensor]]:
        # ensures that the initial state (and cell) is passed to the rnn, along with the input, replicating the logic
        # in the untraced model
        x = x[0]
        init = rnn.init.expand((1, x.shape[0], -1))

        if isinstance(rnn, OLLSTM):
            return x, (init, rnn.start.expand((1, x.shape[0], -1)))

        return x, init

    traced.hidden.target.register_forward_pre_hook(_init_hook, prepend=True)

    def _cumulative_hook(_, x: Tuple[Tensor, Tensor | Tuple[Tensor, Tensor]], y: Tuple[Tensor, Tensor]) -> Tensor:
        # concatenates initial state with input-matched states to give full state sequence
        states, _ = y
        init = x[-1][0] if isinstance(rnn, OLLSTM) else x[-1]

        return cat([init.transpose(-2, -3), states], dim=-2)  # [1xBxD -> Bx1xD]|BxLxD -> BxL+1xD

    traced.hidden.target.register_forward_hook(_cumulative_hook, prepend=True)

    return traced


@inference_mode()
def as_traced_mlp(mlp: OLMLP, kind: str = 'rumelhart') -> Sequential:
    """
    Builds a traced serialised MLP for ease of tracing.

    :param mlp: original MLP
    :param kind: type of MLP, one of 'rumelhart' and 'learned'
    :return: traced MLP
    """
    check(mlp, OLMLP).check(val=kind in {'rumelhart', 'learned'})

    mlp = deepcopy(mlp)

    if kind == 'rumelhart':
        def _cumulative_hook(lin: Linear, x: Tensor, _) -> Tensor:
            return ((lin.weight.data * x[0].squeeze(0)).cumsum(dim=-1).T + lin.bias.data).unsqueeze(0)  # \

        mlp.embed.register_forward_hook(_cumulative_hook)

        return Sequential(OrderedDict(
            embed=mlp.embed,
            hidden=_trace(mlp.nonlin, 'counter'),
            logits=mlp.decoder.logits,
            y=_trace(mlp.decoder.y, 'y')))

    return Sequential(OrderedDict(
        embed=mlp.embed,
        hidden=mlp.nonlin,
        red=_trace(REDUCED_TO_CUMULATIVE[mlp.decoder.red.kind], 'counter'),
        logits=mlp.decoder.logits,
        y=_trace(mlp.decoder.y, 'y')))


@inference_mode()
def as_traced_cnn(cnn: OLCNN) -> Sequential:
    """
    Builds a traced serialised CNN for ease of tracing.
    :param cnn: original CNN
    :return: traced CNN
    """
    check(cnn, OLCNN)

    cnn = deepcopy(cnn)

    def _cumulative_hook(_, x: Tensor, __) -> Tensor:
        return x[0].cummax(dim=-1).values.transpose(-1, -2)  # 1xD-1xL -> 1xD-1xL -> 1xLxD-1

    cnn.pool.register_forward_hook(_cumulative_hook)

    return Sequential(OrderedDict(
        conv=cnn.conv,
        pool=_trace(cnn.pool, 'pool'),
        logits=cnn.logits,
        y=_trace(cnn.y, 'y')))


@inference_mode()
def as_traced_transformer(model: GPT, kind: str = 'learned') -> Sequential:
    """
    Builds a traced serialised encoder-Transformer for ease of tracing.

    :param model: original Transformer
    :param kind: type of Transformer, one of 'chiang', 'learned'
    :return: traced Transformer
    """
    check(model, GPT).check(val=kind in {'chiang', 'learned'}, msg=f'[{kind}]')

    return (_traced_chiang_transformer if kind == 'chiang' else _traced_learned_transformer)(deepcopy(model))


def _traced_chiang_transformer(model: GPT) -> Sequential:
    block, *other_blocks = model.body.blocks  # SUPPORTS ONLY 2-LAYER 2-HEAD CAUSAL-ATTENTION ENCODER-TRANSFORMERS
    H, G = block.att.H, block.att.G  # G = D//H
    norm, softmax = sqrt(G), block.att.softmax
    block_1, block_2 = block, other_blocks[0]
    proj_in_1 = block_1.att.projs
    proj_in_2 = block_2.att.projs

    qkv_split_1 = FuncModule(
        lambda x: tuple(m.reshape(*x.shape[:2], H, -1).transpose(-2, -3) for m in proj_in_1(x).chunk(3, dim=-1)),
        name='qkv-1')
    qkv_split_2 = FuncModule(
        lambda x: tuple(m.reshape(*x.shape[:2], H, -1).transpose(-2, -3) for m in proj_in_2(x).chunk(3, dim=-1)),
        name='qkv-2')
    att = FuncModule(lambda q, k, v: (softmax(q @ k.transpose(-1, -2) / norm), v), 'att', unpack=True)
    att_y = FuncModule(lambda att, v: (att @ v).transpose(-2, -3).contiguous().view(v.shape[0], v.shape[2], H * G),
                       'att-y', unpack=True)

    modules = Sequential(OrderedDict(
        we=PassThrough(model.body.wte, swap=True),  # .       x ->                    x, we
        pe=PassThrough(model.body.wpe),  # .                  x, we ->                pe, we
        embed=FuncModule(sum, 'embed'),  # .                  pe, we ->               embed

        att_ln_1=PassThrough(block_1.ln_1),  # .              embed ->                att_ln, embed
        att_qkv_1=PassThrough(qkv_split_1),  # .              att_ln, embed ->        att_qkv, embed
        att_1=PassThrough(att),  # .                          att_qkv, embed ->       [att,v], embed
        att_y_1=PassThrough(att_y),  # .                      [att,v], embed ->       att_y, embed
        att_out_1=PassThrough(block_1.att.out),  # .          att_y, embed ->         att_out, embed
        att_res_1=FuncModule(sum, 'att-res-1'),  # .          att_out, embed ->       att_res

        ffw_ln_1=PassThrough(block_1.ln_2),  # .              att_res ->              ffw_ln, att_res
        ffw_exp_1=PassThrough(block_1.ffw.expansion),  # .    ffw_ln, att_res ->      ffw_exp, att_res
        ffw_squash_1=PassThrough(block_1.ffw.squash),  # .    ffw_exp, att_res ->     ffw_squash, att_res
        ffw_con_1=PassThrough(block_1.ffw.contraction),  # .  ffw_squash, att_res ->  ffw_con, att_res
        ffw_res_1=FuncModule(sum, 'ffw-res-1'),  # .          ffw_con, att_res.ys ->  ffw_res

        att_ln_2=PassThrough(block_2.ln_1),  # .              embed ->                att_ln, embed
        att_qkv_2=PassThrough(qkv_split_2),  # .              att_ln, embed ->        att_qkv, embed
        att_2=PassThrough(deepcopy(att)),  # .                att_qkv, embed ->       [att,v], embed
        att_y_2=PassThrough(deepcopy(att_y)),  # .            [att,v], embed ->       att_y, embed
        att_out_2=PassThrough(block_2.att.out),  # .          att_y, embed ->         att_out, embed
        att_res_2=FuncModule(sum, 'att-res-2'),  # .          att_out, embed ->       att_res

        ffw_ln_2=PassThrough(block_2.ln_2),  # .              att_res ->              ffw_ln, att_res
        ffw_exp_2=PassThrough(block_2.ffw.expansion),  # .    ffw_ln, att_res ->      ffw_exp, att_res
        ffw_squash_2=PassThrough(block_2.ffw.squash),  # .    ffw_exp, att_res ->     ffw_squash, att_res
        ffw_con_2=PassThrough(block_2.ffw.contraction),  # .  ffw_squash, att_res ->  ffw_con, att_res
        ffw_res_2=FuncModule(sum, 'ffw-res-2'),  # .          ffw_con, att_res.ys ->  ffw_res

        ln=model.body.ln,  # .                                ffw_res ->              ln
        reducer=model.head.red,  # .                          ln ->                   red
        logits=model.head.logits,  # .                        red ->                  logits
        y=model.head.y))  # .                                 logits ->               y

    def _att_context_unpacking_hook(_, x: Tuple[Tuple[Tensor, Tensor]], __) -> Tensor:
        """
        maps attention (batch >>> head >> token >> weights) 1x2xTxT and
        value (batch >> head >> token >> dim) 1x2xTx5 matrices into a
        context (batch >> token >> dim) TxTx10 matrix
        """
        att, v = x[0][0].squeeze(0), x[0][1].squeeze(0)  # [1x2xTxT' -> 2xTxT'],  [1x2xTx5 -> 2xTx5]
        # 2xTxT' o' 2xTx5 -> 2xTxT'x5
        hadamard = einsum('htw, hwd -> htwd', att, v)  # head >> token >> weights >> dims
        # similarly to the learned transformer, the attention @ value operation is unpacked into a broadcast hadamard
        # product followed by a cumulative sum. The two differences are: the flipping of the weights axis to account for
        # the chiang transformer picking the last token as the output instead of the first one (what BERT does);
        # and the reshaping to concatenate the two heads, which the learned transformers don't need because they only
        # have on; after these operations, each first row, for every sequence, represents the number of 1s
        # 2xTxT'x5 -> 2xTxT'"x5 -> 2xTxT'"`x5 -> T'"`xTx2x5 -> T'"`xTx10
        return hadamard.flip(dims=[-2]).cumsum(dim=-2).transpose(-2, -4).reshape(*hadamard.shape[1:-1], -1)

    def _att_res_unpacking_hook(_, x: Tuple[Tensor, Tensor], __) -> Tensor:
        att_out, embed = x[0]
        return att_out + embed.expand(embed.shape[-2], *embed.shape[1:])  # TxT'x10 + [1xTx10 -> TxT'x10] -> TxT'x10

    modules.att_y_1.register_forward_hook(_att_context_unpacking_hook)
    modules.att_res_1.register_forward_hook(_att_res_unpacking_hook)

    return Sequential(OrderedDict(att_y=modules[:7],
                                  fork=PassThrough(PassThrough(_trace(FuncModule(lambda att_con: att_con[:, 0, :])))),
                                  unfork=FuncModule(lambda _att_y, embed: (_att_y[-1], embed), unpack=True),
                                  logits=modules[7: -1],
                                  y=_trace(modules[-1])))


def _traced_learned_transformer(model: GPT) -> Sequential:
    # SUPPORTS ONLY 1-LAYER 1-HEAD CAUSAL-ATTENTION ENCODER-TRANSFORMERS
    block, *_ = model.body.blocks
    proj_in = block.att.projs
    norm, softmax = sqrt(block.att.G), block.att.softmax

    qkv_split = FuncModule(lambda x: proj_in(x).chunk(3, dim=-1), 'qkv')
    att = FuncModule(lambda q, k, v: (softmax(q @ k.transpose(-1, -2) / norm), v), 'att', unpack=True)
    att_y = FuncModule(Tensor.matmul, 'att-y', unpack=True)

    modules = Sequential(OrderedDict(
        we=PassThrough(model.body.wte, swap=True),  # .     x ->                    x, we
        pe=PassThrough(model.body.wpe),  # .                x, we ->                pe, we
        embed=FuncModule(sum, 'embed'),  # .                pe, we ->               embed

        att_ln=PassThrough(block.ln_1),  # .                embed ->                att_ln, embed
        att_qkv=PassThrough(qkv_split),  # .                att_ln, embed ->        att_qkv, embed
        att=PassThrough(att),  # .                          att_qkv, embed ->       [att,v], embed
        att_y=PassThrough(att_y),  # .                      [att,v], embed ->       att_y, embed
        att_out=PassThrough(block.att.out),  # .            att_y, embed ->         att_out, embed
        att_res=FuncModule(sum, 'att-res'),  # .            att_out, embed ->       att_res

        ffw_ln=PassThrough(block.ln_2),  # .                att_res ->              ffw_ln, att_res
        ffw_exp=PassThrough(block.ffw.expansion),  # .      ffw_ln, att_res ->      ffw_exp, att_res
        ffw_squash=PassThrough(block.ffw.squash),  # .      ffw_exp, att_res ->     ffw_squash, att_res
        ffw_con=PassThrough(block.ffw.contraction),  # .    ffw_squash, att_res ->  ffw_con, att_res
        ffw_res=FuncModule(sum, 'ffw-res'),  # .            ffw_con, att_res.ys ->  ffw_res

        ln=model.body.ln,  # .                              ffw_res ->              ln
        reducer=model.head.red,  # .                        ln ->                   red
        logits=model.head.logits,  # .                      red ->                  logits
        y=model.head.y))  # .                               logits ->               y

    def _att_context_unpacking_hook(_, x: Tuple[Tensor, Tensor], __) -> Tensor:
        """
        1. sets attention weights that are not counting 1s for OR to emphasise the counting
        2. accumulates attention weighted dimensions for each token
        3. makes the first/CLS token cumulative weights into the 1st axis (for the benefit of the reducer downstream);

        After these operations, each first row, for every sequence, represents the number of 1s (OR) or 0s (AND)
        """
        att, v = x[0][0].squeeze(0), x[0][1].squeeze(0)  # [1xTxT' -> TxT'], [1xTxD -> TxD]
        return einsum('tw,wd -> twd', att, v).cumsum(dim=-2).transpose(-2, -3)  # TxT' o' TxD -> TxT'xD -> TxT'`xD -> T'`xTxD

    def _att_res_unpacking_hook(_, x: Tuple[Tensor, Tensor], __) -> Tensor:
        """
        1. reshapes embeddings to match the shape of attention out-projection
        2. scales attention out-projection and embeddings to make up for the operations in attention context
        """
        att_out, embed = x[0]
        embed = embed.expand(embed.shape[-2], *embed.shape[1:])  # 1xTxD -> TxT'xD ; for no-residual, set to 0
        return att_out * 100 - embed * 1.1 + embed  # for no-residual, replace 100 with 30 for better visuals

    modules.att_y.register_forward_hook(_att_context_unpacking_hook)
    modules.att_res.register_forward_hook(_att_res_unpacking_hook)

    return Sequential(OrderedDict(att_y=modules[:7],  # TxT'xD
                                  fork=PassThrough(PassThrough(_trace(FuncModule(lambda att_y: att_y[:, 0, :])))),
                                  unfork=FuncModule(lambda _att_y, embed: (_att_y[-1], embed), unpack=True),
                                  logits=modules[7: -1],
                                  y=_trace(modules[-1])))


# ----------------------------------------------------------------------------------------------------------------------

def _trace(target: Module, name: str = '', dump: bool = False, capture: int | None = None) -> TracingModule:
    return TracingModule(target, name, dump, capture)


def _for_viewing(tens: Tensor | Tuple[Tensor, Tensor]) -> Tensor:
    """ transforms tensor for viewing """
    if isinstance(tens, Tuple):
        tens = tens[0]
    clone = tens.detach().clone()

    if tens.ndim >= 3:  # assumes Bx...
        clone = clone.squeeze(0).transpose(-1, -2)
    elif tens.ndim == 2:  # assumes BxD
        clone = clone.squeeze(0)
        if clone.ndim == 2:
            clone = clone.transpose(-1, -2)
    else:
        raise Exception(f'dimension [{clone.ndim}] is not allowed')

    return clone


def get_levels(model: Module) -> Tuple[Tensor, ...]:  # DxL..
    """ aggregates the output of traced submodule in the given module """
    return tuple(module.ys[0] for module in model.modules() if isinstance(module, TracingModule))
