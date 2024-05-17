"""
Module-based models used in the AIS paper
"""
from collections import OrderedDict
from functools import reduce
from itertools import starmap
from math import sqrt, inf as INF, pi as PI
from typing import Sequence, Iterator, Literal

from torch import Tensor, zeros, ones, long, arange, einsum, stack, empty, eye
from torch.nn import Parameter, Module, Linear, Dropout, Embedding, ModuleList, ModuleDict, LSTM, Softmax, RNNBase, \
    Sequential, Conv1d, AdaptiveMaxPool1d, AdaptiveAvgPool1d
from torch.nn.functional import layer_norm
from torch.types import Device

from ais import check, Fn, px

MIN_PROB = 1e-12


class WFSA(Module):
    """
    Linear representation of a Weighted Finite State Automata. It can be trained with SGD or constructed
    with parameter tensors estimated with spectral learning. The decoder is an abstraction to allow multiple output
    types. See `TwoClassDecoder` and `LMDecoder`.
    """

    def __init__(self, init: Tensor, transits: Tensor, decoder: Module):
        super().__init__()
        self.init = Parameter(init.unsqueeze(-1))  # D -> Dx1
        self.transits = Parameter(transits)  # DxDxV
        self.decoder = decoder

    @property
    def initial(self) -> Tensor:
        return self.init.data.squeeze(-1)  # Dx1 -> D

    @property
    def transitions(self) -> Tensor:
        return self.transits.data.permute(-1, -2, -3)  # DxDxV -> VxDxD

    @property
    def final(self) -> Tensor:
        return self.decoder.weight if hasattr(self.decoder, 'weight') else empty(0)

    def forward(self, xs: Tensor) -> Tensor:
        check(xs, Tensor, lambda: xs.ndim == 3)

        state = self.init.expand((-1, xs.shape[0]))  # expands init vector to match batch size: Dx1 -> DxB

        for x in xs.permute(1, 2, 0):  # BxTxV -> TxVxB -> T:VxB... puts time dimension first to ease iteration
            trans = einsum('ikv,vb->bik', self.transits, x)  # DxDxV @ VxB -> DxDxB -> BxDxD
            state = einsum('bik,kb->ib', trans, state)  # BxDxD @ DxB -> BxDxB -> DxB

        return self.decoder(state.transpose(1, 0))  # DxB -> BxD (-> B(xK))


# ----------------------------------------- RNNS -----------------------------------------------------------------------

class OLSRNN(Module):
    """
    1-layer classifier Elman RNN with no embedding layer but with a decoder. The decoder is an abstraction
    to allow multiple output types.
    """

    def __init__(self, init: Tensor, rnn: RNNBase, decoder: Module, reset: bool = False):
        super().__init__()

        self.init = Parameter(init.unsqueeze(0))  # D -> 1xD
        self.rnn = rnn  # DxV, DxD
        self.decoder = decoder  # KxD
        self._reset = reset

    @property
    def reset(self) -> bool:
        return self._reset

    def forward(self, xs: Tensor) -> Tensor:
        init = self.init.expand((1, xs.shape[0], -1)) * (0 if self.reset and self.training else 1)  # 1xD -> 1xBxD

        state = init  # Bx0xV for the empty string else:
        if xs.shape[-2]:
            _, state = self.rnn(xs, init)  # (BxTxV, 1xBxD) -> (BxTxD, 1xBxD)

        return self.decoder(state.squeeze(0))  # BxTxD @ [KxD -> DxK] -> BxTxK | [BxTxD -> BxD] @ [KxD -> DxK] -> BxK


class OLGRU(OLSRNN):
    """
    1-layer classifier GRU RNN with no embedding layer but with a decoder. Constructor parameters as for
    OLSRNN.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OLLSTM(Module):
    """
    1-layer classifier LSTM with no embedding layer but with a decoder. The decoder is an abstraction
    to allow multiple output types.
    """

    def __init__(self, init: Tensor, start: Tensor, lstm: LSTM, decoder: Module, reset: bool = False):
        super().__init__()

        self.init = Parameter(init.unsqueeze(0))  # D -> 1xD
        self.start = Parameter(start.unsqueeze(0))  # D -> 1xD
        self.rnn = lstm
        self.decoder = decoder  # KxD
        self._reset = reset

    @property
    def reset(self) -> bool:
        return self._reset

    def forward(self, xs: Tensor) -> Tensor:
        init = self.init.expand((1, xs.shape[0], -1)) * (0 if self.reset and self.training else 1)  # 1xD -> 1xBxD
        start = self.start.expand((1, xs.shape[0], -1)) * (0 if self.reset and self.training else 1)  # 1xD -> 1xBxD

        state = init  # Bx0xV for the empty string
        if xs.shape[-2]:
            __, (state, _) = self.rnn(xs, (init, start))  # (BxTxD, (1xBxD, 1xBxD)) -> (BxTxD, (1xBxD, 1xBxD))

        return self.decoder(state.squeeze(0))  # [1xBxD -> BxD] @ [KxD -> DxK] -> BxK


# ------------------------------------------- FFWS ---------------------------------------------------------------------

class OLCNN(Sequential):
    """
    1-layer classifier CNN with no embedding layer but with a softmax layer.
    """

    def __init__(self, conv: Conv1d, pool: AdaptiveAvgPool1d | AdaptiveMaxPool1d, logits: Linear, y: Module):
        super().__init__(OrderedDict(conv=conv, pool=pool, logits=logits, y=y))


class OLMLP(Sequential):
    """
    1-layer classifier MLP with no embedding layer but with a decoder. The decoder is an abstraction
    to allow multiple output types.
    """

    def __init__(self, embed: Module, nonlin: Module, decoder: Module):
        super().__init__(OrderedDict(embed=embed, nonlin=nonlin, decoder=decoder))


# ------------------------------------------ TRANSFORMER ---------------------------------------------------------------


class LayerNorm(Module):
    """Transformer LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, dim: int, bias: bool):
        super().__init__()
        self._weight = Parameter(ones(dim))  # D
        self._bias = Parameter(zeros(dim)) if bias else None  # D

    @property
    def weight(self) -> Tensor:
        return self._weight

    @property
    def bias(self) -> Tensor:
        return self._bias

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.weight.shape, weight=self.weight, bias=self.bias, eps=1e-5)  # BxTxD -> BxTxD


class NoOpLayerNorm(LayerNorm):
    """ No-op Transformer LayerNorm """

    def __init__(self, ):
        super().__init__(0, False)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return iter(())

    def forward(self, x: Tensor) -> Tensor:
        return x


class PE(Embedding):
    """ GPT-style learned Transformer Position Embedding """

    def __init__(self, emb: Embedding):
        super().__init__(1, 1)
        self.emb = emb

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.emb.parameters(recurse)

    @property
    def weight(self) -> Tensor:
        return self.emb.weight

    def forward(self, idx: Tensor) -> Tensor:
        pos = arange(idx.shape[-1], dtype=long, device=idx.device).unsqueeze(0)  # BxT -> T -> 1xT
        return self.emb(pos)  # 1xT[xL] @ LxD -> 1xTxD


class SelfAttention(Module):
    """ Possibly masked multi-headed self-attention Transformer layer """

    def __init__(self, projs: Linear, att_drop: Dropout, out: Linear, res_drop: Dropout, mask: Tensor, num_heads: int):
        super().__init__()
        self.projs = projs  # H3GxD, key, query, value projections for all heads
        self.register_buffer('mask', mask)  # 1x1xLxL or 0
        self.softmax = Softmax(dim=-1)
        self.att_drop = att_drop
        self.out = out  # DxD
        self.res_drop = res_drop
        self.H = num_heads  # H
        self.G = self.projs.weight.shape[0] // 3 // self.H  # H3G -> HG -> G

    def forward(self, x: Tensor) -> Tensor:
        if self.H == 1:
            return self._one_head_forward(x)

        B, T, D = x.shape  # BxTxD

        q, k, v = self.projs(x).chunk(3, dim=-1)  # BxTxD @ [H3GxD -> DxH3G] -> BxTxH3G -> BxTxHG * 3
        q, k, v = (m.view(B, T, self.H, self.G).transpose(-2, -3) for m in [q, k, v])  # BxTxHG -> BxTxHxG -> BxHxTxG

        att = (q @ k.transpose(-1, -2)) * (1 / sqrt(self.G))  # BxHxTxG @ [BxHxTxG -> BxHxGxT] -> BxHxTxT'
        # v BxHxTxT' ~ 1x1xTxT -> BxHxTxT'
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, -INF) if self.mask.numel() else att
        att = self.att_drop(self.softmax(att))  # BxHxTxT' -> BxHxTxT' -> BxHxTxT'

        y = att @ v  # BxHxTxT' @ BxHxTxG -> BxHxTxG
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # BxHxTxG -> BxTxHxG -> BxTxD

        return self.res_drop(self.out(y))  # BxTxD @ DxD -> BxTxD -> BxTxD

    def _one_head_forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape  # BxTxD

        q, k, v = self.projs(x).chunk(3, dim=-1)  # BxTx3D -> BxTxD * 3

        att = (q @ k.transpose(-1, -2)) * (1 / sqrt(self.G))  # BxTxD @ [BxTxD -> BxDxT] -> BxTxT'
        if self.mask.numel():
            att = att.masked_fill(self.mask[0, :, :T, :T] == 0, -INF)  # BxTxT' ~ 1xTxT' -> BxTxT'
        att = self.att_drop(self.softmax(att))  # BxTxT' -> BxTxT' -> BxTxT'

        y = att @ v  # BxTxT' @ BxTxD -> BxTxD

        return self.res_drop(self.out(y))  # BxTxD @ DxD -> BxTxD -> BxTxD


class FFW(Sequential):
    """
    Transformer Feed-forward Layer
    """

    def __init__(self, expansion: Linear, squash: Module, contraction: Linear, drop: Dropout):
        # BxTxD @ DxF -> BxTxF -> BxTxF @ FxD -> BxTxD -> BxTxD
        super().__init__(OrderedDict(expansion=expansion, squash=squash, contraction=contraction, drop=drop))


class Block(Module):
    """ Transformer block """

    def __init__(self, ln_1: LayerNorm, att: SelfAttention, ln_2: LayerNorm, ffw: FFW):
        super().__init__()
        self.ln_1 = ln_1
        self.att = att
        self.ln_2 = ln_2
        self.ffw = ffw

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.att(self.ln_1(x))  # BxTxD -> BxTxD  + [BxTxD  -> BxTxD ~ att -> BxTxD] -> BxTxD'
        x = x + self.ffw(self.ln_2(x))  # BxTxD -> BxTxD' + [BxTxD' -> BxTxD ~ ffw -> BxTxD] -> BxTxD
        return x


class ChiangPositionEmbedding(Embedding):
    def __init__(self, num_tokens: int = 3, hid_dim: int = 10):
        super().__init__(0, 0)
        self.pad1 = num_tokens  # 0, 1, CLS
        self.pad2 = hid_dim - num_tokens - 2  # 2 -> [i/n, cos(i*PI)]

    def forward(self, x: Tensor) -> Tensor:  # TODO ADD SUPPORT FOR BATCH SIZE > 1
        n = x.shape[-1]
        zeroes = zeros(n).to(x.device)
        pos = arange(n).to(x.device)
        # BxT[xV=3] -> BxTxD=10
        return stack([zeroes] * self.pad1 + [pos / n, (pos * PI).cos()] + [zeroes] * self.pad2, dim=-1)


class ChiangTokenEmbedding(Embedding):
    def __init__(self, num_tokens: int = 3, hid_dim: int = 10):
        super().__init__(0, 0)
        self.lookup = eye(num_tokens, hid_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.lookup[x.fliplr()]  # BxT[xV=3] -> BxTxD=10


class GPT(Module):
    """
    A general single-stack Transformer

    B -> batch
    T -> sequence length
    L -> max sequence length
    D -> embedding dimension
    H -> number of heads

    @ -> matrix multiplication
    + -> matrix addition

    Based off https://github.com/karpathy/nanoGPT/blob/master/model.py
    """

    def __init__(self,
                 wte: Embedding,
                 wpe: Embedding,
                 drop: Dropout,
                 blocks: Sequence[Block],
                 ln: LayerNorm,
                 head: Module,
                 tying: bool = False):
        super().__init__()

        # wte VxD | wpe TxD
        self.body = ModuleDict(dict(wte=wte, wpe=wpe, drop=drop, blocks=ModuleList(blocks), ln=ln))
        self.head = head
        self.tying = tying
        if self.tying:  # only applies to seq2seq scenario where input vocab is the same as output vocab (LM)
            self.body.wte.weight = self.head.weight  # https://paperswithcode.com/method/weight-tying

    def forward(self, idxs: Tensor) -> Tensor:
        tok_emb = self.body.wte(idxs)  # BxT[xV] @ VxD -> BxTxD

        pos_emb = self.body.wpe(idxs)  # BxT[xV] (-> 1xT[xL] @ TxD) -> 1xTxD

        x = self.body.drop(tok_emb + pos_emb)  # [BxTxD + 1xTxD] -> BxTxD -> BxTxD

        x = reduce(lambda x, block: block(x), self.body.blocks, x)  # BxTxD -> BxTxD...

        return self.head(self.body.ln(x))  # [BxTxD -> BxTxD] -> BxTxD @ (DxK -> BxTxK | -> BxD @ DxK -> BxK)


# ----------------------------------------------- GENERIC --------------------------------------------------------------


class AdapterModel(Module):
    """
    Provides a convenient enhanced interface to a Module-based model
    """

    def __init__(self, delegate: Module):
        super().__init__()
        self.delegate = delegate

    @property
    def dev(self) -> Device:
        return next(self.delegate.parameters()).device  # this might not work with model sharding

    @property
    def decoder(self) -> Module:
        return (self.delegate.decoder if hasattr(self.delegate, 'decoder') else
                self.delegate.head if hasattr(self.delegate, 'head') else
                self.delegate)

    @property
    def adapted(self) -> Module:
        return self.delegate

    @property
    def weight(self) -> Tensor:
        return self.delegate.weight if hasattr(self.delegate, 'weight') else empty(0)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.delegate(*args, **kwargs)

    def __str__(self) -> str:
        return str(self.delegate)


class FuncModule(Module):
    """
    Generic class supporting inference simple enough that can fit in a function
    """

    def __init__(self, fn: Fn, name: str = '', unpack: bool = False):
        """
        :param fn: The function to call within the forward() method
        :param name: the name of the module, acts as a pseudo-type
        :param unpack: whether to unpack the positional arguments passed to the forward() method before passing them on
            to `fn`
        """
        super().__init__()
        self.fn = fn
        self._name = name or self.fn.__class__.__name__
        self.unpack = unpack

    @property
    def name(self) -> str:
        return self._name

    def forward(self, *args, **kwargs) -> Tensor:
        if self.unpack:
            return tuple(starmap(self.fn, args))[0]

        return self.fn(*args, **kwargs)


class IdentitySoftmax(Softmax):
    """
    Abstraction over a softmax module that returns the input as it is (expected to be logits) when training, assuming
    a CrossEntropyLoss loss function; and that returns a softmax when not in training or when set to do so
    """

    def __init__(self, dim: int | None = -1, normalise: bool = False):
        super().__init__(dim)
        self.normalise = normalise

    def forward(self, x: Tensor) -> Tensor:
        return x if (self.training or not self.normalise) else super().forward(x)


class TwoClassDecoder(Module):
    """
    A decoder module meant for classifier WFSAs, that takes its weight vector and turns it into a binary [0, 1] or
    polar [-1, 1] 2-dimensional output
    """

    def __init__(self, dec: Tensor, polar: bool = False):
        super().__init__()
        self.dec = Parameter(dec)  # D
        self.polar = polar

    @property
    def weight(self) -> Tensor:
        return self.dec.data

    def forward(self, x: Tensor) -> Tensor:
        p = x @ self.dec  # B(xT)xD @ D -> B(xT)
        out = stack([(-1 * p) if self.polar else (1 - p), p]).transpose(1, 0)  # B(xT) -> 2xB(xT) -> Bx2(xT)
        if out.ndim == 3:
            out = out.transpose(-1, -2)  # Bx2(xT) -> B(xT)x2

        return out


class LMDecoder(Module):
    """
    A decoder module meant for LM WFSAs, that takes its weight vector and turns it into a binary [0, 1]
    2-dimensional output, with 2nd dimension standing for the probability of the sequence being True, and the 1st
    dimension standing of the probability of being False
    """

    def __init__(self, dec: Tensor, logarithm: bool = False):
        super().__init__()
        self.dec = Parameter(dec)  # D
        self.logarithm = logarithm  # True if the output should be a log probability, to train with xent/KL

    @property
    def weight(self) -> Tensor:
        return self.dec.data

    def forward(self, x: Tensor) -> Tensor:
        p = (x @ self.dec).clip(0, 1)  # B(xT)xD @ D -> B(xT)
        out = stack([1 - p, p]).transpose(1, 0)  # B(xT) -> 2xB(xT) -> Bx2(xT)

        if self.logarithm:
            out[out == 0.] = MIN_PROB  # avoids -inf when log-ing
            out = out.log()

        return out


class Reducer(FuncModule):
    """
    A reduction/aggregation module that collapses a sequence of token embedding to a single embedding. Meant for
    sequence classification.
    """

    def __init__(self, reduction: Literal['sum', 'mean', 'prod', 'first', 'last']):
        check(val=reduction in {'sum', 'mean', 'prod', 'first', 'last'})
        super().__init__(px(_reduce, reduction=reduction), name=f'reducer-{reduction}')

    @property
    def kind(self) -> str:
        return self.name.split('-')[-1]


def _reduce(x: Tensor, reduction: str) -> Tensor:
    # assumes BxTxD

    return (x.sum(dim=-2) if reduction == 'sum' else
            x.mean(dim=-2) if reduction == 'mean' else
            x.prod(dim=-2) if reduction == 'prod' else
            x[:, -1, :] if reduction == 'last' else
            x[:, 0, :])
