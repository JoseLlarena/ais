"""
Function and constants for creating Module-based models used in the AIS paper
"""
from __future__ import annotations

from collections import OrderedDict
from functools import partial
from itertools import product
from numbers import Number
from typing import TypeVar, Literal

from torch import zeros, relu, tanh, tril, ones, empty, cat, Tensor, tensor, heaviside, inference_mode
from torch.nn import Module, RNN, Embedding, Linear, Dropout, ReLU, GELU, Sequential, Conv1d, AdaptiveAvgPool1d, \
    AdaptiveMaxPool1d, GRU, LSTM, Identity, Tanh, ModuleDict, ModuleList, Softmax
from torch.nn.functional import gelu, pad
from torch.serialization import add_safe_globals

from ais import Fn, check, px
from ais.models import LayerNorm, Block, GPT, SelfAttention, FFW, OLMLP, ChiangTokenEmbedding, ChiangPositionEmbedding, \
    PE, NoOpLayerNorm, FuncModule, Reducer, OLCNN, OLLSTM, WFSA, TwoClassDecoder, IdentitySoftmax, OLGRU, OLSRNN, \
    LMDecoder, _reduce

M = TypeVar('M', bound=Module)

APPROX_GELU = px(gelu, approximate='tanh')
TRANSFORMER_MODES = {'-'.join(['class'] + list(parts)).replace('- ', '') for parts in
                     product(['sum', 'mean', 'prod', 'first', 'last', ' '], ['lin', 'xent'])} \
                    | {'seq2seq', 'seq2seq-xent'}
MLP_MODES = {'-'.join(parts).replace('- ', '') for parts in
             product(['sum', 'mean', 'prod', 'first'], ['lin', 'xent'])}
WFSA_MODES = {'binary', 'polar', 'lm', 'log-lm'}


def wfsa_from(initial: Tensor,
              transits: Tensor,
              final: Tensor,
              mode: Literal['binary', 'polar', 'lm', 'log-lm'] = 'binary') -> WFSA:
    """
    Builds a Weighted Finite State Automaton in its linear representation. The `mode` parameter determines the type of
    output:
        `binary`: 2-dimensional one-hot vector with 1st component standing for False and 2nd for True
        `polar`:  2-dimensional vector with values in [-1,1]; 1st component standing for False and 2nd for True
        'lm`:     1-dimensional vector in [0, 1] representing the input sequence's probability
        'log-lm`: 1-dimensional vector in [-inf, 0] representing the input sequence's log-probability

   'lm' is meant for training with regression losses (MSE, MAE,...) and 'log-lm' for probabilistic losses
    (cross-entropy, kullback-leibler divergence,...)

    V: vocabulary size, D: weight vector dimensionality/ number of states

    :param initial: initial weight vector of size D
    :param transits: tensor of transition matrices of size DxDxV, 1st axis for to-states, 2nd axis for from-states
    :param final: final weight vector of size D
    :param mode: the type of output
    :return: a WFSA as a Pytorch Module
    """
    check(initial, Tensor, lambda: initial.ndim == 1 and initial.numel(), msg=lambda: f'{initial.shape}')
    check(transits, Tensor,
          lambda: transits.ndim == 3 and transits.shape[-3] == transits.shape[-2] and transits.numel(),
          lambda: f'{transits.shape}')
    check(final, Tensor, lambda: final.ndim == 1 and final.numel(), msg=lambda: f'{final.shape}')
    check(val=mode in WFSA_MODES, msg=f'{mode}')

    decoder = LMDecoder(final, logarithm=mode == 'log-lm') if mode.endswith('lm') else \
        TwoClassDecoder(final, polar=mode == 'polar')
    return WFSA(init=initial, transits=transits, decoder=decoder)


def make_wfsa(v: int, hid_dim: int, mode: Literal['binary', 'polar', 'lm', 'log-lm']) -> WFSA:
    """
    Builds a Weighted Finite State Automaton in its linear representation. The `mode` parameter determines the type of
    output:
        `binary`: 2-dimensional one-hot vector with 1st component standing for False and 2nd for True
        `polar`:  2-dimensional vector with values in [-1,1]; 1st component standing for False and 2nd for True
        'lm`:     1-dimensional vector in [0, 1] representing the input sequence's probability
        'log-lm`: 1-dimensional vector in [-inf, 0] representing the input sequence's log-probability

    :param v: input vocabulary size
    :param hid_dim: size of hidden dimension/weight vector/ number of states
    :param mode: the type of output
    :return: a WFSA as a Pytorch Module
    """
    check(v, int, lambda: v > 0).check(hid_dim, int, lambda: hid_dim > 0)

    return wfsa_from(initial=zeros(hid_dim), transits=zeros(hid_dim, hid_dim, v), final=zeros(hid_dim), mode=mode)


def make_olsrnn(v: int,
                hid_dim: int,
                out_dim: int,
                nonlin: Fn = relu,
                reset: bool = False,
                bias: bool = False,
                mode: Literal['class', 'class-xent'] = 'class') -> OLSRNN:
    """
    Builds a 1-layer Simple/Elman Recurrent Neural Network with a decoder layer, meant for classification.

    :param v: input vocabulary size
    :param hid_dim: size of hidden dimension
    :param out_dim: size of output dimension
    :param bias: whether a bias should be added to the linear transforms
    :param nonlin: the type of linearity in the RNN's hidden layer
    :param reset: whether the initial state should be reset during training
    :param mode: the type of output: `class` for linear, `class-xent` for linear+softmax
    :return: a simple RNN classifier
    """
    check(v, int, lambda: v > 0).check(hid_dim, int, lambda: hid_dim > 0).check(out_dim, int, lambda: out_dim > 0)
    check(nonlin, type(relu) | type(tanh) | type(ReLU) | type(Tanh)).check(val=mode in {'class', 'class-xent'})

    state_0 = zeros(hid_dim)  # D
    rnn = RNN(v, hid_dim, nonlinearity=nonlin.__name__.lower(), bias=bias, batch_first=True)
    decoder = Sequential(OrderedDict(logits=Linear(hid_dim, out_dim, bias),
                                     y=IdentitySoftmax(normalise=mode == 'class-xent')))

    return OLSRNN(state_0, rnn, decoder=decoder, reset=reset)


def make_olgru(v: int,
               hid_dim: int,
               out_dim: int,
               reset: bool = False,
               bias: bool = False,
               mode: Literal['class', 'class-xent'] = 'class') -> OLGRU:
    """
    Builds a 1-layer Gated Recurrent Unit with a decoder layer, meant for classification.

    :param v: input vocabulary size
    :param hid_dim: size of hidden dimension
    :param out_dim: size of output dimension
    :param bias: whether a bias should be added to the linear transforms
    :param reset: whether the initial state should be reset during training
    :param mode: the type of output: `class` for linear, `class-xent` for linear+softmax
    :return: a GRU classifier
    """
    check(v, int, lambda: v > 0).check(hid_dim, int, lambda: hid_dim > 0).check(out_dim, int, lambda: out_dim > 0)
    check(val=mode in {'class', 'class-xent'})

    state_0 = zeros(hid_dim)  # D
    rnn = GRU(v, hid_dim, bias=bias, batch_first=True)
    decoder = Sequential(OrderedDict(logits=Linear(hid_dim, out_dim, bias),
                                     y=IdentitySoftmax(normalise=mode == 'class-xent')))

    return OLGRU(state_0, rnn, decoder=decoder, reset=reset)


def make_ollstm(v: int,
                hid_dim: int,
                out_dim: int,
                reset: bool = False,
                bias: bool = False,
                mode: Literal['class', 'class-xent'] = 'class') \
        -> OLLSTM:
    """
    Builds a 1-layer Long Short-Term Memory RNN with a decoder layer, meant for classification.

    :param v: input vocabulary size
    :param hid_dim: size of hidden dimension
    :param out_dim: size of output dimension
    :param bias: whether a bias should be added to the linear transforms
    :param reset: whether the initial state should be reset during training
    :param mode: the type of output: `class` for linear, `class-xent` for linear+softmax
    :return: an LSTM classifier
    """
    check(v, int, lambda: v > 0).check(hid_dim, int, lambda: hid_dim > 0).check(out_dim, int, lambda: out_dim > 0)
    check(val=mode in {'class', 'class-xent'})

    state_0 = zeros(hid_dim)  # D
    mem_0 = zeros(hid_dim)  # D
    rnn = LSTM(v, hid_dim, bias=bias, batch_first=True)
    decoder = Sequential(OrderedDict(logits=Linear(hid_dim, out_dim, bias),
                                     y=IdentitySoftmax(normalise=mode == 'class-xent')))

    return OLLSTM(state_0, mem_0, rnn, decoder=decoder, reset=reset)


def make_mlp(v: int, hid_dim: int, out_dim: int = -1, nonlin: Fn = relu, bias: bool = False, decoder: str = 'sum-lin') \
        -> OLMLP:
    """
    Builds a 1-hidden-layer Multi-Layer Perceptron with at decoder, meant for classification

    The actual architecture is controlled by the `mode` parameter according to:
        reduction over token-embeddings before logits: sum (`sum`), mean (`mean`), product (`prod`),
            first token (`first`) or last token (`last`); only valid in conjunction with `class` task
        readout: linear (empty string) vs linear+softmax ('xent')

    :param v: input vocabulary size
    :param hid_dim: size of hidden dimension
    :param out_dim: size of output dimension
    :param bias: whether a bias should be added to the linear transforms
    :param nonlin: the type of linearity in the hidden layer
    :param reset: whether the initial state should be reset during training
    :param decoder: type of architecture, based on readout and reduction/aggregation layer type
    :return: a classifier MLP
    """
    check(v, int, lambda: v > 0).check(hid_dim, int, lambda: hid_dim > 0)
    check(out_dim, int, lambda: out_dim == -1 or out_dim > 0).check(nonlin, Fn).check(val=decoder in MLP_MODES)

    normalise = '-xent' in decoder
    reduction = decoder.replace('-xent', '').split('-')[0]

    return OLMLP(embed=Linear(v, hid_dim, bias),
                 nonlin=FuncModule(nonlin),
                 decoder=Sequential(OrderedDict(red=Reducer(reduction),
                                                logits=Linear(hid_dim, out_dim, bias),
                                                y=IdentitySoftmax(normalise=normalise))))


@inference_mode()
def make_rumelhart_mlp(hid_dim: int = 2) -> OLMLP:
    """
    Builds a 1-layer MLP that solves PARITY for sequences of a given length, a per Rumelhart et al. 1986. "Learning
    Internal Representations by Error Propagation"

    :param hid_dim: size of hidden layer which should equal the length of the input sequence
    :return: a classifier MLP
    """
    check(hid_dim, int, lambda: hid_dim >= 1)

    hid_weight = ones(hid_dim, hid_dim)
    hid_bias = tensor([-(i + 1) + .5 for i in range(hid_dim)])

    out_weight = zeros(2, hid_dim)
    out_weight[0] = tensor([(-1) ** (i + 1) for i in range(hid_dim)])
    out_weight[1] = tensor([(-1) ** i for i in range(hid_dim)])
    out_bias = tensor([.5, -.5])

    threshold = FuncModule(px(heaviside, values=zeros(1)))

    return OLMLP(embed=_make_linear(hid_weight, hid_bias),  # token representation should be binary, not one-hot
                 nonlin=threshold,
                 decoder=Sequential(OrderedDict(logits=_make_linear(out_weight, out_bias), y=threshold)))


def make_cnn(v: int, hid_dim: int, out_dim: int, k: int, bias: bool = False, mode: Literal['avg', 'max'] = 'max') \
        -> OLCNN:
    """
    Builds a 1-hidden-layer Convolutional Neural Network, meant for classification

    :param v: input vocabulary size
    :param hid_dim: size of hidden dimension
    :param k: size of convolution kernel
    :param out_dim: size of output dimension
    :param bias: whether a bias should be added to the linear transforms
    :param mode: type of global pooling layer: `avg` for max pool aor `max` for average pool
    :return: a classifier CNN
    """
    check(v, int, lambda: v > 0).check(hid_dim, int, lambda: hid_dim > 0)
    check(k, int, lambda: k > 1).check(out_dim, int, lambda: out_dim == -1 or out_dim > 0)
    check(val=mode in {'avg', 'max'}, msg=f'[{mode}]')

    conv = Conv1d(v, hid_dim, k, bias=bias)
    conv.register_forward_pre_hook(_cnn_conv_prehook, prepend=True)

    pool = AdaptiveMaxPool1d(1) if mode == 'max' else AdaptiveAvgPool1d(1)
    pool.register_forward_hook(_cnn_pool_posthook, prepend=True)

    return OLCNN(conv, pool, Linear(hid_dim, out_dim, bias=bias), IdentitySoftmax(normalise=True))


def make_transformer(v: int,
                     hid_dim: int,
                     out_dim: int,
                     length: int,
                     depth: int = 1,
                     h: int = 1,
                     drop: float = 0,
                     tie: bool = False,
                     nonlin: Module = GELU('tanh'),
                     factor: int = 4,
                     bias: bool = False,
                     mode: str = 'class-sum-xent') -> GPT:
    """
    Builds an encoder- or decoder-only-transformer, depending on the arguments, and based on the GPT-architecture;
    follows the conventions in https://github.com/karpathy/nanoGPT

    The actual architecture is controlled by the `mode` parameter according to:
        task: classification/encoder (`class`) vs sequence-to-sequence/decoder/masked-attention (`seq2seq`)
        reduction over token-embeddings before logits: sum (`sum`), mean (`mean`), product (`prod`),
            first token (`first`) or last token (`last`); only valid in conjunction with `class` task
        readout: linear (empty string) vs linear+softmax ('xent')

    :param v: vocabulary size
    :param length: maximum sequence length / context size
    :param hid_dim: embedding dimension
    :param out_dim: output dimension
    :param depth: number of blocks
    :param h: number of heads per block
    :param bias: whether a bias should be added to the linear layers
    :param drop: the amount of dropout to use during training
    :param tie: whether output and input embeddings should share the same parameters, meant for next-token prediction
    :param nonlin: the type of non-linearity used in the feedforward layers
    :param factor: determines the size of the feedforward layers' hidden layer as a factor of the embedding dimension
    :param mode: type of architecture, based on task, readout and reduction/aggregation layer type
    :return: a transformer
    """
    check(v, int, lambda: v > 0).check(length, int, lambda: length > 0).check(hid_dim, int, lambda: hid_dim > 0)
    check(out_dim, int, lambda: out_dim == -1 or out_dim > 0).check(depth, int, lambda: depth > 0)
    check(h, int, lambda: h > 0 and not hid_dim % h).check(drop, Number, lambda: 0 <= drop <= 1)
    check(nonlin, Fn).check(factor, int, lambda: factor > 0).check(val=mode in TRANSFORMER_MODES, msg=str(mode))

    wte = Embedding(v, hid_dim)
    wpe = PE(Embedding(length, hid_dim))
    dropout = Dropout(drop)
    blocks = [_make_block(hid_dim, length=length, num_heads=h, bias=bias, drop=drop, nonlin=nonlin, factor=factor,
                          masked=mode.startswith('seq2seq'))
              for _ in range(depth)]

    ln = LayerNorm(hid_dim, bias=bias)

    reduction = 'none' if mode.startswith('seq2seq') else mode.split('-')[1]
    head = Sequential(OrderedDict(red=Identity() if reduction == 'none' else Reducer(reduction=reduction),
                                  logits=Linear(hid_dim, out_dim, bias=bias),
                                  y=IdentitySoftmax(normalise=mode.endswith('-xent'))))

    return GPT(wte, wpe, dropout, blocks, ln, head, tying=tie)


@inference_mode()
def make_chiang_transformer(factor: float = 1e3) -> GPT:
    """
    Builds a 2-head 2-layer encoder-transformer without layer normalisation that solves PARITY as described in
    Chiang & Cholak, 2022. "Overcoming a theoretical limitation of self-attention" and based on
    https://github.com/ndnlp/parity but following the conventions in https://github.com/karpathy/nanoGPT

    :param factor: the factor amplifying the logits, the larger it is the lower the entropy of the output distribution
    :return: a classifier Transformer
    """
    check(factor, float, lambda: factor > 0)

    hid_dim = 10
    num_heads = 2
    ffw_dim = 3
    out_dim = 2

    z = zeros(hid_dim, hid_dim)
    half_z = zeros(hid_dim // num_heads, hid_dim)
    drop = Dropout(0)

    we = ChiangTokenEmbedding()  # 0: [1, 0, 0], 1: [0, 1, 0], CLS: [0, 0, 1]
    pe = ChiangPositionEmbedding()  # [0, 0, 0, i/n, cos(iπ)]; dim 4 in [0, (n-1)/n], dim 5 in {-1, 1}

    # attention weights are always all 1/n
    # attention combo has k/n in 1st dim and 1/n in 2nd dim
    # att-out puts 1st dim into 6th dim and 2nd dim into 7th dim
    # att-res has [I[i=0], I[i=1], I[i=CLS], 1/n cos(iπ), k/n 1/n]
    v1 = half_z.clone()
    v1[0, 1] = 1  # indicates 1s in the 1st dim of V matrix, which computes number of 1s (k)
    v1[1, 2] = 1  # indicates CLS the 2nd dim of V matrix
    projs = _make_linear(cat([half_z.clone()] * 4 + [v1, half_z.clone()], dim=0))  # v1v2k1k2v1v2

    out = z.clone()
    out[5, 0] = 1  # k/n
    out[6, 1] = 1  # 1/n
    out = _make_linear(out)

    # after expansion and relu, hidden layer has [max(0, k-i-1),max(0, k-i),max(0, k-i+1)]
    exp = zeros(ffw_dim, hid_dim)
    exp[0, 3] = -1  # k-i-1
    exp[0, 5] = 1
    exp[0, 6] = -1
    exp[1, 3] = -1  # k-i
    exp[1, 5] = 1
    exp[2, 3] = -1  # k-i+1
    exp[2, 5] = 1
    exp[2, 6] = 1
    ffw_exp = _make_linear(exp)

    con = zeros(hid_dim, ffw_dim)  # computes I[i=k] and puts it in the 8th dim
    con[7, 0] = 1
    con[7, 1] = -2
    con[7, 2] = 1
    ffw_con = _make_linear(con)

    block_1 = Block(ln_1=NoOpLayerNorm(),
                    att=SelfAttention(projs, drop, out, drop, mask=empty(0), num_heads=num_heads),
                    ln_2=NoOpLayerNorm(),
                    ffw=FFW(ffw_exp, ReLU(), ffw_con, drop))

    q1 = half_z.clone()  # Q1 indicates CLS
    q1[0, 2] = 1  # = sqrt(10)/sqrt(10)
    q2 = q1.clone()  # Q2 also indicates CLS

    k1 = half_z.clone()  # K1 indicates oddity (-1 in paper)
    k1[0, 4] = 1
    k2 = k1.clone() * -1  # K2 indicates parity (1 in paper)

    v1 = half_z.clone()  # V1 copies 8th dim, indicates I[i=k]
    v1[0, 7] = 1
    v2 = v1.clone()  # V1 also copies 8th dim (-1 in paper), indicates I[i=k]

    # head 1 attention weights up odd positions, head 2 even positions, but only at the last token (CLS)
    # attention-combo puts smaller weight in 1st dim and larger in 6th dim for parity (resp. oddity) at CLS position
    # attention-out puts positive weight in 9th dim if k odd and negative if k even at CLS position, this is the first
    # layer where parity/oddity is discriminated in a single dimension
    projs = _make_linear(cat([q1, q2, k1, k2, v1, v2], dim=0))

    out = z.clone()  # subtracts weights at 6th dim from weights at 1st dim
    out[8, 0] = -1
    out[8, 5] = 1
    out = _make_linear(out)

    block_2 = Block(ln_1=NoOpLayerNorm(),
                    att=SelfAttention(projs, drop, out, drop, mask=empty(0), num_heads=num_heads),
                    ln_2=NoOpLayerNorm(),
                    ffw=FFW(_make_linear(zeros(ffw_dim, hid_dim)),
                            ReLU(),
                            _make_linear(zeros(hid_dim, ffw_dim)),
                            drop))

    dec = zeros(out_dim, hid_dim)
    dec[1, 8] = factor  # multiple of 9th dim, it's `1` in the Chiang & Cholak paper
    decoder = _make_linear(dec)
    head = Sequential(OrderedDict(red=Reducer(reduction='last'),  # "last" is not like BERT where it's "first"
                                  logits=decoder,
                                  y=IdentitySoftmax(normalise=True)))

    return GPT(we, pe, drop, [block_1, block_2], ln=NoOpLayerNorm(), head=head)


# ---------------------------------------- DELEGATE FUNCTIONS ----------------------------------------------------------


def _cnn_conv_prehook(_, x: Tensor) -> Tensor:
    """
    ensures the sequence axis is put last as expected by Conv1D; and pads the sequence with a dummy initial token
    to ensure the hidden layer (convolution) learns an embedding for the initial state, just like RNNs do.
    """
    return pad(x[0].transpose(-1, -2), pad=(1, 0))  # BxTxD -> BxDxT -> BxDxT+1


def _cnn_pool_posthook(_, __, y: Tensor) -> Tensor:
    """ ensures the logit layer gets the tensor shape it expects, ie, no length axis """
    return y.squeeze(-1)  # BxDx1 -> BxD


def _make_block(dim: int,
                length: int,
                num_heads: int = 1,
                bias: bool = False,
                drop: float = 0,
                nonlin: Module = GELU('tanh'),
                factor: int = 4,
                masked: bool = True) -> Block:
    ln_1 = LayerNorm(dim, bias=bias)
    attn = _make_attention(dim=dim, length=length, num_heads=num_heads, bias=bias, drop=drop, masked=masked)
    ln_2 = LayerNorm(dim, bias=bias)
    mlp = _make_ffw(dim, bias, drop, nonlin=nonlin, factor=factor)
    return Block(ln_1, attn, ln_2, mlp)


def _make_attention(dim: int, length: int, num_heads: int = 1, bias: bool = False, drop: float = 0, masked: bool = True) \
        -> SelfAttention:
    projs = Linear(dim, 3 * dim, bias=bias)
    att_drop = Dropout(drop)
    out = Linear(dim, dim, bias=bias)
    res_drop = Dropout(drop)
    # causal/right mask (decoder) or no mask (encoder)
    mask = tril(ones(length, length)).view(1, 1, length, length) if masked else empty(0)

    return SelfAttention(projs, att_drop, out, res_drop, mask, num_heads)


def _make_ffw(dim: int, bias: bool = False, drop: float = 0, nonlin: Module = GELU('tanh'), factor: int = 4) -> FFW:
    expansion = Linear(dim, factor * dim, bias=bias)
    contraction = Linear(factor * dim, dim, bias=bias)
    dropout = Dropout(drop)

    return FFW(expansion, nonlin, contraction, dropout)


def _make_linear(weight: Tensor, bias: Tensor | None = None) -> Linear:
    lin = Linear(weight.shape[-1], weight.shape[0], bias=bias is not None)
    lin.weight.copy_(weight)
    if bias is not None:
        lin.bias.copy_(bias)
    return lin


import torch

add_safe_globals([WFSA, OLSRNN, OLGRU, OLLSTM, OLMLP, OLCNN, GPT, ModuleDict, Embedding, PE, Dropout, RNN, Sequential,
                  Linear, IdentitySoftmax, LSTM, GRU, ModuleList, Block, LayerNorm, SelfAttention, Softmax, FFW, ReLU,
                  Reducer, FuncModule, Conv1d, AdaptiveMaxPool1d, TwoClassDecoder, torch.nn.functional.relu, partial,
                  _reduce, str, set, _cnn_conv_prehook, _cnn_pool_posthook])
