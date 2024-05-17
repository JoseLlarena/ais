from torch import eye, inference_mode, arange
from torch.nn import ReLU
from torch.nn.init import constant_

from ais.model_creation import make_wfsa, make_cnn, make_mlp, make_rumelhart_mlp, make_olsrnn, make_olgru, make_ollstm, \
    make_transformer, make_chiang_transformer
from ais.sgd import M
from ais.tracing import as_traced_wfsa, as_traced_cnn, as_traced_mlp, as_traced_rnn, as_traced_transformer
from unittests import assert_close


@inference_mode()
def _init(net: M) -> M:
    for param in net.parameters():
        constant_(param, 1)

    return net


@inference_mode()
def test_traces_wfsa():
    input = eye(2).repeat(2, 1).unsqueeze(0)

    wfsa = _init(make_wfsa(v=2, hid_dim=2, mode='binary'))
    traced_wfsa = as_traced_wfsa(wfsa)

    assert_close(traced_wfsa(input)[:, -1], wfsa(input))
    assert traced_wfsa(input).shape[-2] == input.shape[-2] + 1


@inference_mode()
def test_traces_cnn():
    input = eye(2).repeat(2, 1).unsqueeze(0)

    cnn = _init(make_cnn(v=2, hid_dim=2, out_dim=2, k=2))
    traced_cnn = as_traced_cnn(cnn)

    assert_close(traced_cnn(input)[:, -1], cnn(input))
    assert traced_cnn(input).shape[-2] == input.shape[-2]


@inference_mode()
def test_traces_learned_mlp():
    input = eye(2).repeat(2, 1).unsqueeze(0)

    mlp = _init(make_mlp(v=2, hid_dim=2, out_dim=2))
    traced_mlp = as_traced_mlp(mlp, kind='learned')

    assert_close(traced_mlp(input)[:, -1], mlp(input))
    assert traced_mlp(input).shape[-2] == input.shape[-2]


@inference_mode()
def test_traces_rumelhart_mlp():
    input = eye(2).view(1, -1)

    mlp = make_rumelhart_mlp(hid_dim=input.shape[-1])
    traced_mlp = as_traced_mlp(mlp, kind='rumelhart')

    assert_close(traced_mlp(input)[:, -1], mlp(input))
    assert traced_mlp(input).shape[-2] == input.shape[-1]


@inference_mode()
def test_traces_srnn():
    input = eye(2).repeat(2, 1).unsqueeze(0)

    srnn = _init(make_olsrnn(v=2, hid_dim=2, out_dim=2))
    traced_srnn = as_traced_rnn(srnn)

    assert_close(traced_srnn(input)[:, -1], srnn(input))
    assert traced_srnn(input).shape[-2] == input.shape[-2] + 1


@inference_mode()
def test_traces_gru():
    input = eye(2).repeat(2, 1).unsqueeze(0)

    gru = _init(make_olgru(v=2, hid_dim=2, out_dim=2))
    traced_gru = as_traced_rnn(gru)

    assert_close(traced_gru(input)[:, -1], gru(input))
    assert traced_gru(input).shape[-2] == input.shape[-2] + 1


@inference_mode()
def test_traces_lstm():
    input = eye(2).repeat(2, 1).unsqueeze(0)

    lstm = _init(make_ollstm(v=2, hid_dim=2, out_dim=2))
    traced_lstm = as_traced_rnn(lstm)

    assert_close(traced_lstm(input)[:, -1], lstm(input))
    assert traced_lstm(input).shape[-2] == input.shape[-2] + 1


@inference_mode()
def test_traces_learned_transformer():
    input = arange(3).unsqueeze(0)

    mlp = _init(make_transformer(v=3, hid_dim=2, out_dim=2, length=input.shape[-1], nonlin=ReLU()))
    traced_mlp = as_traced_transformer(mlp, kind='learned')

    assert_close(traced_mlp(input)[-1:], mlp(input))
    assert traced_mlp(input).shape[-2] == input.shape[-1]


@inference_mode()
def test_traces_chiang_transformer():
    input = arange(3).unsqueeze(0)

    mlp = make_chiang_transformer()
    traced_mlp = as_traced_transformer(mlp, kind='chiang')

    assert_close(traced_mlp(input)[-1:], mlp(input))
    assert traced_mlp(input).shape[-2] == input.shape[-1]
