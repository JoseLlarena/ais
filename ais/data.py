"""
Functions and constants to transform generic data
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby, zip_longest
from random import shuffle, sample
from types import SimpleNamespace
from typing import Tuple, Dict, TypeVar, Generic, Any, Iterable, Iterator, List, Mapping, Protocol

from more_itertools import unzip, divide
from torch import randperm, Tensor, stack, cat, concat, tensor, ones_like, empty, equal
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader

from ais import Fn, check, T, UNK, ident, BOS, last, ZERO, ONE

X, Y = TypeVar('X'), TypeVar('Y')

KINDS = {'class', 'class-id', 'class-pad', 'class-pad-id', 'class-polar',
         'lm',
         'seq2seq', 'seq2seq-id', 'seq2seq-pad', 'seq2seq-pad-id'}


class EmptyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, item):
        return ()

    def __len__(self) -> int:
        return 0


EMPTY_DATASET = EmptyDataset()


class SupeDataset(Dataset):
    """
    Dataset for supervised learning, for inputs having the same length
    """

    def __init__(self, xs: Tensor, ys: Tensor):
        super().__init__()  # ASSUMES BxTxD or BxT
        check(xs, Tensor, lambda: xs.ndim in {2, 3}).check(ys, Tensor, lambda: ys.ndim in {2, 3})
        self.xs = xs
        self.ys = ys

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.xs[index], self.ys[index]

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        return iter(zip(self.xs, self.ys))

    def __len__(self) -> int:
        return self.xs.shape[0]

    def shuffle(self) -> SupeDataset:
        perm = randperm(self.xs.shape[0])
        return SupeDataset(self.xs[perm], self.ys[perm])

    def cat(self, other: SupeDataset) -> SupeDataset:
        return SupeDataset(cat([self.xs, other.xs]), cat([self.ys, other.ys]))

    def chunk(self, num: int) -> Tuple[SupeDataset, ...]:
        x_parts = self.xs.chunk(num)
        y_parts = self.ys.chunk(num)

        return tuple(SupeDataset(xs, ys) for xs, ys in zip(x_parts, y_parts))


class RaggedSupeDataset(Dataset):
    """
    Dataset for supervised learning, for inputs not having the same length. Meant to avoid padding.
    """

    def __init__(self, data: Sequence[Tuple[Tensor, Tensor]]):
        super().__init__()
        check(data, Sequence, lambda: len(data) > 0)

        self.data = data

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.data[index]

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, other):
        if isinstance(other, RaggedSupeDataset):

            for this, that in zip_longest(self, other):
                if not equal(this[0], that[0]) or not equal(this[1], that[1]):
                    return False

            return True

        return False

    def cat(self, other: RaggedSupeDataset) -> RaggedSupeDataset:
        check(other, RaggedSupeDataset)

        return RaggedSupeDataset(tuple(self.data) + tuple(other.data))

    def chunk(self, num: int) -> Tuple[RaggedSupeDataset, ...]:
        return tuple(RaggedSupeDataset(tuple(part)) for part in divide(num, self.data))


DATA = TypeVar('DATA', SupeDataset, RaggedSupeDataset)


class LengthSampler:
    """
    Samples batches from within subsets of data with the same input length
    """

    def __init__(self, lengths: Iterable[int], *, bs: int = -1, shuffled: bool = True):
        """
        :param lengths: lengths of input data
        :param bs: maximum batch size; if -1, the size of the length-group will be used
        :param shuffled: whether batches should be shuffled
        """
        super().__init__()
        check(lengths, Iterable).check(bs, int, lambda: bs == -1 or bs > 0)
        lengths = tuple(lengths)
        check(val=len(lengths) > 0)

        self.ids = tuple(range(len(lengths)))
        self.orig_lengths = [tuple(group) for group in map(last, groupby(self.ids, lambda idx: lengths[idx]))]
        self._sampled_lengths = list(self.orig_lengths)
        self.n = len(lengths)

        self.bs = bs if bs != -1 else int(1e12)
        self.shuffled = shuffled

    def __iter__(self) -> Iterator[List[int]]:
        """
        resets list of length groups to sample from

        :return: itself, a collection of groups of ids, each group corresponding to inputs having the same length
        """
        if self.shuffled:
            shuffle(self.orig_lengths)
        self._sampled_lengths = list(self.orig_lengths)

        return self

    def __next__(self) -> List[int]:
        """ samples the next batch """
        if not self._sampled_lengths:
            raise StopIteration()
        # first samples a same-length group
        # then samples data points from the chosen same-length group
        # returned batch size is the smaller of the group size and the batch size passed to the constructor
        return sample(batch := self._sampled_lengths.pop(), k=min(len(batch), self.bs))

    def __len__(self) -> int:
        return self.n


def as_dataset(inputs: Iterable[Sequence[X]],
               target: Fn[[Sequence[X]], Sequence[Y]],
               x_vocab: Iterable[X],
               y_vocab: Iterable[Y],
               kind: str) -> DATA:
    """
    Builds a Pytorch dataset for sequence classification and transduction.
    `kind` determines the choice of dataset type and input and output encodings:

        '' vs '-pad':           variable-length (RaggedSupeDataset) vs padded/fixed-length (SupeDataset)
        '' vs '-id':            one-hot-encoding vs integer-encoding for inputs
        '' vs '-polar' vs 'lm': one-hot-encoding vs polar-encoding vs lm-encoding for targets
        'class' vs 'seq2seq':   classification/rank-1 (BxK) vs transduction/rank-2 (BxLxK) for targets

    :param inputs: a collection of sequences
    :param target: a function to map a sequence to its target
    :param x_vocab: the input vocabulary
    :param y_vocab: the output vocabulary
    :param kind: determines the input and output encodings, and the dataset type
    :return: a Pytorch iterable dataset
    """
    check(inputs, Iterable).check(target, Fn).check(x_vocab, Iterable).check(y_vocab, Iterable)
    x_vocab, y_vocab = map(tuple, [x_vocab, y_vocab])
    check(val=x_vocab and y_vocab, msg=f'[{len(x_vocab)}], [{len(y_vocab)}]')
    check(val=kind in KINDS, msg=f'[{kind}]')

    flat, fixed, id_, polar = kind.startswith('class'), '-pad' in kind, kind.endswith('-id'), '-polar' in kind

    x_coder = (id_coder_from if id_ else one_hot_coder_from)(x_vocab)
    y_coder = SimpleNamespace(tensorise=lambda x, *args, **kwargs: tensor(x).float()) if kind == 'lm' else \
        (polar_coder_from if polar else one_hot_coder_from)(y_vocab)

    XY = ((seq, target(seq)) for seq in inputs)
    xs, ys = unzip((x_coder.tensorise(x) if x else empty(0, x_coder.n), y_coder.tensorise(y, flat=flat)) for x, y in XY)

    return SupeDataset(stack(tuple(xs)), stack(tuple(ys))) if fixed else RaggedSupeDataset(tuple(zip(xs, ys)))


def bert_target(seq: Iterable[X], task: Fn[[Iterable[X]], Y]) -> Tuple[Y, ...]:
    # assumes first token is CLS-like and thus ignores it for the purposes of computing the target
    seq = tuple(seq)
    return class_target(seq, task, bos=seq[0])


def class_target(seq: Iterable[X], task: Fn[[Iterable[X]], Y], bos: X = BOS) -> Tuple[Y, ...]:
    # if the sequence starts with a beginning-of-sequence token, it removes it, assuming `task` does not know how
    # to deal with it (otherwise `bos` should be set to a different value). This leaves sequences made of only `bos`
    # empty, and so it assumes `task` knows how to deal with empty sequences
    seq = tuple(seq)

    if seq and seq[0] == bos:
        seq = seq[1:]

    return task(seq),


def seq2seq_target(seq: Iterable[X], task: Fn[[Iterable[X]], Y], bos: X = BOS) -> Tuple[Y, ...]:
    seq = tuple(seq)

    if seq[0] == bos:
        seq = seq[1:]
        y_bos = task(()),
    else:
        y_bos = ()

    prefixes = [seq[:t + 1] for t in range(len(seq))]

    return y_bos + tuple(task(prefix) for prefix in prefixes)


class Coder(Generic[T]):
    """
    Encapsulates the encoding and decoding of sequences into and from tensors.
    """

    def __init__(self, vocab: Iterable[T], tens: Fn = ident, untens: Fn = ident, *, bias: bool = False):
        """
        :param vocab: vocabulary
        :param tens: sequence-to-tensor function
        :param untens: tensor-to-sequence function
        :param bias: whether to add an extra dimension to account for a linear bias
        """
        self.idx_to_token = dict(enumerate(vocab))
        self.token_to_idx = {token: idx for idx, token in self.idx_to_token.items()}
        self.tens = tens
        self.untens = untens
        self.bias = bias

    @property
    def n(self) -> int:
        """
        Returns the number of tokens in the vocabulary.

        :return: the size of the vocabulary
        """
        return len(self.token_to_idx)

    def tensorise(self, seq: Sequence[T], flat: bool = False) -> Tensor:
        """
        Converts a sequence to a tensor

        :param seq: sequence to be converted
        :param flat: True for a V(+1)-shaped tensor , False for a NxV(+1)-shaped tensor
        :return: a tensorised sequence
        """
        as_tensor = self.tens(seq, self.token_to_idx)
        if self.bias:
            as_tensor = with_bias(as_tensor)
        return as_tensor.view(1, -1).squeeze(0) if flat else as_tensor

    def untensorise(self, seq: Tensor) -> Tuple[T, ...]:
        """
        Converts a tensor into a sequence

        :param seq: tensored sequence to be untensorised
        :return: sequence
        """
        if self.bias:
            seq = without_bias(seq)

        return self.untens(seq, self.idx_to_token)

    def has_bias(self) -> bool:
        return self.bias


def binary_coder_from(zero: Any = ZERO, one: Any = ONE) -> Coder[T]:
    """
    Hacky binary encoder meant for the Rumelhart-XOR-MLP

    :param zero: the character to represent True
    :param one: the character to represent False
    :return: a tensorise-only Coder
    """
    coder = SimpleNamespace(tensorise=lambda seq: tensor([[float(str(token) == one) for token in seq]]),
                            untensorise=lambda tens: tuple(zero if val == 0. else one for val in tens.squeeze(0)),
                            n=lambda: 2,
                            has_bias = lambda: False)  # FIXME HACK
    return coder


def one_hot_coder_from(vocab: Iterable[T]) -> Coder[T]:
    """
    Creates a one-hot coder
    :param vocab: vocabulary of sequences to encode/decode
    :return: coder
    """
    check(vocab, Iterable)
    vocab = tuple(vocab)
    check(val=len(vocab) > 0)

    return Coder(vocab, one_hot_tensorise, one_hot_untensorise)


def polar_coder_from(vocab: Iterable[T]) -> Coder[T]:
    """
    Creates a polar {-1, 1} coder
    :param vocab: vocabulary of sequences to encode/decode
    :return: coder
    """
    check(vocab, Iterable)
    vocab = tuple(vocab)
    check(val=len(vocab))

    def polar_tensorise(sequence: Sequence[T], token_to_id: Dict[T, int]) -> Tensor:
        tens = one_hot_tensorise(sequence, token_to_id)
        tens[tens == 0] = -1
        return tens

    def polar_untensorise(seq_tensor: Tensor, id_to_token: Dict[int, T]) -> Tuple[T, ...]:
        seq_tensor[seq_tensor == -1] = 0
        return one_hot_untensorise(seq_tensor, id_to_token)

    return Coder(vocab, polar_tensorise, polar_untensorise)


def id_coder_from(vocab: Iterable[T]) -> Coder[T]:
    """
    Creates an integer-id coder, meant for torch.nn.Embedding

    :param vocab: vocabulary of tokens to encode/decode
    :return: coder
    """
    check(vocab, Iterable)
    vocab = tuple(vocab)
    check(val=len(vocab) > 0)

    return Coder(vocab, int_tensorise, id_untensorise)


def id_untensorise(seq_tensor: Tensor, id_to_token: Mapping[int, T]) -> Tuple[T, ...]:
    """
    Converts tensor into sequence of tokens. Assumes `seq_tensor` is a 1st-order tensor, ie, it's a sequence of
    ids and looks them up

    If a token is not found, the index of the `UNK` special string is used instead. This is meant for handling OOVs.

    :param seq_tensor: a tensor if shape N
    :param id_to_token: an id-to-token mapping, should contain the `UNK` token
    :return: a tuple of tokens
    """
    check(seq_tensor, Tensor, lambda: seq_tensor.ndim == 1, f'Tensor shape should be N but was [{seq_tensor.shape}]')
    check(id_to_token, Mapping, lambda: len(id_to_token) > 1)

    return tuple(id_to_token.get(idx, UNK) for idx in seq_tensor.tolist())


def one_hot_tensorise(sequence: Sequence[T], token_to_id: Mapping[T, int]) -> Tensor:
    """
    Converts a sequence of N tokens into a tensor of size NxK, where N is the length of the
    sequence and K is the size of `token_to_id`, ie, the vocabulary. In other words, a matrix whose rows correspond to
    the one-hot encoding of each token in the sequence, with a 1 in the dimension corresponding to the token index;
    meant for input to dense layers in Pytorch NNs. This is a localist encoding.

    If a token is not found, the index of the `UNK` special token is used instead. This is meant for handling OOVs.

    :param sequence: the sequence of K tokens to encode
    :param token_to_id: the mapping from token to identifier
    :return: a tensor of size NxK
    :raises: KeyError if any token, including `UNK`, is missing from `token_to_id`
    """
    check(sequence, Sequence, lambda: len(sequence) > 0, f'sequence is empty [{sequence}]')
    check(token_to_id, Mapping, lambda: len(token_to_id) > 1)

    indices = int_tensorise(sequence, token_to_id)

    return one_hot(indices, num_classes=len(token_to_id)).float()


def one_hot_untensorise(seq_tensor: Tensor, id_to_token: Mapping[int, T]) -> Tuple[T, ...]:
    """
    Converts NxK tensor into sequence of N tokens. Assumes `seq_tensor` is rank-2, ie, it's a sequence of
    one-hot-encoded vectors and looks up the index with the highest value in each row

    If a token is not found, the index of the `UNK` special token is used instead. This is meant for handling OOVs.

    :param seq_tensor: an NxK tensor representing a sequence of tokens
    :param id_to_token: an index-to-token mapping
    :return: an untensorised tuple of N strings
    """
    check(seq_tensor, Tensor, lambda: seq_tensor.ndim == 2, f'should be NxK but was [{seq_tensor.shape}]')
    check(id_to_token, Mapping, lambda: len(id_to_token) > 1 and len(id_to_token) == seq_tensor.shape[-1])

    indices = seq_tensor.argmax(dim=-1).long()

    return tuple(id_to_token.get(idx, UNK) for idx in indices.tolist())


def int_tensorise(sequence: Sequence[T], token_to_id: Mapping[T, int]) -> Tensor:
    """
    Converts a sequence of tokens into a tensor of size K where K is the length of the sequence.
    This is meant as input to sparse layers in Pytorch NNs. This is a localist encoding.

    If a token is not found, the index of the `UNK` special token is used instead. This is meant for handling OOVs.

    :param sequence: the sequence of tokens to encode
    :param token_to_id: the mapping from token to identifier
    :return: a tensor of size K
    :raises: KeyError if any token, including `UNK`, is missing from `token_to_id`
    """
    check(sequence, Sequence, lambda: len(sequence) > 0).check(token_to_id, Mapping, lambda: len(token_to_id) > 1)

    unk_id = token_to_id.get(UNK)
    return tensor([token_to_id.get(token, unk_id) for token in sequence])


def with_bias(seq: Tensor) -> Tensor:
    """
    Add a bias dimension to the tensorised sequence

    :param seq: unbiased tensorised sequence
    :return: biased tensorised sequence
    """
    check(seq, Tensor, lambda: seq.ndim == 2 and seq.shape[-1] > 0, lambda: f'{seq.shape}')
    return concat([seq, ones_like(seq[:, :1])], dim=-1)  # NxD -> NxD+1


def without_bias(seq: Tensor) -> Tensor:
    """
    Removes bias dimension from the tensorised sequence

    :param seq: biased tensorised sequence
    :return: unbiased tensorised sequence
    """
    check(seq, Tensor, lambda: seq.ndim == 2 and seq.shape[-1] > 0)
    return seq[:, :-1]  # NxD -> NxD-1


def format_input(seq: Sequence[X],
                 length: int = -1,
                 *,
                 pad: str | None = None,
                 bos: str | None = None,
                 eos: str | None = None) -> Tuple[X, ...]:
    """
    Formats input sequence ready for tensorisation

    :param seq: the sequence to be formatted
    :param length: the minimum sequence the sequence should have, assuming the length of the sequence is <= this
    :param pad: the padding token
    :param bos: the beginning-of-sequence token
    :param eos: the end-of-sequence token
    :return: formatted sequence
    """
    check(seq, Sequence, lambda: len(seq) >= 0)
    check(length, int, lambda: length == -1 or (0 < len(seq) <= length and pad))
    check(pad, str | None).check(bos, str | None).check(eos, str | None)

    seq = ((bos,) if bos else ()) + tuple(seq) + ((eos,) if eos else ())
    padding = (pad,) * (length - len(seq)) if pad else ()

    return seq + padding


def format_target(target: Fn[[Sequence[X]], Sequence[Y]], kind: str, length: int = -1, pad: str | None = None) \
        -> Fn[[Sequence[X]], Sequence[Y]]:
    """
    Formats the input sequence before being fed to the given target function

    :param target: the target function to be wrapped
    :param kind: the dataset kind, same as `kind` in `as_dataset()`
    :param length: the minimum sequence the sequence should have, assuming the length of the sequence is <= this
    :param pad: the padding token
    :return: wrapped target function
    """
    check(target, Fn).check(val=kind in KINDS)
    check(length, int, lambda: length == -1 or (length > 0 and pad)).check(pad, str | None)

    def _formatted_target(input: Sequence[X]) -> Sequence[Y]:
        y = tuple(target([token for token in input if token != pad]))
        return y + ((pad,) * (length - len(y)) if pad and kind.startswith('seq2seq') else ())

    return _formatted_target


def make_loader(dataset: Dataset | None, bs: int, shuffled: bool) -> DataLoader:
    """
    Convenience function to encapsulate the logic for both standard and length-sampled batches

    :param dataset: the dataset to be loaded
    :param bs: the batch size, negative for length-sampled batches, positive for normal batches
    :param shuffled: whether the data should be shuffled before sampling
    :return: the dataloader for the given dataset
    """
    check(dataset, Dataset).check(bs, int, lambda: bs)

    if not dataset:
        return DataLoader(EMPTY_DATASET)

    if bs < 0:  # special case for boolean functions, but more generally, to avoid padding at all
        return DataLoader(dataset,
                          batch_sampler=LengthSampler([len(x) for x, y in tuple(dataset)],
                                                      shuffled=shuffled,
                                                      bs=abs(bs) if bs < -1 else bs),
                          pin_memory=False)

    return DataLoader(dataset, batch_size=bs, shuffle=shuffled, num_workers=0, pin_memory=False)
