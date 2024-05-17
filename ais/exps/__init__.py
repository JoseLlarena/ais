from typing import Tuple

from matplotlib.colors import to_rgba

from ais import Fn
from ais.boolean_data import make_dataset_partitions
from ais.data import DATA
from ais.viz.drawing import R, B

BINARY_COLOURS = to_rgba(R, .1), to_rgba(B, .15)


def make_standard_partitions(task: Fn, scheme: str = 'class') \
        -> Tuple[DATA, DATA, DATA, Tuple[bool | str, ...], Tuple[bool | str, ...]]:
    return make_dataset_partitions(task, max_lengths=[12, 18, 24], ns=[-1, 100, 100], scheme=scheme)
