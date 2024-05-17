"""
Functions to save and load Pytorch models
"""

import re
from datetime import datetime
from hashlib import md5
from os.path import join
from typing import Any

from torch import save, load

from ais import Fn
from ais.model_creation import M

MODEL_DIR = f'../../models'


def register_model(model: M, metadata: str, registry: str = join(MODEL_DIR, 'registry.txt')) -> M:
    """
    Saves model and adds an entry to the registry, with the contents in `metadata` and the model's file path. The model
    is saved in the `MODEL_DIR` directory. The file name is a concatenation of the model's class's name and a hash of
    the metadata's contents.

    :param model: model to save
    :param metadata: metadata to save in the registry file
    :param registry: path to registry file
    :return: the saved model
    """
    path = join(MODEL_DIR, f'{model.__class__.__name__ + "-" + md5(metadata.encode()).hexdigest() + ".pt"}')
    save(model.eval(), path)

    with open(registry, 'a') as file:
        file.write(f'{path} {metadata}\n')

    return model


def find_registered_model(pattern: str, registry: str = join(MODEL_DIR, 'registry.txt')) -> M:
    """
    Finds a model that has previously been registered in the registry. The code will find the best matches and choose
    the most recent ones amongst them. The found model is given a `rec` attribute that contains its registry entry.

    :param pattern: a regular expression to match the model's entry in the registry
    :param registry: path to registry file
    :return: the found model
    """
    regex = re.compile('.*' + pattern + '.*')

    with open(registry) as file:
        matches = regex.findall(file.read())

    if not matches:
        raise ValueError(f'[{pattern}] could not be found in [{registry}]')

    record = sorted(matches, key=lambda line: datetime.fromisoformat(line.split()[1]))[-1]
    _, time, *rest = record.split()
    model = load(record.split()[0]).train(False)

    model.rec = ' '.join([datetime.fromisoformat(time).strftime('%d.%m.%y %H:%M:%S')] + rest).replace('=True',
                                                                                                      '').upper()

    return model


def make_metadata(**kwargs) -> str:
    """
    Makes a model metadata string from the given key-value pairs. It prepends the current datetime in ISO format.

    :param kwargs: metadata as key-value pairs
    :return: the metadata string
    """
    now = datetime.now()
    time = now.isoformat(timespec='seconds')

    extra_args = [f'{_name_of_key(key)}{_name_of_value(value, key)}' for key, value in kwargs.items()]

    return ' '.join([time] + extra_args)


# --------------------------------------- DELEGATE FUNCTIONS -----------------------------------------------------------

def _name_of_key(key: str) -> str:
    return f'{key}='.replace('epochs=', 'eps=').replace('model=', '').replace('init=', '').replace('task=', '') \
        .replace('shape=', '').replace('loss_fn=', 'error=')


def _name_of_value(value: Any, key: str) -> str:
    if key == 'loss_fn':
        name = f"{value.replace('Loss', '')}"
    elif key == 'model':
        name = f'{value.reg_str}'.replace('True', '✓').replace('False', '×')
    elif key == 'epochs':
        name = f'{value:3d}'
    elif key.endswith('loss'):
        name = f'{value.item():8.2e}'
    elif key == 'init' and isinstance(key, Fn):
        name = f'{value.__name__[:-1]:9.9}'
    elif key == 'task':
        name = f'{value.__name__ if isinstance(value, Fn) else value:7.7}'
    elif key == 'bs':
        name = f'{value:-3d}'
    elif key == 'C':
        name = 'V' if value else 'T'
    elif key == 'shape':
        name = f'{value:7.7}'
    elif hasattr(value, 'item') or isinstance(value, float):
        name = f'{value.item() if hasattr(value, "item") else value :1.1e}'.replace('0.', '.').replace('-0.', '-.')
    elif isinstance(value, bool):
        name = '✓' if value else '×'
    elif isinstance(value, Fn):
        name = value.__class__.__name__
    else:
        name = str(value)

    return name.replace('()', '').replace('OneCycleLR', 'OneCycle')
