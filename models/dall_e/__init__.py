import io, requests
import torch
import torch.nn as nn

import sys
import os
import urllib
from tqdm import tqdm
from os.path import dirname

sys.path.append(dirname(__file__))

from .encoder import Encoder
from .decoder import Decoder
from .utils import map_pixels, unmap_pixels

# constants
CACHE_PATH = os.path.expanduser("~/.cache/muge")

# helpers methods


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location = torch.device('cpu'))


def download(url, filename = None, root = CACHE_PATH, load_only=False):
    os.makedirs(root, exist_ok = True)
    filename = default(filename, os.path.basename(url))

    download_target = os.path.join(root, filename)
    download_target_tmp = os.path.join(root, f'tmp.{filename}')

    if load_only is False:
        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")

        if os.path.isfile(download_target):
            return download_target

        with urllib.request.urlopen(url) as source, open(download_target_tmp, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

        os.rename(download_target_tmp, download_target)

    return download_target

