"""Load the pinned D3 CNN without vendoring upstream code; expose its null mode."""
import ast
from functools import cache
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SOURCE_COMMIT = '76d1976b03bb844531482f2104e589562a56880b'
SOURCE_HASHES = {
    'model/layers.py': '1fac59c96b0a8b4a0581e05db017a6b343650bef419dc025222b929cd9dc7b2e',
    'model/cnn.py': '76503d1e97d0160a845a5f27a3f61d0d73c16d7af6911b60e477eca9af4256d3',
    'model_zoo/promoter/models.py': 'c683df92587d195001b36313a3e5878bf802d5bdc5f9d0d12b2fc01968763fb5',
}


def _definitions(root, relative, names, namespace):
    tree = ast.parse((root/relative).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('Missing pinned author classes: '+relative)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(root/relative), 'exec'), namespace)


@cache
def _load_verified_classes(root):
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math,
                 'Optional': Optional, 'DictConfig': SimpleNamespace}
    # Configs are namespaces, not dicts requiring OmegaConf. Only these class
    # bodies execute: no FlashAttention, Lightning or author's module imports.
    _definitions(root, 'model/layers.py', ['TimestepEmbedder', 'GaussianFourierProjection', 'Dense'], namespace)
    _definitions(root, 'model/cnn.py', ['ConvolutionalModel'], namespace)
    _definitions(root, 'model_zoo/promoter/models.py', ['PromoterConvolutionalModel'], namespace)
    return SimpleNamespace(ConvolutionalModel=namespace['ConvolutionalModel'],
                          PromoterConvolutionalModel=namespace['PromoterConvolutionalModel'])


def load_author_cnn(source_dir):
    """Read a supplied checkout; no clone, install, source patch or GPU execution."""
    root = Path(source_dir).resolve(strict=True)
    for relative, expected in SOURCE_HASHES.items():
        path = root/relative
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Author D3 source differs from the pinned implementation: '+relative)
    return _load_verified_classes(root)


def make_unconditional_model(*, source_dir, **kwargs):
    """The same joint network/state-dict; pass labels=None to select learned null.

    This does not create or claim a separately trained unconditional checkpoint.
    """
    from .conditional import make_d3_model
    return make_d3_model(source_dir=source_dir, **kwargs)
