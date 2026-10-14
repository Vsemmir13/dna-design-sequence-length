"""Unconditioned author CNN; see docs/third_party.md for the source checkout."""
import hashlib
import importlib.util
from pathlib import Path

AUTHOR_MODEL_SHA256 = "41488cd15de322aef0c2a3953937059bc918deb498b4b85a4887d99607529847"


def load_author_cnn(source_dir):
    """Import the audited source without downloading or modifying it."""
    path = Path(source_dir).resolve() / "dna_model.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != AUTHOR_MODEL_SHA256:
        raise ValueError("Author source differs from the pinned implementation")
    spec = importlib.util.spec_from_file_location("nonb_dna_author_cnn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CNNModel


def make_discrete_flow_model(*, source_dir):
    """Build the unconditioned reference CNN, not the joint Tm/CFG checkpoint."""
    CNNModel = load_author_cnn(source_dir)
    return CNNModel(vocab_size=5, hidden_dim=128, num_cnn_stacks=4,
                    p_dropout=.3, num_classes=1, cls_free_guidance=False)
