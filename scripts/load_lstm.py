"""Load a hash-checked tensor-only released LSTM; no training or new cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.lstm.unconditional.architecture import G4LSTM


def load_released_lstm(*, regime, repo_id="vsemmir13/return-of-the-lstm-nonb-dna", local_root=None, revision=None):
    if regime not in {"short", "nt512", "nt1024"}:
        raise ValueError("Unknown released regime")
    prefix=f"lstm/unconditional/{regime}"
    def resolve(filename):
        relative=f"{prefix}/{filename}"
        return Path(local_root)/relative if local_root else Path(hf_hub_download(repo_id,relative,revision=revision))
    config=json.loads(resolve("config.json").read_text())
    metadata=json.loads(resolve("metadata.json").read_text())
    weights=resolve("model.safetensors")
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=metadata["released_weights_sha256"]:
        raise ValueError("Released weights hash mismatch")
    model=G4LSTM(**config["architecture"]).eval()
    model.load_state_dict(load_file(str(weights),device="cpu"),strict=True)
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=config["trainable_parameters"]:
        raise ValueError("Released parameter count mismatch")
    return model,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime",choices=["short","nt512","nt1024"],default="short")
    parser.add_argument("--repo_id",default="vsemmir13/return-of-the-lstm-nonb-dna")
    parser.add_argument("--revision")
    parser.add_argument("--local_root",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    model,metadata=load_released_lstm(regime=args.regime,repo_id=args.repo_id,
        local_root=args.local_root,revision=args.revision)
    with torch.inference_mode():logits=model(torch.tensor([[4,0,1,2,3]]))
    if not torch.isfinite(logits).all():raise ValueError("Nonfinite CPU forward")
    print(json.dumps({"regime":args.regime,"parameters":3662383,
        "selected_epoch":metadata["selected_epoch_one_based"],"cpu_forward_shape":list(logits.shape),
        "scientific_metrics":"PENDING"}))


if __name__=="__main__":main()

