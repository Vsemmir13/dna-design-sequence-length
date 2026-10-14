"""CPU-only capacity and interface tests; no training or claimed quality."""
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.lstm.unconditional.architecture import G4LSTM
from models.lstm.conditional.architecture import G4TmLSTM, TemperatureEncoder


def count(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_unconditional_capacity_and_no_condition_encoder():
    model=G4LSTM().eval()
    assert count(model)==3662383
    assert not any("condition" in key or "length" in key for key in model.state_dict())
    with torch.no_grad():assert model(torch.tensor([[4,0,1,2,3]])).shape==(1,5,7)


def test_continuous_condition_and_capacity():
    model=G4TmLSTM(tm_mean=60,tm_std=10).eval()
    assert count(model)==3665463 and count(TemperatureEncoder())==88
    tokens=torch.tensor([[4,0,1]])
    with torch.no_grad():
        low=model(tokens,torch.tensor([50.]));high=model(tokens,torch.tensor([80.]))
    assert low.shape==high.shape==(1,3,7) and not torch.equal(low,high)

