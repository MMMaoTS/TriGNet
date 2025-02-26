from model.TriGNet import Net
from model.Baseline import Baseline
from model.Ablation import Ablation
import config


def Network(model: str):
    if model == "Net":
        return Net()
    elif model == "Baseline":
        return Baseline()
    elif model == "Ablation":
        return Ablation()