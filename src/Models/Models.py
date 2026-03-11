import torch.nn as nn

from src.Models import egamma


def get_model(tag: str, input_dim: int) -> nn.Module:
    if tag == "V1":
        return egamma.ModelV1(input_dim)
    if tag == "V1_tanh":
        return egamma.ModelV1_tanh(input_dim)
    if tag == "V2":
        return egamma.ModelV2(input_dim)
    if tag == "V3":
        return egamma.ModelV3(input_dim)
    if tag == "V4":
        return egamma.ModelV4(input_dim)
    if tag == "V5":
        return egamma.ModelV5(input_dim)
    if tag == "Run2_ModelV1":
        return egamma.Run2_ModelV1(input_dim)
    raise Exception("No model tag was informed")
