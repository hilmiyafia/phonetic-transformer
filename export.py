import os
import torch
import subprocess
import torch.onnx
import config as cfg
from model import Transformer
from utils import DictDataset

class Wrapper(torch.nn.Module):

  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, tokens):
    buffer = self.model.forward(tokens[None])
    return buffer[0, -1].argmax(-1)

if __name__ == "__main__":
  dataset = DictDataset(cfg.DATASET_PATH)
  model = Transformer(
    len(dataset.tokens), 
    cfg.MODEL_DEPTH, 
    cfg.MODEL_HEAD, 
    cfg.MODEL_DIM, 
    cfg.MODEL_HIDDEN_DIM)
  model.load_state_dict(
    torch.load("outputs/model.pt"))
  model.eval()
  dummy = torch.randint(0, len(dataset.tokens), (5,))

  os.makedirs("outputs", exist_ok=True)
  torch.onnx.export(
    Wrapper(model), 
    dummy, 
    "outputs/model.onnx",
    input_names=["tokens"],
    output_names=["next_token"],
    dynamic_shapes={
      "tokens": {0: "token_count"}
    },
    external_data=False)
  subprocess.call([
    "onnxsim", 
    "outputs/model.onnx", 
    "outputs/simplified.onnx"])
