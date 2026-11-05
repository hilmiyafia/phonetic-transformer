import torch
import config as cfg
from model import Transformer
from utils import DictDataset
import sys

if __name__ == "__main__":
  text = sys.argv[1]
  dataset = DictDataset(cfg.DATASET_PATH)
  model = Transformer(
    len(dataset.tokens), 
    cfg.MODEL_DEPTH, 
    cfg.MODEL_HEAD, 
    cfg.MODEL_DIM, 
    cfg.MODEL_HIDDEN_DIM)
  model.load_state_dict(torch.load("outputs/model.pt"))
  part_a = []
  part_b = []
  for word in text.split(" "):
    if word == "" or word.isalpha() == False: 
      continue
    tokens = dataset.encode([
      cfg.DATASET_START_SYMBOL, 
      *list(word), 
      cfg.DATASET_EQUAL_SYMBOL])
    result = dataset.decode(model.sample(
      tokens, 
      dataset.tokens.index(cfg.DATASET_END_SYMBOL)))
    phonemes = [[]]
    for item in result:
      if item == cfg.DATASET_START_SYMBOL: 
        continue
      if item == cfg.DATASET_END_SYMBOL:
        continue
      if item == cfg.DATASET_EQUAL_SYMBOL: 
        phonemes.append([])
        continue
      phonemes[-1].append(item)
    part_a.append("".join(phonemes[0]))
    part_b.append("_".join(phonemes[1]))
  print(f"{' '.join(part_a)}\n= {' '.join(part_b)}")
