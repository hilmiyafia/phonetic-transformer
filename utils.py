import os
import torch
import config as cfg

class DictDataset(torch.utils.data.Dataset):

  def __init__(self, path, count=-1):
    self.count = count
    self.tokens = [
      cfg.DATASET_START_SYMBOL, 
      cfg.DATASET_EQUAL_SYMBOL, 
      cfg.DATASET_COMMA_SYMBOL,
      cfg.DATASET_END_SYMBOL]
    self.letters = []
    self.phonemes = []
    self.data = []
    self.masks = []
    buffer = {}
    with open(path, encoding="utf-8") as file:
      for line in file:
        data = line.strip("\n").split("#")[0].split(" ")
        data = list(filter(lambda x: x != "", data))
        data[0] = data[0].split("(")[0].lower()
        for i in range(1, len(data)):
          data[i] = data[i].upper()
        if data[0] not in buffer.keys():
          buffer[data[0]] = []
        buffer[data[0]].append(data[1:])
    for key in buffer.keys():
      self.data.append([cfg.DATASET_START_SYMBOL])
      self.masks.append([0])
      for char in key:
        if char not in self.letters:
          self.letters.append(char)
        self.data[-1].append(char)
        self.masks[-1].append(0)
      self.data[-1].append(cfg.DATASET_EQUAL_SYMBOL)
      self.masks[-1].append(0)
      for i, item in enumerate(buffer[key]):
        for phoneme in item:
          if phoneme not in self.phonemes:
            self.phonemes.append(phoneme)
          self.data[-1].append(phoneme)
          self.masks[-1].append(1)
        if i < len(buffer[key]) - 1:
          self.data[-1].append(cfg.DATASET_COMMA_SYMBOL)
          self.masks[-1].append(1)
    self.letters.sort()
    self.phonemes.sort()
    self.tokens += self.letters
    self.tokens += self.phonemes
    self.max_length = max(*[len(datum) for datum in self.data]) + 1
    for i in range(len(self.data)):
      pad = self.max_length - len(self.data[i])
      self.data[i] += [cfg.DATASET_END_SYMBOL] * pad
      self.masks[i] += [1] * pad
      for j in range(len(self.data[i])):
        self.data[i][j] = self.tokens.index(self.data[i][j])
    if count > 0:
      self.data = self.data[:count]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    data = self.data[index % len(self.data)]
    mask = self.masks[index % len(self.masks)]
    return (
      torch.as_tensor(data[:-1]), 
      torch.as_tensor(data[1:]),
      torch.as_tensor(mask[1:]))

  def encode(self, text):
    return [self.tokens.index(char) for char in text]

  def decode(self, tokens):
    return [self.tokens[i] for i in tokens]

def load_latest_ckpt():
  result = []
  for dirpath, _, filenames in os.walk("lightning_logs"):
    for filename in filenames:
      if filename.endswith(".ckpt") == False:
        continue
      result.append(f"{dirpath}/{filename}")
  if len(result) > 0:
    result.sort(key=lambda x: int(x.split("=")[-1].split(".")[0]))
    return result[-1]
  return None
