import os
import torch
import lightning
import config as cfg
from model import Transformer
from utils import DictDataset, load_latest_ckpt

class Trainer(lightning.LightningModule):

  def __init__(self, token_count, encode=None, decode=None, end_token=None):
    super().__init__()
    self.transformer = Transformer(
      token_count, 
      depth=cfg.MODEL_DEPTH, 
      head=cfg.MODEL_HEAD, 
      dim=cfg.MODEL_DIM, 
      hidden_dim=cfg.MODEL_HIDDEN_DIM)
    self.loss = torch.nn.CrossEntropyLoss(reduction="none")
    self.encode = encode
    self.decode = decode
    self.end_token = end_token

  def configure_optimizers(self):
    opt = torch.optim.Muon(
      self.transformer.parameters(), 
      cfg.TRAIN_LR)
    sch = torch.optim.lr_scheduler.StepLR(
      opt, 
      cfg.TRAIN_SCHEDULER_INTERVAL, 
      cfg.TRAIN_SCHEDULER_GAMMA)
    return {
      "optimizer": opt,
      "lr_scheduler": {
        "scheduler": sch,
        "interval": "step",
        "frequency": 1}}

  def training_step(self, batch, _):
    x0, x1, mask = batch
    y = self.transformer(x0)
    loss = (self.loss(y.mT, x1) * mask).mean()
    self.log("loss", loss, True)
    return loss

  def validation_step(self, _, __):
    tokens = self.encode([
      cfg.DATASET_START_SYMBOL, 
      *list(cfg.TRAIN_VALIDATION_WORD), 
      cfg.DATASET_EQUAL_SYMBOL])
    result = self.decode(self.transformer.sample(tokens, self.end_token))
    print(f"\n{result}\n")

if __name__ == "__main__":
  dataset = DictDataset(cfg.DATASET_PATH)
  def get_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
      dataset=dataset,
      batch_size=batch_size,
      shuffle=True,
      drop_last=True,
      persistent_workers=True,
      num_workers=4)
  model = Trainer(
    len(dataset.tokens), 
    dataset.encode, 
    dataset.decode,
    dataset.tokens.index(cfg.DATASET_END_SYMBOL))
  trainer = lightning.Trainer(
    check_val_every_n_epoch=None,
    val_check_interval=cfg.TRAIN_VALIDATION_INTERVAL,
    max_epochs=cfg.TRAIN_EPOCH)
  trainer.fit(
    model=model,
    train_dataloaders=get_dataloader(dataset, cfg.TRAIN_BATCH_SIZE),
    val_dataloaders=get_dataloader(DictDataset(cfg.DATASET_PATH, 1), 1),
    ckpt_path=load_latest_ckpt())
  os.makedirs("outputs", exist_ok=True)
  torch.save(model.transformer.state_dict(), "outputs/model.pt")
