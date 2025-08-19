# %%
from jpsma.data import PETDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
from jpsma.model import NNUnet
from pytorch_lightning import Trainer
# %%
fold = 0
root_dir = "/data2/core-rad/data/deep-psma-preprocessed-bothways/"
case_ids = sorted(os.listdir(root_dir))
# %%
datamodule = PETDataModule(root_dir=root_dir, case_ids=case_ids, use_ct=False, batch_size=2, fold=2, crop_size=(256, 128, 128))
model = NNUnet()
logger = TensorBoardLogger(save_dir="logs", name="f2_fin")
checkpoint_cb = ModelCheckpoint(monitor="val/loss", mode="min", filename="{epoch:02d}-{val/loss:.4f}")
print("Starting training")
trainer = Trainer(
    max_epochs=500,
    check_val_every_n_epoch=5,
    num_sanity_val_steps=2,
    logger=logger,
    callbacks=[checkpoint_cb],
    accelerator="gpu",
    devices=2,
    precision="bf16-mixed",
)
# %%
trainer.fit(model, datamodule=datamodule)
