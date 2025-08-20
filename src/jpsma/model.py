import pytorch_lightning as pl
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import DynUNet
from torchmetrics import Metric


class FPFN(Metric):
    """Counts False Positives (FP) and False Negatives (FN) for binary classification."""

    full_state_update = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.long()
        target = target.long()

        fp = ((preds == 1) & (target == 0)).sum()
        fn = ((preds == 0) & (target == 1)).sum()

        self.fp += fp
        self.fn += fn

    def compute(self):
        return {"FP": self.fp, "FN": self.fn}


class NNUnet(pl.LightningModule):
    def __init__(self, learning_rate: float = 2e-4, sw_batch_size: int = 4):
        super().__init__()
        self.patch_size = (256, 128, 128)
        self.sw_mode = "gaussian"
        self.sw_overlap = 0.5
        self.sw_batch_size = sw_batch_size
        self.learning_rate = learning_rate
        self.deep_supervision = True

        self.backbone = DynUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=2,
            filters=[32, 64, 128, 256, 320, 320],
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]],
            upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]],
            norm_name="instance",
            act_name=("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
            res_block=True,
            # dropout=0.1,
            deep_supervision=self.deep_supervision,
            deep_supr_num=3,
        )

        self.loss_fn = DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        self.dice_fdg = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, ignore_empty=True)
        self.dice_psma = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, ignore_empty=True)
        self.fdg_confusion = FPFN()
        self.psma_confusion = FPFN()

    def compute_loss(self, prediction, label):
        if self.deep_supervision:
            loss, weights = 0.0, 0.0
            for i in range(prediction.shape[1]):
                loss += self.loss_fn(prediction[:, i], label) * 0.5**i
                weights += 0.5**i
            return loss / weights
        return self.loss_fn(prediction, label)

    def forward(self, x):
        return self.sliding_window_inference(x)

    def sliding_window_inference(self, image):
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.backbone,
            overlap=self.sw_overlap,
            mode=self.sw_mode,
        )

    def training_step(self, batch, batch_idx):
        volume = torch.cat(
            [
                batch["FDG_PET"][:, None, ...],
                batch["FDG_TOTSEG"][:, None, ...],
                batch["PSMA_PET"][:, None, ...],
                batch["PSMA_TOTSEG"][:, None, ...],
            ],
            dim=1,
        )

        label = torch.cat(
            [
                batch["FDG_TTB"][:, None, ...],
                batch["PSMA_TTB"][:, None, ...],
            ],
            dim=1,
        )

        logits = self.backbone(volume)
        loss = self.compute_loss(logits, label)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        volume = torch.cat(
            [
                batch["FDG_PET"][:, None, ...],
                batch["FDG_TOTSEG"][:, None, ...],
                batch["PSMA_PET"][:, None, ...],
                batch["PSMA_TOTSEG"][:, None, ...],
            ],
            dim=1,
        )

        label = torch.cat(
            [
                batch["FDG_TTB"][:, None, ...],
                batch["PSMA_TTB"][:, None, ...],
            ],
            dim=1,
        )

        logits = self.forward(volume)  # [1, 2, D, H, W]

        # PET-intensity masks
        fdg_mask = batch["FDG_PET"][:, None, ...] >= 0
        psma_mask = batch["PSMA_PET"][:, None, ...] >= 0
        mask = torch.cat([fdg_mask, psma_mask], dim=1)

        loss = self.loss_fn(logits, label)
        prediction = torch.ge(torch.sigmoid(logits), 0.5)
        prediction = prediction * mask.float()

        # case_id = str(batch["case_id"][0])
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)

        self.dice_fdg(y_pred=prediction[:, 0:1, ...], y=label[:, 0:1, ...])
        self.dice_psma(y_pred=prediction[:, 1:2, ...], y=label[:, 1:2, ...])

        self.fdg_confusion.update(prediction[:, 0:1, ...], label[:, 0:1, ...])
        self.psma_confusion.update(prediction[:, 1:2, ...], label[:, 1:2, ...])

        return loss

    def on_validation_epoch_end(self):
        dice_fdg = self.dice_fdg.aggregate().item()
        dice_psma = self.dice_psma.aggregate().item()
        self.log("val/dice_fdg", dice_fdg, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        self.log("val/dice_psma", dice_psma, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        self.dice_fdg.reset()
        self.dice_psma.reset()

        res = self.psma_confusion.compute()
        self.log("val/FP_psma", res["FP"].item(), on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        self.log("val/FN_psma", res["FN"].item(), on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        self.psma_confusion.reset()

        res = self.fdg_confusion.compute()
        self.log("val/FP_fdg", res["FP"].item(), on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        self.log("val/FN_fdg", res["FN"].item(), on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        self.fdg_confusion.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.trainer.max_epochs, power=0.9)
        return [optimizer], [scheduler]
