import torch
import lightning as L
from torch.nn import CrossEntropyLoss
from torchmetrics.functional import accuracy, recall, specificity, f1_score, cohen_kappa
from torchmetrics.classification import MulticlassConfusionMatrix

# from matplotlib import pyplot as plt
from model import Network
import config

import matplotlib

matplotlib.use("Agg")
from dvclive import Live
from matplotlib import pyplot as plt



class LitModel(L.LightningModule):

    def __init__(
        self,
        seed=config.SEED,
        exp=config.EXP,
        model=config.MODEL,
        dataset=config.DATASET,
        num_epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        learning_rate=config.LEARNING_RATE,
        weight=config.WEIGHT,
        in_channel=config.IN_CHANNEL,
        num_classes=config.NUM_CLASSES,
        dc_num_layers=config.DC_NUM_LAYERS,
        dc_scale=config.DC_SCALE,
        fun=config.FUN,
        task=config.TASK,
        sch_patience=config.SCH_PATIENCE,
    ):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.task = task
        self.sch_patience = sch_patience
        self.save_hyperparameters()


        self.model = Network(model)
        self.criterion = CrossEntropyLoss(weight=weight)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.criterion(preds, labels)
        acc = accuracy(preds, labels, self.task, num_labels=self.num_classes)

        tensorboard_logs = {"train_loss": loss, "train_acc": acc}
        self.training_step_outputs.append({"loss": loss, "acc": acc, "log": tensorboard_logs})
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        train_acc = torch.stack([x["acc"] for x in self.training_step_outputs]).mean()

        tensorboard_logs = {"train_loss": avg_loss, "train_acc": train_acc, "step": self.trainer.current_epoch + 1}
        self.log_dict(tensorboard_logs, prog_bar=True)

    def validation_step(self, batch, batch_idx):

        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.criterion(preds, labels)
        acc = accuracy(preds, labels, self.task, num_labels=self.num_classes)
        tensorboard_logs = {"val_loss": loss, "val_acc": acc}
        self.validation_step_outputs.append({"loss": loss, "acc": acc, "log": tensorboard_logs})

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        val_acc = torch.stack([x["acc"] for x in self.validation_step_outputs]).mean()

        tensorboard_logs = {"val_loss": avg_loss, "val_acc": val_acc, "step": self.trainer.current_epoch + 1}
        self.log_dict(tensorboard_logs, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.criterion(preds, labels)
        acc = accuracy(preds, labels, self.task, num_labels=self.num_classes)
        tensorboard_logs = {"test_loss": loss, "test_acc": acc}
        self.test_step_outputs.append(
            {"loss": loss, "acc": acc, "preds": preds, "targets": labels, "log": tensorboard_logs}
        )

    def on_test_epoch_end(self):
        preds = torch.argmax(torch.cat([x["preds"] for x in self.test_step_outputs]), dim=1).cpu()
        targets = torch.argmax(torch.cat([x["targets"] for x in self.test_step_outputs]), dim=1).cpu()

        test_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()
        test_acc = accuracy(preds, targets, "multiclass", num_classes=self.num_classes)
        test_recall = recall(preds, targets, "multiclass", num_classes=self.num_classes, average="macro")
        test_sp = specificity(preds, targets, "multiclass", num_classes=self.num_classes)
        f1 = f1_score(preds, targets, "multiclass", num_classes=self.num_classes, average="macro")
        kappa = cohen_kappa(preds, targets, "multiclass", num_classes=self.num_classes)

        confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        confusion_matrix.update(preds, targets)
        cm_fig, cm_ax = confusion_matrix.plot()

        with Live(
            dir=f"./DvcLiveLogger/Seed[{config.SEED}]_Class[{config.NUM_CLASSES}]/Exp[{config.EXP}]",
            resume=True,
            save_dvc_exp=False,
        ) as live:
            live.log_image("ConfusionMatrix.png", cm_fig)
            plt.close()
        tensorboard_logs = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_recall": test_recall,
            "test_sp": test_sp,
            "test_f1": f1,
            "test_kappa": kappa,
        }
        self.log_dict(tensorboard_logs)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=self.sch_patience, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
