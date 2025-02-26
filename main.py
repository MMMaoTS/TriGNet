import lightning as L
from dvclive.lightning import DVCLiveLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from dataloadder import loader
from lit import LitModel
import config

import warnings

warnings.filterwarnings("ignore")


def main():
    L.seed_everything(seed=config.SEED)
    train, val, test = loader(config.BATCH_SIZE, config.NUM_WORKERS)
    model = LitModel()
    exp_dir = f"Seed[{config.SEED}]_Class[{config.NUM_CLASSES}]/Exp[{config.EXP}]"
    logger = DVCLiveLogger(
        dir=f"./DvcLiveLogger/{exp_dir}_test",
        run_name=exp_dir,
        dvcyaml=f"./DvcLiveLogger/{exp_dir}/dvc.yaml",
        save_dvc_exp=False,
    )

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, verbose=True),
        EarlyStopping(monitor="val_loss", patience=100),
    ]
    trainer = L.Trainer(max_epochs=config.NUM_EPOCHS, logger=logger, callbacks=callbacks, precision="16-mixed")
    trainer.fit(model, train, val)
    trainer.test(model, test, ckpt_path="best")

if __name__ == "__main__":
    main()
