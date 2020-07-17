import numpy as np
import pandas as pd
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from ranger import Ranger
from argparse import ArgumentParser
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_utils.dataset import WheatDatasetEfficientDet, collate_fn
from data_utils.augmentations import get_train_transform, get_valid_transform
from metrics.kaggle_metric import calculate_image_precision, iou_thresholds
from net.models import get_train_model

DIR_INPUT = "./data"
DIR_TRAIN = f"{DIR_INPUT}/train"
DIR_TEST = f"{DIR_INPUT}/test"
train_df = pd.read_csv("./train_df.csv")
valid_df = pd.read_csv("./valid_df.csv")


def get_training_datasets(img_size):
    global train_df, valid_df, DIR_TRAIN
    datasets = {}
    datasets["train"] = WheatDatasetEfficientDet(
        train_df, img_size, DIR_TRAIN, False, get_train_transform(img_size)
    )
    datasets["valid"] = WheatDatasetEfficientDet(
        valid_df, img_size, DIR_TRAIN, True, get_valid_transform(img_size)
    )
    return datasets


class LightningWheat(LightningModule):
    def __init__(self, model_name=None, hparams=None):
        super(LightningWheat, self).__init__()
        self.hparams = hparams
        self.model = get_train_model(
            model_name, self.hparams["img_size"], useGN=self.hparams["use_gn"]
        )

    def forward(self, images, targets):
        return self.model(images, targets)

    def prepare_data(self):
        datasets = get_training_datasets(self.hparams["img_size"])
        self.train_dataset = datasets["train"]
        self.valid_dataset = datasets["valid"]

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            collate_fn=collate_fn,
            # pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            collate_fn=collate_fn,
            # pin_memory=True
        )
        return valid_loader

    def configure_optimizers(self):
        optimizer = Ranger(
            self.model.parameters(),
            lr=self.hparams["learning_rate"],
            alpha=0.5,
            k=6,
            N_sma_threshhold=5,
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.5,
            verbose=True,
            patience=self.hparams["scheduler_pat"],
        )

        return [optimizer], [{"scheduler": scheduler}]

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        images = torch.stack(images)
        images = images.float()
        boxes = [box["bbox"].type(torch.float32) for box in targets]
        labels = [label["cls"].type(torch.float32) for label in targets]
        img_sizes = [
            img_size["img_size"].type(torch.float32) for img_size in targets
        ]
        img_scales = [
            img_scale["img_scale"].type(torch.float32) for img_scale in targets
        ]
        targets = {}
        targets["bbox"] = boxes
        targets["cls"] = labels
        targets["img_scale"] = img_scales
        targets["img_size"] = img_sizes
        loss_dict = self.model(images, targets)
        losses = loss_dict["loss"]
        tensorboard_logs = {"train_loss": loss_dict["loss"]}
        return {
            "loss": losses,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        images = torch.stack(images).float()
        gt_boxes = np.array(
            list(target["bbox"].data.cpu().numpy() for target in targets)
        )
        boxes = [box["bbox"].type(torch.float32) for box in targets]
        labels = [label["cls"].type(torch.float32) for label in targets]
        img_scales = torch.tensor([1] * images.shape[0], dtype=torch.float32)
        img_scales = img_scales.type_as(boxes[0])
        img_sizes = torch.ones((2,), dtype=torch.float32, device=self.device)
        img_sizes = img_sizes.new_full(
            (images.shape[0], 2), float(self.hparams["img_size"])
        )
        img_sizes = img_sizes.type_as(boxes[0])
        targets = {}
        targets["bbox"] = boxes
        targets["cls"] = labels
        targets["img_scale"] = img_scales
        targets["img_size"] = img_sizes
        outputs = self.model(images, targets)
        loss = float(outputs["loss"])
        validation_image_precisions = []
        for i in range(images.shape[0]):
            pred_boxes = (
                outputs.get("detections")[i].detach().cpu().numpy()[:, :4]
            )
            scores = outputs.get("detections")[i].detach().cpu().numpy()[:, 4]
            indexes = np.where(scores > 0.4)[0]
            pred_boxes = pred_boxes.astype(np.int32)
            # xywh (coco)-> xyxy (pascal_voc)= x,y,x+w,y+h
            pred_boxes[:, 2] = pred_boxes[:, 2] + pred_boxes[:, 0]
            pred_boxes[:, 3] = pred_boxes[:, 3] + pred_boxes[:, 1]
            target_boxes = gt_boxes[i].astype(np.int32)
            target_boxes = target_boxes[:, [1, 0, 3, 2]]
            image_precision = calculate_image_precision(
                target_boxes,
                pred_boxes[indexes],
                thresholds=iou_thresholds,
                form="pascal_voc",
            )
            validation_image_precisions.append(image_precision)
        score = np.array(validation_image_precisions).mean()
        return {"val_loss": loss, "score": score}

    def validation_epoch_end(self, outputs):
        avg_loss = np.array([x["val_loss"] for x in outputs]).mean()
        avg_score = np.array([x["score"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss, "avg_score": avg_score}
        return {
            "avg_val_loss": avg_loss,
            "avg_score": avg_score,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }


def main(args):
    dict_args = vars(args)
    FLAGS = {}
    FLAGS["num_workers"] = dict_args["num_workers"]
    FLAGS["batch_size"] = dict_args["batch_size"]
    FLAGS["accumulation_steps"] = dict_args["acc_steps"]
    FLAGS["learning_rate"] = dict_args["lr"]
    FLAGS["weight_decay"] = dict_args["weight_decay"]
    FLAGS["num_epochs"] = dict_args["num_epochs"]
    FLAGS["exp_name"] = dict_args["model_name"]
    FLAGS["fold"] = dict_args["folds"]  # "0, 1, 2, 3"
    FLAGS["scheduler_pat"] = dict_args["scheduler_patience"]
    FLAGS["img_size"] = dict_args["img_size"]
    FLAGS["use_gn"] = dict_args["use_gn"]

    model = LightningWheat(model_name=dict_args["model_name"], hparams=FLAGS)
    checkpoint_callback = ModelCheckpoint(
        filepath="./model_{epoch}-{avg_score:.5f}",
        monitor="avg_score",
        mode="max",
        save_last=True,
        save_weights_only=True,
    )
    tb_logger = TensorBoardLogger(save_dir="./lightning_logs")
    trainer = Trainer(
        gpus=1,
        deterministic=True,
        logger=[tb_logger],
        max_epochs=125,
        accumulate_grad_batches=FLAGS["accumulation_steps"],
        weights_summary="top",
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--model_name",
        type=str,
        default="tf_efficientdet_d1",
        help="format: tf_efficientdet_d[x] where x isin [0:7]",
    )
    parser.add_argument(
        "--use_gn",
        type=bool,
        default=True,
        help="use GroupNorm instead of BatchNorm",
    )
    parser.add_argument(
        "--img_size", type=int, default=1024, help="train image size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers in dataloader",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="batch size",
    )
    parser.add_argument(
        "--acc_steps",
        type=int,
        default=16,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--lr", type=int, default=0.22, help="learning rate",
    )
    parser.add_argument(
        "--weight_decay", type=int, default=3e-3, help="weight decay",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=125, help="number of epochs",
    )
    parser.add_argument(
        "--scheduler_patience", type=int, default=7, help="scheduler patience",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0, 1, 2, 3",
        help="folds being used to train (for logging)",
    )
    parser.add_argument(
        "--log_neptune",
        type=str,
        default='none',
        help="key for logging to Neptune",
    )
    args = parser.parse_args()
    main(args)
