import numpy as np
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

IMG_SIZE = 320


def get_training_datasets(train_df, valid_df, DIR_TRAIN):
    datasets = {}
    datasets["train"] = WheatDatasetEfficientDet(
        train_df, DIR_TRAIN, False, get_train_transform()
    )
    datasets["valid"] = WheatDatasetEfficientDet(
        valid_df, DIR_TRAIN, True, get_valid_transform()
    )
    return datasets


class LightningWheat(LightningModule):
    def __init__(self, model_name=None, hparams=None):
        super(LightningWheat, self).__init__()
        self.hparams = hparams
        self.model = get_train_model(model_name)

    def forward(self, images, targets):
        return self.model(images, targets)

    def prepare_data(self):
        datasets = get_training_datasets()
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
        t = {}
        t["bbox"] = boxes
        t["cls"] = labels
        t["img_scale"] = img_scales
        t["img_size"] = img_sizes
        loss_dict = self.model(images, t)
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
        img_sizes = img_sizes.new_full((images.shape[0], 2), float(IMG_SIZE))
        img_sizes = img_sizes.type_as(boxes[0])
        t = {}
        t["bbox"] = boxes
        t["cls"] = labels
        t["img_scale"] = img_scales
        t["img_size"] = img_sizes
        outputs = self.model(images, t)
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
    FLAGS["num_workers"] = 4
    FLAGS["batch_size"] = 2
    FLAGS["accumulation_steps"] = 16
    FLAGS["learning_rate"] = 2e-3
    FLAGS["weight_decay"] = 3e-3
    FLAGS["num_epochs"] = 125
    FLAGS["exp_name"] = dict_args["model_name"]
    FLAGS["fold"] = "0, 1, 2, 3"
    FLAGS["scheduler_pat"] = 7

    model = LightningWheat(model_name=dict_args["model_name"], hparams=FLAGS)
    checkpoint_callback = ModelCheckpoint(
        filepath='./{epoch}-{avg_score:.5f}',
        monitor='avg_score',
        mode='max',
    )
    tb_logger = TensorBoardLogger(save_dir="./lightning_logs")
    trainer = Trainer(
        logger=[tb_logger],
        checkpoint_callback=checkpoint_callback,
        gpus=1,
        max_epochs=125,
        weights_summary="top",
        accumulate_grad_batches=4,
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument(
        "--model_name",
        type=str,
        default="tf_efficientdet_d1",
        help="format: tf_efficientdet_d[x] where x isin [0:7]",
    )

    args = parser.parse_args()

    # train
    main(args)
