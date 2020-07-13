import os
import time
import datetime
import gc
import random
import numpy as np
import pandas as pd

import cv2
import re
from tqdm.notebook import tqdm

import torch
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler

from ranger import Ranger

from getpass import getpass
# from google.colab import auth
# from google.cloud import storage

# import neptune

from dataset import WheatDatasetEfficientDet, collate_fn
from metrics import calculate_image_precision, iou_thresholds
from models import get_train_model
from augmentations import get_train_transform, get_valid_transform


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_training_datasets(train_df, valid_df):
    datasets = {}
    datasets["train"] = WheatDatasetEfficientDet(
        train_df, DIR_TRAIN, False, get_train_transform()
    )
    datasets["valid"] = WheatDatasetEfficientDet(
        valid_df, DIR_TRAIN, True, get_valid_transform()
    )
    return datasets


def train_one_epoch(
    loader, model, optimizer, device, scaler, scheduler=None, log=False
):
    model.train()
    running_loss = 0.0
    batch_loss = 0.0
    optimizer.zero_grad()
    for i, (images, targets) in tqdm(enumerate(loader), total=len(loader)):
        images = torch.stack(images)
        images = images.to(device).float()
        boxes = [b["bbox"].to(device, dtype=torch.float32) for b in targets]
        labels = [l["cls"].to(device, dtype=torch.float32) for l in targets]

        img_size = torch.ones((2,), dtype=torch.float32)
        img_size = img_size.new_full(
            (images.shape[0], 2), float(IMG_SIZE), device=device
        )
        img_scale = torch.tensor(
            [IMG_SIZE / 1024] * images.shape[0],  # use 1
            dtype=torch.float32,
            device=device,
        )

        t = {}
        t["bbox"] = boxes
        t["cls"] = labels
        t["img_scale"] = img_scale
        t["img_size"] = img_size

        # lam = -1
        # if np.random.random() < 0.2:
        #    images, targets1, targets2, lam = mixup(images, t, alpha=5)
        #    loss_dict1 = model(images, targets1)
        #    loss_dict2 = model(images, targets2)
        #    losses1 = loss_dict1['loss'] / FLAGS['accumulation_steps']
        #    losses2 = loss_dict2['loss'] / FLAGS['accumulation_steps']
        #    total_loss = lam * losses1 + (1-lam) * losses2
        # else:
        loss_dict = model(images, t)
        losses = loss_dict["loss"] / FLAGS["accumulation_steps"]
        total_loss = losses

        loss_value = total_loss.item()
        running_loss += float(loss_value)
        batch_loss += float(loss_value)

        if scaler:
            scaler.scale(total_loss).backward()
        else:
            (total_loss).backward()

        if (i + 1) % FLAGS["accumulation_steps"] == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            if scheduler:
                scheduler.step()
            if log:
                neptune.log_metric("batch_loss/train", batch_loss)
            batch_loss = 0.0
            #     neptune.log_metric('class_loss/train', float(lam * loss_dict1.get('class_loss').item() + (1-lam) * loss_dict2.get('loss').item()))
            #     neptune.log_metric('box_loss/train', float(lam * loss_dict1.get('box_loss').item() + (1-lam) * loss_dict2.get('loss').item()))
            # else:
            #     neptune.log_metric('loss/train', float(loss_dict.get('loss').item()))
            #     neptune.log_metric('class_loss/train', float(loss_dict.get('class_loss').item()))
            #     neptune.log_metric('box_loss/train', float(loss_dict.get('box_loss').item()))

    return running_loss / len(loader)


def val_one_epoch(model, loader):
    model.eval()
    running_loss = 0.0
    validation_image_precisions = []
    with torch.no_grad():
        for idx, (images, targets) in tqdm(
            enumerate(loader), total=len(loader)
        ):
            images = torch.stack(images)
            images = images.to(device).float()
            gt_boxes = np.array(
                list(target["bbox"].data.cpu().numpy() for target in targets)
            )
            gt_labels = np.array(
                list(target["cls"].data.cpu().numpy() for target in targets)
            )
            boxes = [
                b["bbox"].to(device, dtype=torch.float32) for b in targets
            ]
            labels = [
                l["cls"].to(device, dtype=torch.float32) for l in targets
            ]

            t = {}
            t["bbox"] = boxes
            t["cls"] = labels

            img_scales = torch.tensor(
                [1] * images.shape[0], dtype=torch.float32, device=device
            )
            img_size = torch.ones((2,), dtype=torch.float32)
            img_size = img_size.new_full(
                (images.shape[0], 2), float(IMG_SIZE)
            ).to(device)
            t["img_scale"] = img_scales
            t["img_size"] = img_size
            outputs = model(images, t)
            running_loss += float(outputs["loss"])

            # ['loss', 'class_loss', 'box_loss', 'detections']
            for i in range(images.shape[0]):
                pred_boxes = (
                    outputs.get("detections")[i].detach().cpu().numpy()[:, :4]
                )
                scores = (
                    outputs.get("detections")[i].detach().cpu().numpy()[:, 4]
                )
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

    return (
        np.array(validation_image_precisions).mean(),
        running_loss / len(loader),
    )


def upload_blob(
    source_file_name, destination_blob_name,
):
    """Uploads a file to the bucket."""
    global bucket_name, project_id
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    dt_now = datetime.datetime.now().strftime("%d_%B")
    destination_blob_name = (
        "global-wheat-detection/" + dt_now + "/" + destination_blob_name
    )
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
    print("uploaded blob name: ", destination_blob_name)


def save_upload(
    bench, optimizer, epoch, scheduler, val_metric, exp_name, fold=None
):
    if fold is not None:
        NAME = (
            exp_name
            + f"_fold_{fold}_{str(epoch+1)}_map_"
            + str(round(val_metric, 4))
            + ".ckpt"
        )

    NAME = (
        exp_name
        + f"_{str(epoch+1)}_map_"
        + str(round(val_metric, 5))
        + ".ckpt"
    )
    MODEL_PATH = NAME
    torch.save({"model_state_dict": bench.model.state_dict(),}, MODEL_PATH)
    print(f"Saved ckpt for epoch {epoch+1}")
    # upload_blob(MODEL_PATH, NAME)
    # print(f"Uploaded ckpt for epoch {epoch+1}")


def train_job(model_name, train_df, valid_df, model_ckpt=None, log=True):

    if log:
        neptune.set_project("utsav/wheat-det")
        neptune.init("utsav/wheat-det", api_token=NEPTUNE_API_TOKEN)
        neptune.create_experiment(
            FLAGS["exp_name"],
            exp_description,
            params=FLAGS,
            upload_source_files="*.txt",
        )
    best_score = 0.0
    start_epoch = 0

    datasets = get_training_datasets(train_df, valid_df)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=FLAGS["batch_size"],
        num_workers=FLAGS["num_workers"],
        shuffle=True,  # sampler=sampler, #
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        datasets["valid"],
        batch_size=FLAGS["batch_size"] * 2,
        shuffle=False,
        num_workers=FLAGS["num_workers"],
        collate_fn=collate_fn,
    )

    if model_ckpt is not None:
        model = get_train_model(model_name, model_ckpt)
    else:
        model = get_train_model(model_name)
    model.to(device)

    optimizer = Ranger(
        model.parameters(),
        lr=FLAGS["learning_rate"],
        alpha=0.5,
        k=6,
        N_sma_threshhold=5,
        weight_decay=FLAGS["weight_decay"],
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.5,
        verbose=True,
        patience=FLAGS["scheduler_pat"],
    )

    scaler = GradScaler()
    es = 0
    for epoch in range(start_epoch, FLAGS["num_epochs"]):

        print("-" * 27 + f"Epoch #{epoch+1} started" + "-" * 27)

        train_loss = train_one_epoch(
            train_loader,
            model,
            optimizer,
            device,
            scaler,
            scheduler=None,
            log=log,
        )
        print(f"Average loss for epoch #{epoch+1} : {train_loss}")

        val_metric, val_loss = val_one_epoch(model, val_loader)
        scheduler.step(val_loss)

        print(f"metric/val : {val_metric}")
        print(f"loss/val : {val_loss}")

        if log:
            neptune.log_metric("metric/val", val_metric)
            neptune.log_metric("loss/val", val_loss)

        # if epoch==FLAGS['unfreeze_epoch']:
        #    model = unfreeze_all_layers(model)

        if (val_metric > best_score) or (best_score - val_metric < 0.01):
            es = 0
            if val_metric > best_score:
                best_score = val_metric
            if epoch > 9:
                save_upload(
                    model,
                    optimizer,
                    epoch,
                    scheduler,
                    val_metric,
                    exp_name=FLAGS["exp_name"],
                )
        # else:
        #    if epoch>24:
        #        es+=1
        # if es > FLAGS['early_stop_count']:
        #    print('Early stopping...')
        #    break

        print("-" * 28 + f"Epoch #{epoch+1} ended" + "-" * 28)

    neptune.stop()


FLAGS = {}
FLAGS["num_workers"] = 4
FLAGS["batch_size"] = 2
FLAGS["accumulation_steps"] = 16
FLAGS["learning_rate"] = 2e-3
FLAGS["num_epochs"] = 60
FLAGS["weight_decay"] = 3e-3
FLAGS["exp_name"] = "tf_efficientdet_d0"
FLAGS["fold"] = "0, 3, 5"
# FLAGS['early_stop_count'] = 9
FLAGS["scheduler_pat"] = 7
exp_description = """
tf_efficientdet_d0,
Ranger, ReduceLROnPlateau
New Aug + Cutmix,
imsize 512
"""
seed_everything(43)

DIR_INPUT = "./data"
DIR_TRAIN = f"{DIR_INPUT}/train"
DIR_TEST = f"{DIR_INPUT}/test"


device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

train_df = pd.read_csv("./train_df.csv")
valid_df = pd.read_csv("./valid_df.csv")

NEPTUNE_API_TOKEN = getpass(prompt="Enter neptune api token: ")
project_id = getpass(prompt="enter gcs project id: ")
bucket_name = getpass(prompt="enter gcs bucketname: ")

if NEPTUNE_API_TOKEN != "":
    os.environ["NEPTUNE_API_TOKEN"] = NEPTUNE_API_TOKEN
    log = True
else:
    log = False

# auth.authenticate_user()


train_job("tf_efficientdet_d5", train_df, valid_df, log=False)
