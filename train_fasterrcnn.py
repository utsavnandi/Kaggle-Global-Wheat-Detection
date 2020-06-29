import gc
import os
import re
import time
import datetime
import random
import cv2
import numpy as np
import pandas as pd

from typing import List, Union, Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler


from ranger import Ranger

import neptune

from getpass import getpass
from google.colab import auth
from google.cloud import storage

from dataset import WheatDatasetFasterRCNN, collate_fn
from metrics import calculate_image_precision
from models import get_model
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
    datasets["train"] = WheatDatasetFasterRCNN(
        train_df, DIR_TRAIN, False, transforms=get_train_transform()
    )
    datasets["valid"] = WheatDatasetFasterRCNN(
        valid_df, DIR_TRAIN, True, transforms=get_valid_transform()
    )
    return datasets


def train_one_epoch(
    model, loader, optimizer, scaler, device, scheduler=None, log=False
):

    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for i, (images, targets) in tqdm(enumerate(loader), total=len(loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        try:
            loss_dict = model(images, targets)
        except:
            tqdm.write("error detected...")
            continue
        losses = sum(
            loss / FLAGS["accumulation_steps"] for loss in loss_dict.values()
        )
        loss_value = losses.item()
        running_loss += float(loss_value)
        if scaler:
            scaler.scale(losses).backward()
        else:
            losses.backward()

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
            for key in loss_dict:
                neptune.log_metric(
                    f"{key}/train",
                    float(loss_dict[key].item()) / FLAGS["accumulation_steps"],
                )

    return running_loss / len(loader)


def val_one_epoch(model, loader):
    model.eval()
    running_loss = 0.0
    validation_image_precisions = []
    with torch.no_grad():
        for images, targets in tqdm(loader, total=len(loader)):
            images = list(image.to(device) for image in images)
            gt_boxes = np.array(
                [target["boxes"].numpy() for target in targets]
            )
            gt_labels = np.array(
                [target["labels"].numpy() for target in targets]
            )
            try:
                outputs = model(images)
            except:
                continue
            for i in range(len(images)):
                pred_boxes = (
                    outputs[i]
                    .get("boxes")
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
                scores = outputs[i].get("scores").detach().cpu().numpy()
                indexes = np.where(scores > 0.55)[0]
                target_boxes = gt_boxes[i].astype(np.int32)
                try:
                    image_precision = calculate_image_precision(
                        target_boxes,
                        pred_boxes[indexes],
                        thresholds=iou_thresholds,
                        form="pascal_voc",
                    )
                except:
                    print("error in score calculation: ")
                    print("target_boxes", target_boxes)
                    print("pred_boxes", pred_boxes[indexes])
                    continue
                validation_image_precisions.append(image_precision)

    return np.array(validation_image_precisions).mean()


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
    model, optimizer, epoch, scheduler, val_metric, exp_name, fold=None
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
        + str(round(val_metric, 4))
        + ".ckpt"
    )
    MODEL_PATH = NAME
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch,}, MODEL_PATH
    )
    print(f"Saved ckpt for epoch {epoch+1}")
    upload_blob(MODEL_PATH, NAME)
    print(f"Uploaded ckpt for epoch {epoch+1}")


device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
FLAGS = {}
FLAGS["num_workers"] = 4
FLAGS["batch_size"] = 2
FLAGS["accumulation_steps"] = 4
FLAGS["learning_rate"] = 1e-4
FLAGS["num_epochs"] = 60
FLAGS["weight_decay"] = 5e-3
FLAGS["exp_name"] = "gluon_resnext101_32x4d_fasterrcnn"
FLAGS["fold"] = "0, 1, 2, 3"
FLAGS["trainable_layers"] = 5
FLAGS["unfreeze_epoch"] = 20
exp_description = """
gluon_resnext101_32x4d_fasterrcnn,
Ranger, ReduceLROnPlateau
imsize 1024
"""


def train_job(
    model_name, detector_name, train_df, valid_df, model_ckpt=None, log=True
):
    # exp_name = FLAGS["exp_name"]
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
        shuffle=True,  # shuffle=True,  # sampler=sampler
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["valid"],
        batch_size=FLAGS["batch_size"] * 4,
        shuffle=False,
        num_workers=FLAGS["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if model_ckpt is None:
        model = get_model(model_name, detector_name, FLAGS["trainable_layers"])
    else:
        model = get_model(
            model_name, detector_name, FLAGS["trainable_layers"], model_ckpt
        )
        start_epoch = torch.load(model_ckpt)["epoch"]
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
        optimizer, "min", factor=0.75, verbose=True, patience=4
    )

    es = 0
    scaler = GradScaler()
    for epoch in range(start_epoch, FLAGS["num_epochs"]):

        print("-" * 27 + f"Epoch #{epoch+1} started" + "-" * 27)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            scheduler=None,
            log=log,
        )
        print(f"Average loss for epoch #{epoch+1} : {train_loss}")

        val_metric = val_one_epoch(model, val_loader)
        scheduler.step(-val_metric)
        print(f"metric/val : {val_metric}")

        if log:
            neptune.log_metric("metric/val", val_metric)

        if (val_metric > best_score) or (best_score - val_metric < 0.01):
            es = 0
            if val_metric > best_score:
                best_score = val_metric
            if epoch > 4:
                save_upload(
                    model,
                    optimizer,
                    epoch,
                    scheduler,
                    val_metric,
                    exp_name=FLAGS["exp_name"],
                )
        else:
            es += 1
            print("early stop counter: ", es)
            if es > 9 and epoch > 19:
                print("early stopping...")
                break

        print("-" * 28 + f"Epoch #{epoch+1} ended" + "-" * 28)

    neptune.stop()


seed_everything(43)

NEPTUNE_API_TOKEN = getpass(prompt="Enter neptune api token: ")
project_id = getpass(prompt="enter gcs project id: ")
bucket_name = getpass(prompt="enter gcs bucketname: ")

if NEPTUNE_API_TOKEN != "":
    os.environ["NEPTUNE_API_TOKEN"] = NEPTUNE_API_TOKEN
    log = True
else:
    log = False

auth.authenticate_user()

DIR_INPUT = "./data"
DIR_TRAIN = f"{DIR_INPUT}/train"
DIR_TEST = f"{DIR_INPUT}/test"
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

train_df = pd.read_csv("./train_df.csv")
valid_df = pd.read_csv("./valid_df.csv")

train_job(
    "gluon_resnext101_32x4d", "fasterrcnn", train_df, valid_df, log=False
)
