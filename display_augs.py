import os
import random
import numpy as np
import pandas as pd
import cv2
import torch

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from data_utils.dataset import WheatDatasetEfficientDet, collate_fn
from data_utils.augmentations import get_train_transform, get_valid_transform


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


def write_output(image, target_boxes, filename):
    thickness = 3 if image.shape[0] > 512 else 1
    image = image * 255.
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8)
    image = np.ascontiguousarray(image)
    for box in target_boxes:
        cv2.rectangle(
            image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness
        )
    if not os.path.exists("./output/augmentations/"):
        os.makedirs("./output/augmentations/")
    if filename is not None:
        cv2.imwrite("./output/augmentations/" + filename + ".jpg", image)


def display_augs(dataset_name, img_size, batch_size=4):
    loader = DataLoader(
        get_training_datasets(img_size)[dataset_name],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    # device = "cuda"
    with torch.no_grad():
        if dataset_name == "train":
            for images, targets, image_ids in loader:
                images = torch.stack(images)
                # images = images.to(device).float()
                boxes1 = [b["bbox"].type(torch.float32) for b in targets]
                labels1 = [l["cls"].type(torch.float32) for l in targets]
                img_scale1 = [
                    s["img_scale"].type(torch.float32) for s in targets
                ]
                img_size1 = [
                    sz["img_size"].type(torch.float32) for sz in targets
                ]
                t = {}
                t["bbox"] = boxes1
                t["cls"] = labels1
                t["img_scale"] = img_scale1
                t["img_size"] = img_size1
                for image, target, image_id in zip(images, targets, image_ids):
                    sample = image.permute(1, 2, 0).detach().cpu().numpy()
                    sample = (
                        sample * np.array([0.229, 0.224, 0.225])
                    ) + np.array([0.485, 0.456, 0.406])
                    sample = sample[..., ::-1].copy()
                    boxes1 = (
                        target["bbox"].detach().cpu().numpy().astype(np.int32)
                    )
                    boxes1 = boxes1[:, [1, 0, 3, 2]]
                    write_output(
                        sample,
                        boxes1,
                        image_id + '_' + str(random.uniform(0, 99999)),
                    )
                    print(image_id)
                break
        else:
            for images, targets1, image_ids in loader:
                images = torch.stack(images)
                # images = images.to(device).float()
                for image, target1, image_id in zip(
                    images, targets1, image_ids
                ):
                    sample = image.permute(1, 2, 0).detach().cpu().numpy()
                    sample = (
                        sample * np.array([0.229, 0.224, 0.225])
                    ) + np.array([0.485, 0.456, 0.406])
                    boxes1 = (
                        target1["bbox"].detach().cpu().numpy().astype(np.int32)
                    )
                    boxes1 = boxes1[:, [1, 0, 3, 2]]
                    write_output(sample, boxes1, image_id)
                    print(image_id)
                break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="train", help="train/val",
    )
    parser.add_argument(
        "--img_size", type=int, default=1024, help="image size",
    )
    parser.add_argument(
        "--img_count", type=int, default=16, help="image count",
    )
    args = vars(parser.parse_args())
    display_augs(args["dataset_name"], args["img_size"], args['img_count'])
