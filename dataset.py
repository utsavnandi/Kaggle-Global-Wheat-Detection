import random
import numpy as np
import numba
from numba import jit
import cv2
import torch
from torch.utils.data import Dataset

DIR_INPUT = "./data"
DIR_TRAIN = f"{DIR_INPUT}/train"
DIR_TEST = f"{DIR_INPUT}/test"


class WheatDatasetFasterRCNN(Dataset):
    def __init__(self, dataframe, image_dir, isValid=True, transforms=None):
        super().__init__()
        self.image_ids = dataframe["image_id"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.isValid = isValid

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index: int):

        if self.isValid:
            image, boxes = self.load(index)
        else:
            rng = np.random.random()
            if rng < 0.15:
                image, boxes = self.load(index)
            elif rng < 0.85:
                image, boxes = self.cutmix(index)
            else:
                image, boxes = self.mixup(index)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.ones((boxes.shape[0]), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index], dtype=torch.float32)
        target["area"] = area

        if self.transforms:
            sample = self.transforms(
                **{"image": image, "bboxes": target["boxes"], "labels": labels}
            )
            image = sample["image"]
            target["labels"] = target["labels"].type(torch.int64)
            target["boxes"] = torch.as_tensor(
                sample["bboxes"], dtype=torch.float32
            )
        return image, target

    def load(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df["image_id"] == image_id]
        image = cv2.imread(
            f"{self.image_dir}/{image_id}.jpg", cv2.IMREAD_COLOR
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        boxes = records[["x", "y", "w", "h"]].values.astype(np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    def mixup(self, index: int):
        image, boxes = self.load(index)
        rand_index = self.get_rand_index()
        image_2, boxes_2 = self.load(rand_index)
        # alpha = 5
        # lam = np.random.beta(alpha, alpha)
        lam = 0.5
        mixup_image = image * lam + image_2 * (1 - lam)
        mixup_boxes = np.concatenate((boxes, boxes_2), axis=0)
        return mixup_image.astype(np.uint8), mixup_boxes

    def cutmix(self, index: int):
        image_1, boxes_1 = self.load(index)
        rand_index = self.get_rand_index()
        image_2, boxes_2 = self.load(rand_index)
        imsize = image_1.shape[0]

        # create two random points
        x1, y1 = [
            int(random.uniform(imsize * 0.0, imsize * 0.4)) for _ in range(2)
        ]
        x2, y2 = [
            int(random.uniform(imsize * 0.6, imsize * 1.0)) for _ in range(2)
        ]

        # clip the random image
        mixup_image = image_1.copy()
        mixup_target = boxes_2
        mixup_target[:, [0, 2]] = mixup_target[:, [0, 2]].clip(min=x1, max=x2)
        mixup_target[:, [1, 3]] = mixup_target[:, [1, 3]].clip(min=y1, max=y2)

        # remove cutout bboxes from first image
        boxes_1 = np.delete(
            boxes_1,
            np.where(
                (boxes_1[:, 0] > x1)
                & (boxes_1[:, 2] < x2)
                & (boxes_1[:, 1] > y1)
                & (boxes_1[:, 3] < y2)
            ),
            0,
        )

        # remove boxes with low visibility
        cutout_ords = np.array([x1, y1, x2, y2])
        boxes_1 = filter_boxes(boxes_1, cutout_ords)
        boxes_1 = boxes_1[
            np.where(
                (boxes_1[:, 2] - boxes_1[:, 0])
                * (boxes_1[:, 3] - boxes_1[:, 1])
                > 64
            )
        ]

        # remove any bbox with area less than 64
        mixup_target = mixup_target.astype(np.int32)
        mixup_target = mixup_target[
            np.where(
                (mixup_target[:, 2] - mixup_target[:, 0])
                * (mixup_target[:, 3] - mixup_target[:, 1])
                > 64
            )
        ]
        # mixup
        mixup_target = np.concatenate((boxes_1, mixup_target))
        mixup_image[y1:y2, x1:x2] = image_2[y1:y2, x1:x2]

        return mixup_image, mixup_target

    def get_rand_index(self):
        rand_index = np.random.choice([*range(0, self.image_ids.shape[0])])
        return rand_index


@jit(nopython=True)
def filter_boxes(boxes_1, cutout_ords) -> numba.float32[:, :]:
    boxes_1_copy = np.zeros(boxes_1.shape)
    for i in range(boxes_1.shape[0]):
        vis = calculate_intersection(boxes_1[i], cutout_ords)
        if vis < 0.3:
            boxes_1_copy[i] = boxes_1[i]
    return boxes_1_copy


@jit(nopython=True)
def calculate_intersection(gt, pr) -> float:
    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0])  # + 1
    if dx < 0:
        return 0.0
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1])  # + 1
    if dy < 0:
        return 0.0
    overlap_area = dx * dy
    area_smaller = (gt[3] - gt[1]) * (gt[2] - gt[0])
    vis = overlap_area / area_smaller
    return vis


def collate_fn(batch):
    return tuple(zip(*batch))
