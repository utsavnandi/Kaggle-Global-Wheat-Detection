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

IMG_SIZE = 320


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


class WheatDatasetEfficientDet(Dataset):
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
            image, boxes, image_id = self.load_image(index)
        else:
            rng = np.random.random()
            if rng < 0.33:
                image, boxes, image_id = self.load_image(index)
            elif 0.33 < rng < 0.66:
                image, boxes, image_id = self.load_image(index)
                image, boxes = self.cutmix(image, boxes)
            else:
                image, boxes = self.load_stitch(index)
                image_id = "mosaic"

        target = {}
        target["bbox"] = boxes
        target["img_id"] = torch.tensor([index])
        target["cls"] = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target["img_size"] = torch.tensor(
            [IMG_SIZE, IMG_SIZE], dtype=torch.float32
        )
        target["img_scale"] = torch.ones((1,), dtype=torch.float32)

        if self.transforms:
            for i in range(10):
                sample = self.transforms(
                    **{
                        "image": image,
                        "bboxes": target["bbox"],
                        "labels": target["cls"],
                    }
                )
                if len(sample["bboxes"]) > 0:
                    image = sample["image"]
                    target["bbox"] = torch.stack(
                        tuple(map(torch.tensor, zip(*sample["bboxes"])))
                    ).permute(1, 0)
                    target["bbox"][:, [0, 1, 2, 3]] = target["bbox"][
                        :, [1, 0, 3, 2]
                    ]  # yxyx
                    target["cls"] = torch.stack(sample["labels"])
                    break

        return image, target, image_id

    def load_image(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df["image_id"] == image_id]
        image = cv2.imread(
            f"{self.image_dir}/{image_id}.jpg", cv2.IMREAD_COLOR
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        boxes = records[["x", "y", "w", "h"]].values.astype(np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes, image_id

    def cutmix(self, image_1, boxes_1):
        """cutmix augmentation"""
        rand_index = self.get_rand_index()
        image_2, boxes_2, _ = self.load_image(rand_index)
        imsize = image_1.shape[0]

        # create two random points
        x1, y1 = [
            int(random.uniform(imsize * 0.1, imsize * 0.4)) for _ in range(2)
        ]
        x2, y2 = [
            int(random.uniform(imsize * 0.6, imsize * 0.9)) for _ in range(2)
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
                ((boxes_1[:, 0] > x1) & (boxes_1[:, 2] < x2))
                & ((boxes_1[:, 1] > y1) & (boxes_1[:, 3] < y2))
            ),
            0,
        )

        # remove boxes with low visibility (<0.33 inside cutout)
        cutout_ords = np.array([x1, y1, x2, y2])
        boxes_1 = filter_boxes(boxes_1, cutout_ords, reverse=True, thresh=0.33)
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

    def cutout(self, image_1, boxes_1):
        """cutout augmentations"""
        h, w = image_1.shape[:2]
        # create random masks with random number of items
        scale_lim = random.randint(3, 6)
        scales_stacked = [
            [0.5 / (2 ** x)] * (2 ** x) for x in range(0, scale_lim, 1)
        ]
        scales = [item for sublist in scales_stacked for item in sublist]
        for s in scales:
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))
            # cutout box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)
            # apply cutout
            image_1[ymin:ymax, xmin:xmax] = [
                random.randint(48, 224) for _ in range(3)
            ]
            # clean labels
            if len(boxes_1) and s > 0.03:
                boxes_1 = np.delete(
                    boxes_1,
                    np.where(
                        ((boxes_1[:, 0] > xmin) & (boxes_1[:, 2] < xmax))
                        & ((boxes_1[:, 1] > ymin) & (boxes_1[:, 3] < ymax))
                    ),
                    0,
                )
                cutout_ords = np.array([xmin, ymin, xmax, ymax])
                boxes_1 = filter_boxes(
                    boxes_1, cutout_ords
                )  # <-- 0.33 set here
                boxes_1 = boxes_1[
                    np.where(
                        (boxes_1[:, 2] - boxes_1[:, 0])
                        * (boxes_1[:, 3] - boxes_1[:, 1])
                        > 64
                    )
                ]

        return image_1, boxes_1

    def load_stitch(self, index):
        """load 4 images and stitch them"""
        stitch_labels = []
        s = IMG_SIZE
        yc, xc = [
            int(random.uniform(-x, 2 * s + x)) for x in [-s // 2, -s // 2]
        ]
        indices = [index] + [self.get_rand_index() for _ in range(3)]
        pad = 64

        for i, index in enumerate(indices):
            # Load image
            img, boxes, _ = self.load_image(index)
            h, w = img.shape[:2]
            if i == 0:  # top left
                stitch_image = np.full(
                    (s * 2, s * 2, img.shape[2]), 0, dtype=np.uint8
                )
                xmin, xmax, ymin, ymax = 0, xc, 0, yc
                limits = xmin, xmax, ymin, ymax
                cropped_boxes = crop_boxes_stitch(
                    limits, boxes, pad, thresh=0.4
                )
                stitch_labels = cropped_boxes
            elif i == 1:  # top right
                xmin, xmax, ymin, ymax = xc, w, 0, yc
                limits = xmin, xmax, ymin, ymax
                cropped_boxes = crop_boxes_stitch(
                    limits, boxes, pad, thresh=0.4
                )
                stitch_labels = np.concatenate((stitch_labels, cropped_boxes))
            elif i == 2:  # bottom left
                xmin, xmax, ymin, ymax = 0, xc, yc, h
                limits = xmin, xmax, ymin, ymax
                cropped_boxes = crop_boxes_stitch(
                    limits, boxes, pad, thresh=0.4
                )
                stitch_labels = np.concatenate((stitch_labels, cropped_boxes))
            elif i == 3:  # bottom right
                xmin, xmax, ymin, ymax = xc, w, yc, h
                limits = xmin, xmax, ymin, ymax
                cropped_boxes = crop_boxes_stitch(
                    limits, boxes, pad, thresh=0.4
                )
                stitch_labels = np.concatenate((stitch_labels, cropped_boxes))

            stitch_image[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax]

        return stitch_image, stitch_labels

    def get_rand_index(self):
        rand_index = np.random.choice([*range(0, self.image_ids.shape[0] - 1)])
        return rand_index


def collate_fn(batch):
    return tuple(zip(*batch))


@jit(nopython=True)
def crop_boxes_stitch(limits, labels, pad, thresh=0.4) -> numba.float32[:, :]:
    xmin, xmax, ymin, ymax = limits
    cutout_ords = np.array([xmin, ymin, xmax, ymax])
    boxes = labels[
        (
            (labels[:, 0] > xmin - pad)
            & (labels[:, 2] < xmax + pad)
            & (labels[:, 1] > ymin - pad)
            & (labels[:, 3] < ymax + pad)
        )
    ]
    boxes = filter_boxes(boxes, cutout_ords, reverse=False, thresh=0.4)
    boxes = boxes[
        np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0)
    ]
    return boxes


@jit(nopython=True)
def filter_boxes(
    boxes_1, cutout_ords, reverse=True, thresh=0.33
) -> numba.float32[:, :]:
    boxes_1_copy = np.zeros(boxes_1.shape)
    for i in range(boxes_1.shape[0]):
        inter = calculate_intersection(boxes_1[i], cutout_ords)
        if not reverse:
            if inter > thresh:
                boxes_1_copy[i] = boxes_1[i]
        else:
            if inter < thresh:
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
