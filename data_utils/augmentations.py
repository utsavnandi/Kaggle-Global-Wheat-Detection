import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

IMG_SIZE = 320


# Albumentations
def get_train_transform():
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(
                shift_limit=0.075,
                scale_limit=0.1,
                rotate_limit=10,
                interpolation=1,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=(1, 4), p=0.5),
                    A.MedianBlur(blur_limit=4, p=0.5),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
                    A.HueSaturationValue(
                        hue_shift_limit=25,
                        sat_shift_limit=25,
                        val_shift_limit=30,
                        p=0.5,
                    ),
                ],
                p=0.75,
            ),
            A.CLAHE(clip_limit=3.0, p=0.5),
            A.ImageCompression(quality_lower=80, p=0.33),
            A.OneOf(
                [
                    A.ISONoise(p=0.33),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.33),
                    A.MultiplicativeNoise(p=0.33),
                ],
                p=0.5,
            ),
            A.Normalize(always_apply=True, p=1.0),
            ToTensorV2(always_apply=True, p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=10,
            min_visibility=0,
            label_fields=["labels"],
        ),
    )


def get_valid_transform():
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
            A.Normalize(always_apply=True, p=1.0),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=0,
            min_visibility=0,
            label_fields=["labels"],
        ),
    )
