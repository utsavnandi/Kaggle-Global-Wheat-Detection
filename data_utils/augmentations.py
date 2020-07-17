import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# Albumentations
def get_train_transform(img_size):
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size, p=1),
            A.RandomSizedBBoxSafeCrop(
                img_size, img_size, interpolation=1, p=0.33
            ),
            A.Flip(),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                interpolation=1,
                border_mode=0,
                value=0,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=(1, 3), p=0.33),
                    A.MedianBlur(blur_limit=3, p=0.33),
                    A.ImageCompression(quality_lower=50, p=0.33),
                ],
                p=0.33,
            ),
            A.OneOf(
                [
                    A.RandomGamma(gamma_limit=(90, 110), p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=0.2, p=0.4),
                    A.HueSaturationValue(
                        hue_shift_limit=25,
                        sat_shift_limit=25,
                        val_shift_limit=30,
                        p=0.5,
                    ),
                ],
                p=0.4,
            ),
            A.CLAHE(clip_limit=2, p=0.2),
            A.Normalize(always_apply=True, p=1.0),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=64,
            min_visibility=0.1,
            label_fields=["labels"],
        ),
    )


def get_valid_transform(img_size):
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size, p=1),
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
