import torch
import torchvision

from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN

import timm


def timm_resnet_fpn_backbone(
    backbone_name, pretrained=True, trainable_layers=3
):
    backbone = timm.create_model(backbone_name, pretrained=pretrained)

    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers
    ]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
            print(f"frozen {name}")

    return_layers = {
        "layer1": "0",
        "layer2": "1",
        "layer3": "2",
        "layer4": "3",
    }

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels
    )


def get_model(
    backbone_name="resnet50",
    detector_name="fasterrcnn",
    trainable_layers=3,
    model_ckpt=None,
):
    """Constructs a fasterrcnn or maskrcnn detector with the given backbone"""
    num_classes = 2  # 1 class (wheat) + background
    if model_ckpt:
        # backbone = resnet_fpn_backbone('resnet101', True)
        backbone = timm_resnet_fpn_backbone(
            backbone_name, False, trainable_layers
        )
    else:
        backbone = timm_resnet_fpn_backbone(
            backbone_name, True, trainable_layers
        )
    if detector_name == "fasterrcnn":
        model = FasterRCNN(backbone, num_classes)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )
    elif detector_name == "maskrcnn":
        model = MaskRCNN(backbone, num_classes)
        in_features_mask = (
            model.roi_heads.mask_predictor.conv5_mask.in_channels
        )
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
    else:
        raise Exception(f"{detector_name} is not supported")
    if model_ckpt is not None:
        model.load_state_dict(torch.load(model_ckpt)["model_state_dict"])
        print("loaded ckpt")
    return model


def unfreeze_all_layers(model):
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:3]
    for name, parameter in model.backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(True)
    print(f"unfrozen all layers")
    return model
