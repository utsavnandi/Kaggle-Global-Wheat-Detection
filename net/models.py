import torch
import torch.nn as nn

# import torchvision
import timm

from functools import reduce
from collections import OrderedDict

from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from effdet import (
    get_efficientdet_config,
    EfficientDet,
    # create_model,
    DetBenchTrain,
    DetBenchPredict,
)
from effdet.efficientdet import HeadNet


def timm_resnet_fpn_backbone(
    backbone_name, pretrained=True, trainable_layers=None
):
    """Constructs a fpn backbone for fasterrcnn"""
    backbone = timm.create_model(backbone_name, pretrained=pretrained)

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


def convert_layers(
    model,
    layer_type_old,
    layer_type_new,
    convert_weights=False,
    num_groups=None,
):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_layers(
                module, layer_type_old, layer_type_new, convert_weights
            )
        if type(module) == layer_type_old:
            layer_old = module
            layer_new = layer_type_new(
                module.num_features if num_groups is None else num_groups,
                module.num_features,
                module.eps,
                module.affine,
            )
            if convert_weights:
                layer_new.weight = layer_old.weight
                layer_new.bias = layer_old.bias
            model._modules[name] = layer_new
    return model


def get_train_model(
    config_name, img_size, model_ckpt=None, useGN=False, light=True
):
    config = get_efficientdet_config(config_name)
    model = EfficientDet(config, pretrained_backbone=True)
    config.num_classes = 1
    config.image_size = img_size
    model.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
        norm_kwargs=dict(eps=0.001, momentum=0.01),
    )
    if useGN is True:
        model = convert_layers(
            model, nn.BatchNorm2d, nn.GroupNorm, True, num_groups=2
        )
    # Replace BatchNorm with GroupNorm
    if model_ckpt is not None:
        if light is True:
            count = 0
            state_dict = torch.load(model_ckpt)["state_dict"]
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith("model.model."):
                    new_key = reduce(
                        lambda a, b: a + "." + b, key.split(".")[2:]
                    )
                    if new_key in model.state_dict():
                        new_state_dict[new_key] = value
                        count += 1
            model.load_state_dict(new_state_dict)
            print(f"loaded {count} keys")
        else:
            model.load_state_dict(torch.load(model_ckpt)["state_dict"])

    return DetBenchTrain(model, config)


def get_predict_model(
    config_name, img_size, model_ckpt, useGN=False, light=False
):
    config = get_efficientdet_config(config_name)
    model = EfficientDet(config, pretrained_backbone=False)
    config.num_classes = 1
    config.image_size = img_size
    model.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
        norm_kwargs=dict(eps=0.001, momentum=0.01),
    )
    if useGN is True:
        model = convert_layers(
            model, nn.BatchNorm2d, nn.GroupNorm, True, num_groups=2
        )
    if light is True:
        count = 0
        state_dict = torch.load(model_ckpt)["state_dict"]
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("model.model."):
                new_key = reduce(lambda a, b: a + "." + b, key.split(".")[2:])
                if new_key in model.state_dict():
                    new_state_dict[new_key] = value
                    count += 1
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load(model_ckpt)["state_dict"])

    print(f"loaded {count} keys")
    return DetBenchPredict(model, config)
