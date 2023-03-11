import torch, clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from torch import nn
from .id_loss import IDLoss

BICUBIC = InterpolationMode.BICUBIC
__clip_model__ = None
__clip_preprocess__ = None
__id_model__ = None

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def clip_loss(image, image_bar, text):
    global __clip_preprocess__
    global __clip_model__
    if __clip_model__ is None:
        __clip_model__, _ = clip.load("ViT-B/32", device=image.device)
        __clip_preprocess__ = _transform(__clip_model__.visual.input_resolution)
    image = __clip_preprocess__(image * 128 + 127)
    image_bar = __clip_preprocess__(image_bar * 128 + 127)
    text = clip.tokenize([text]).to(image.device)
    c1 = 1 - __clip_model__(image, text)[0] / 100
    c2 = 1 - __clip_model__(image_bar, text)[0] / 100
    loss = 0.5 * (c1 + c2).sum()
    return loss

def l2_loss(val):
    return torch.sum(torch.pow(val, 2))

def id_loss(image1, image2):
    global __id_model__
    if __id_model__ is None:
        __id_model__ = IDLoss('pretrained_models/model_ir_se50.pt', device=image1.device)
    loss = __id_model__(image1, image2).sum()
    return loss

# segment selection
def ss_area_loss(e):
    return torch.sum(e)

# convolutional attention network
__norms__ = None
def can_area_loss(masks):
    global __norms__
    if __norms__ is None:
        __norms__, res = [], 4
        for i in range(len(masks)//2):
            __norms__.extend([res, res])
            res *= 2

    loss = 0
    for mask, norm in zip(masks, __norms__):
        loss += norm * torch.sum(mask)
    return loss

def tv_loss(masks):
    loss = 0
    for mask in masks:
        loss += torch.pow(mask[:, :, :, :-1] - mask[:, :, :, 1:], 2).sum()
        loss += torch.pow(mask[:, :, :-1, :] - mask[:, :, 1:, :], 2).sum()
    return loss
