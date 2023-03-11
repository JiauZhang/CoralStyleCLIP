import torch
from torch import nn
from .facial_recognition.model_irse import Backbone
# modified from https://github.com/orpatashnik/StyleCLIP/criteria/id_loss.py
class IDLoss(nn.Module):
    def __init__(self, ckpt, device='cuda'):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(ckpt, map_location=device))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet = self.facenet.to(device)

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        mul = torch.mul(y_hat_feats, y_feats)
        loss = (1 - mul.sum(dim=-1)).sum()
        return loss
