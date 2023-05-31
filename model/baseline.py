import torch
import torch.nn as nn

from .backbones.PRSNetEn import MaskAutoencoderConv
from .backbones.PRSNetEnV2 import MaskAutoencoderConvV2


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weigh, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, cfg):
        super(Baseline, self).__init__()

        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAINED
        mask_ratio = cfg.MODEL.MASK_RATIO
        self.use_mixed_precision = cfg.USE_MIXED_PRECISION

        if model_name == 'prs':
            print("The rps network is trained using mask centroid loss!")
            self.base = MaskAutoencoderConv(mask_ratio=mask_ratio)

        elif model_name == 'prsv2':
            self.base = MaskAutoencoderConvV2()

        self.model_name = model_name

        if pretrain_choice and not cfg.MODEL.RESUME_TRAINING and not cfg.TEST.ONLY_TEST:
            self.base.load_param(model_path)
            print('Loading pretrained model ......')

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        base_out, mask_base_out = self.base(x)
        global_feat = self.gap(base_out)
        global_feat = global_feat.view(global_feat.shape[0], -1)

        mask_global_feat = self.gap(mask_base_out)
        mask_global_feat = mask_global_feat.view(global_feat.shape[0], -1)

        # mask_global_feat为增加部分
        return base_out, global_feat, mask_global_feat

    def load_param(self, trained_path, load_specific=None):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if load_specific is not None:
                if load_specific in i:
                    self.state_dict()[i].copy_(param_dict[i])
            else:
                if 'classifier' in i:
                    continue
                self.state_dict()[i].copy_(param_dict[i])



