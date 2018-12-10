import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrainedmodels.models.senet import pretrained_settings, SENet, SEResNeXtBottleneck, model_zoo

from points_to_image import points_to_image
import utils


def strokes_to_seresnext50_32x4d(img_size, window, num_classes):
    return StrokesToSeResNeXt(
        img_size, window, num_classes,
        block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], groups=32, reduction=16,
        dropout_p=None, inplanes=64, input_3x3=False,
        downsample_kernel_size=1, downsample_padding=0,
        pretrained_key='se_resnext50_32x4d'
    )


def srokes_to_seresnext101_32x4d(img_size, window, num_classes):
    return StrokesToSeResNeXt(
        img_size, window, num_classes,
        block=SEResNeXtBottleneck, layers=[3, 4, 23, 3], groups=32, reduction=16,
        dropout_p=None, inplanes=64, input_3x3=False,
        downsample_kernel_size=1, downsample_padding=0,
        pretrained_key='se_resnext101_32x4d'
    )


class StrokesToSeResNeXt(SENet):
    def __init__(self, img_size, window, num_classes, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True,
             downsample_kernel_size=3, downsample_padding=1, pretrained_key=None):
        nn.Module.__init__(self)
        self.img_size = img_size
        self.window = window
        self.num_classes = num_classes
        self.pretrained_key = pretrained_key

        self.initial = nn.Sequential(
            nn.Conv1d(3,  32, kernel_size=3, stride=1, padding=1, dilation=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=4, dilation=4), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=8, dilation=8), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )

        self.convert = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True)
        )

        self.inplanes = inplanes
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.cls = nn.Linear(512 * block.expansion, num_classes)

    def load_pretrained(self):
        pretrained = model_zoo.load_url(pretrained_settings[self.pretrained_key]['imagenet']['url'])
        model_state_dict = self.state_dict()
        update_dict = {k: v for k, v in pretrained.items() if k in model_state_dict and k.startswith('layer')}
        print(update_dict.keys())
        model_state_dict.update(update_dict)
        self.load_state_dict(model_state_dict)

    def partial_freeze(self):
        print('-' * 64)
        print('FREEZE')
        print('-' * 64)
        for name, child in self.named_children():
            if name.startswith('layer'):
                for param in child.parameters():
                    param.requires_grad = False

    def unfreeze(self):
        print('-'*64)
        print('UNFREEZE')
        print('-' * 64)
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, points_tensor, indices_tensor):
        xy = utils.batch_index_select(points_tensor, indices_tensor, 2).permute(0, 2, 1)

        # valid entries of points_tensor are indices_tensor
        dxy = points_tensor[:, :2, 1:] - points_tensor[:, :2, :-1]
        dxy /= 0.035
        t = (points_tensor[:, 2, 1:] + points_tensor[:, 2, :-1]) / 2
        x = torch.cat([dxy, t.unsqueeze(1)], dim=1)
        # valid entries of points_tensor are (indices_tensor - 1):indices_tensor

        x = self.initial(x)

        # Only even windows are valid after doing that difference above
        assert self.window % 2 == 0
        x = x.unfold(2, self.window, 1)
        # valid entries of features are indices_tensor - self.window // 2 - 1:

        x = utils.batch_index_select(x, indices_tensor - 1 - (self.window // 2) + 1, 2)
        batch_size = x.size()[0]
        num_points = x.size()[1]
        x = x.view(batch_size, num_points, -1).permute(0, 2, 1)

        x = self.convert(x)

        i = ((xy[:, :2, :]+1)*((self.img_size - 1) / 2)).long()
        i = torch.clamp(i, 0, self.img_size - 1)

        img = points_to_image(i, x.permute(0, 2, 1)).permute(0, 3, 1, 2)

        x = self.layer1(img)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.cls(x)

        return x
