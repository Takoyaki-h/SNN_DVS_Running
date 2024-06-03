import torch
import torch.nn as nn
from models.Neurons import PVLIF, PVBMLIF, PVBMLIFH, PVBMLYLIFH


class VGG9(nn.Module):
    def __init__(self, num_classes):
        super(VGG9, self).__init__()
        self.avgpool_input = nn.AvgPool2d(kernel_size=4, stride=4)
        self.feature = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            # PVLIF(max_threshold=1),
            # PVBMLYLIFH(max_threshold=1, layer=1),
            PVBMLIFH(max_threshold=1),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            # PVLIF(max_threshold=1),
            # PVBMLYLIFH(max_threshold=1, layer=2),
            PVBMLIFH(max_threshold=1),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            # PVLIF(max_threshold=1),
            PVBMLIFH(max_threshold=1),
            # PVBMLYLIFH(max_threshold=1, layer=3),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            # PVLIF(max_threshold=1),
            PVBMLIFH(max_threshold=1),
            # PVBMLYLIFH(max_threshold=1, layer=4),
            nn.AvgPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            # PVLIF(max_threshold=1),
            # PVBMLYLIFH(max_threshold=1, layer=5),
            PVBMLIFH(max_threshold=1),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            # PVLIF(max_threshold=1),
            PVBMLIFH(max_threshold=1),
            # PVBMLYLIFH(max_threshold=1, layer=6),
            nn.AvgPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            # PVLIF(max_threshold=1),
            PVBMLIFH(max_threshold=1),
            # PVBMLYLIFH(max_threshold=1, layer=7),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            # PVLIF(max_threshold=1),
            PVBMLIFH(max_threshold=1),
            # PVBMLYLIFH(max_threshold=1, layer=8),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes, bias=False),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def set_max_threshold(self, max_threshold):
        for module in self.modules():
            if isinstance(module, PVBMLIFH):
                module.vth = max_threshold

    def forward(self, x):
        x = self.avgpool_input(x)
        out = self.feature(x)

        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        return out
