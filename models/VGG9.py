import torch.nn as nn
from models.Neurons import VEILIF, MVLIF, MVALIF, MALIF, CMLIF, CMVLIF, CMILIF, CMIVLIF, CMSLIF, CMSVLIF, SAMLIF, \
    DropBlockLIFS, DBLIF, WTALIF, LIF, AVGMaskLIF, AVGConMLIF, TAB_Layer, PLIF, MLF_unit, tdBatchNorm
import numpy as np
import matplotlib.pyplot as plt


class VGG9(nn.Module):
    def __init__(self, num_classes, useAvg, inchannel=2):
        super(VGG9, self).__init__()
        self.useAvg = useAvg
        self.avgpool_input = nn.AvgPool2d(kernel_size=4, stride=4)
        # self.vth=nn.Para
        self.feature = nn.Sequential(
            nn.Conv2d(inchannel, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            # tdBatchNorm(nn.BatchNorm2d(64)),
            # TAB_Layer(num_features=64),
            # LIF(),
            # PLIF(),
            # DropLIF(),
            # CMILIF(64),
            # CMIVLIF(64),
            # CMLIF(64),
            # CMVLIF(64),
            # CMSLIF(64),
            # CMSVLIF(64),
            # SAMLIF(64),
            # MVALIF(inchannel=64),
            # MVLIF(inchannel=64),
            # VEILIF(inchannel=64),
            # MLF_unit(),
            # MALIF(),
            DropBlockLIFS(),
            # DBLIF(),
            # WTALIF(),
            # AVGConMLIF(inchannel=64),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            # tdBatchNorm(nn.BatchNorm2d(128)),
            # TAB_Layer(num_features=128),
            # LIF(),
            # PLIF(),
            # DropLIF(),
            # CMILIF(128),
            # CMIVLIF(128),
            # CMLIF(128),
            # CMVLIF(128),
            # CMSLIF(128),
            # CMSVLIF(128),
            # SAMLIF(128),
            # MVALIF(inchannel=128),
            # MVLIF(inchannel=128),
            # VEILIF(inchannel=128),
            # MLF_unit(),
            # MALIF(),
            DropBlockLIFS(),
            # DBLIF(),
            # WTALIF(),
            # AVGConMLIF(inchannel=128),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            # tdBatchNorm(nn.BatchNorm2d(256)),
            # TAB_Layer(num_features=256),
            # LIF(),
            # PLIF(),
            # DropLIF(),
            # CMILIF(256),
            # CMIVLIF(256),
            # CMLIF(256),
            # CMVLIF(256),
            # CMSLIF(256),
            # CMSVLIF(256),
            # SAMLIF(256),
            # MVALIF(inchannel=256),
            # MVLIF(inchannel=256),
            # VEILIF(inchannel=256),
            # MLF_unit(),
            # MALIF(),
            DropBlockLIFS(),
            # DBLIF(),
            # WTALIF(),
            # AVGConMLIF(inchannel=256),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            # tdBatchNorm(nn.BatchNorm2d(256)),
            # TAB_Layer(num_features=256),
            # LIF(),
            # PLIF(),
            # DropLIF(),
            # CMILIF(256),
            # CMIVLIF(256),
            # CMLIF(256),
            # CMVLIF(256),
            # CMSLIF(256),
            # CMSVLIF(256),
            # SAMLIF(256),
            # MVALIF(inchannel=256),
            # MVLIF(inchannel=256),
            # VEILIF(inchannel=256),
            # MLF_unit(),
            # MALIF(),
            DropBlockLIFS(),
            # DBLIF(),
            # WTALIF(),
            # AVGConMLIF(inchannel=256),
            nn.AvgPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            # tdBatchNorm(nn.BatchNorm2d(512)),
            # TAB_Layer(num_features=512),
            # LIF(),
            # PLIF(),
            # DropLIF(),
            # CMILIF(512),
            # CMIVLIF(512),
            # CMLIF(512),
            # CMVLIF(512),
            # CMSLIF(512),
            # CMSVLIF(512),
            # SAMLIF(512),
            # MVALIF(inchannel=512),
            # MVLIF(inchannel=512),
            # VEILIF(inchannel=512),
            # MLF_unit(),
            # MALIF(),
            DropBlockLIFS(),
            # DBLIF(),
            # WTALIF(),
            # AVGConMLIF(inchannel=512),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            # tdBatchNorm(nn.BatchNorm2d(512)),
            # TAB_Layer(num_features=512),
            # LIF(),
            # PLIF(),
            # DropLIF(),
            # CMILIF(512),
            # CMIVLIF(512),
            # CMLIF(512),
            # CMVLIF(512),
            # CMSLIF(512),
            # CMSVLIF(512),
            # SAMLIF(512),
            # MVALIF(inchannel=512),
            # MVLIF(inchannel=512),
            # VEILIF(inchannel=512),
            # MLF_unit(),
            # MALIF(),
            DropBlockLIFS(),
            # DBLIF(),
            # WTALIF(),
            # AVGConMLIF(inchannel=512),
            nn.AvgPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            # tdBatchNorm(nn.BatchNorm2d(512)),
            # TAB_Layer(num_features=512),
            # LIF(),
            # PLIF(),
            # DropLIF(),
            # CMILIF(512),
            # CMIVLIF(512),
            # CMLIF(512),
            # CMVLIF(512),
            # CMSLIF(512),
            # CMSVLIF(512),
            # SAMLIF(512),
            # MVALIF(inchannel=512),
            # MVLIF(inchannel=512),
            # VEILIF(inchannel=512),
            # MLF_unit(),
            # MALIF(),
            DropBlockLIFS(),
            # DBLIF(),
            # WTALIF(),
            # AVGConMLIF(inchannel=512),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            # tdBatchNorm(nn.BatchNorm2d(512)),
            # TAB_Layer(num_features=512),
            # LIF(),
            # PLIF(),
            # DropLIF(),
            # CMILIF(512),
            # CMIVLIF(512),
            # CMLIF(512),
            # CMSLIF(512),
            # CMSVLIF(512),
            # SAMLIF(512),
            # CMVLIF(512),
            # MVALIF(inchannel=512),
            # MVLIF(inchannel=512),
            # VEILIF(inchannel=512),
            # MLF_unit(),
            # MALIF(),
            DropBlockLIFS(),
            # DBLIF(),
            # WTALIF(),
            # AVGConMLIF(inchannel=512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes, bias=False),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # for i in range(8):
        #     visualization_data = x[i*128+4]
        #     visualization_data_cpu = visualization_data.cpu()
        #     w = visualization_data_cpu.shape[1]
        #     h = visualization_data_cpu.shape[2]
        #
        #     rgb_image = np.zeros((w, h, 3))
        #
        #     # 将两个通道映射到RGB图像的前两个通道
        #     rgb_image[:, :, 0] = visualization_data_cpu[0, :, :]  # Red
        #     rgb_image[:, :, 1] = visualization_data_cpu[1, :, :]  # Green
        #
        #     # 显示RGB图像
        #     plt.imshow(rgb_image)
        #     plt.title('RGB Image')
        #     plt.axis('off')
        #     plt.show()
        # return ;
        if self.useAvg:
            x = self.avgpool_input(x)
        # visualization_data = x[5*128+11]
        # visualization_data_cpu = visualization_data.cpu()
        # w = visualization_data_cpu.shape[1]
        # h = visualization_data_cpu.shape[2]
        #
        # rgb_image = np.zeros((w, h, 3))
        #
        # # 将两个通道映射到RGB图像的前两个通道
        # rgb_image[:, :, 0] = visualization_data_cpu[0, :, :]  # Red
        # rgb_image[:, :, 1] = visualization_data_cpu[1, :, :]  # Green
        #
        # # 显示RGB图像
        # plt.imshow(rgb_image)
        # plt.title('RGB Image')
        # plt.axis('off')
        # plt.show()
        # print('-----x.shape----')
        # print(x.shape)
        out = self.feature(x)

        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        return ;
        # return out
