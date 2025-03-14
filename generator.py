import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# To control feature map in generator
ngf = 64


class StableGeneratorResnet(nn.Module):
    def __init__(self, gen_dropout, data_dim, inception=False, isTrain=False):
        """
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        """
        super(StableGeneratorResnet, self).__init__()

        assert data_dim == "high", "set to high"
        logger = logging.getLogger("CDA.inference")

        if isTrain:
            logger.info("Gen Dropout: {}, Depth: {}".format(gen_dropout, data_dim))

        self.inception = inception
        self.data_dim = data_dim
        # Input_size = 3, n, n

        self.block1 = nn.Sequential(
            # nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_channels=3, out_channels=ngf, kernel_size=7, padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=ngf,
                out_channels=ngf * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4, gen_dropout)
        self.resblock2 = ResidualBlock(ngf * 4, gen_dropout)
        self.resblock3 = ResidualBlock(ngf * 4, gen_dropout)
        self.resblock4 = ResidualBlock(ngf * 4, gen_dropout)
        self.resblock5 = ResidualBlock(ngf * 4, gen_dropout)
        self.resblock6 = ResidualBlock(ngf * 4, gen_dropout)
        # self.resblock7 = ResidualBlock(ngf*4)
        # self.resblock8 = ResidualBlock(ngf*4)
        # self.resblock9 = ResidualBlock(ngf*4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 2),
            # nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf),
            # nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            # nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=5)
        )

    def forward(self, input):

        # logger.info(f"Input: {torch.mean(input)}")

        x = self.block1(input)

        # logger.info(f"block1: {torch.mean(x)}")

        x = self.block2(x)

        # logger.info(f"block2: {torch.mean(x)}")

        x = self.block3(x)
        # logger.info(f"block3: {torch.mean(x)}")

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        # logger.info(f"block6: {torch.mean(x)}")

        # x = self.resblock7(x)
        # x = self.resblock8(x)
        # x = self.resblock9(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)

        # logger.info(f"upsample2: {torch.mean(x)}")

        x = self.blockf(x)

        # CHANGED
        return (torch.tanh(x) + 1) / 2.0  # Output range [0 1]

        # return torch.sigmoid(x)  # Output range [0 1]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, gen_dropout):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
            # nn.InstanceNorm2d(num_filters),
            nn.ReLU(True),
            # CHANGED
            # if gen_dropout > 0.01:
            nn.Dropout(gen_dropout),
            # nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
            # nn.InstanceNorm2d(num_filters),
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


if __name__ == "__main__":
    netG = GeneratorResnet(data_dim="low")
    test_sample = torch.rand(1, 3, 32, 32)
    print("Generator output:", netG(test_sample).size())
    print(
        "Generator parameters:",
        sum(p.numel() for p in netG.parameters() if p.requires_grad),
    )
