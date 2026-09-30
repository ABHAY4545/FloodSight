import torch
import torch.nn as nn


def double_conv(in_channel, out_channel):
    block = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, out_channel),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, out_channel),
        nn.ReLU()
    )
    return block


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = double_conv(3,32)
        self.encoder2 = double_conv(32,64)
        self.encoder3 = double_conv(64,128)
        self.encoder4 = double_conv(128,256)

        self.trans_conv4 = nn.ConvTranspose2d(512,256,2,2)
        self.decoder4 = double_conv(512,256)

        self.trans_conv3 = nn.ConvTranspose2d(256,128,2,2)
        self.decoder3 = double_conv(256,128)

        self.trans_conv2 = nn.ConvTranspose2d(128,64,2,2)
        self.decoder2 = double_conv(128,64)

        self.trans_conv1 = nn.ConvTranspose2d(64,32,2,2)
        self.decoder1 = double_conv(64,32)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(256,512)
        self.final_classifier = nn.Conv2d(32,10,1)

    def forward(self, x):
        skip1 = self.encoder1(x)
        x = self.pool(skip1)

        skip2 = self.encoder2(x)
        x = self.pool(skip2)

        skip3 = self.encoder3(x)
        x = self.pool(skip3)

        skip4 = self.encoder4(x)
        x = self.pool(skip4)

        x = self.bottleneck(x)

        x = self.trans_conv4(x)
        x = torch.cat((x,skip4), dim=1)
        x = self.decoder4(x)

        x = self.trans_conv3(x)
        x = torch.cat((x,skip3), dim=1)
        x = self.decoder3(x)

        x = self.trans_conv2(x)
        x = torch.cat((x,skip2), dim=1)
        x = self.decoder2(x)

        x = self.trans_conv1(x)
        x = torch.cat((x,skip1), dim=1)
        x = self.decoder1(x)

        x = self.final_classifier(x)

        return x


if __name__ == "__main__":
    print("Model.py")