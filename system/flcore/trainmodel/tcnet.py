import torch
import torch.nn as nn
import torch.nn.functional as F


class TC_net(nn.Module):
    def __init__(self, in_features=512, out_channels=1, img_size=28):
        super(TC_net, self).__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        self.img_size = img_size

        # Chiếu từ không gian đặc trưng z sang spatial feature map 64x7x7
        self.fc = nn.Linear(in_features, 64 * 7 * 7)

        # Chuỗi deconvolution phóng đại kích thước (7x7 -> 14x14 -> 28x28)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 7x7 -> 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1, bias=False),  # 14x14 -> 28x28
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)
        x_hat = self.deconv(h)
        if x_hat.shape[-1] != self.img_size or x_hat.shape[-2] != self.img_size:
            x_hat = F.interpolate(x_hat, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return x_hat