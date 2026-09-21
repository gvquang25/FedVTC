import torch
import torch.nn as nn
import torch.nn.functional as F


def total_variation_loss(img):
    """
    Computes Total Variation (TV) Loss to penalize high-frequency noise
    and prevent adversarial grain artifacts in generated images.
    Input: img tensor of shape (B, C, H, W)
    """
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


class TC_net(nn.Module):
    def __init__(self, in_features=512, out_channels=1, img_size=28):
        super(TC_net, self).__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        self.img_size = img_size

        # Spatial resolution adapts directly:
        # img_size 28 -> init_size 7 (7 -> 14 -> 28)
        # img_size 32 -> init_size 8 (8 -> 16 -> 32)
        self.init_size = img_size // 4
        self.fc = nn.Linear(in_features, 64 * self.init_size * self.init_size)

        # Use GroupNorm instead of BatchNorm2d:
        # 1. Independent of batch size and running statistics
        # 2. Mathematically sound to average across non-IID clients
        # 3. Exactly identical behavior in train and eval modes
        self.deconv = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # init -> 2*init
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1, bias=False),  # 2*init -> 4*init
            nn.Tanh()
        )

    def forward(self, z):
        # Flatten if z is (B, D, 1, 1) or similar
        if z.dim() > 2:
            z = z.view(z.size(0), -1)
        h = self.fc(z)
        h = h.view(-1, 64, self.init_size, self.init_size)
        x_hat = self.deconv(h)

        # Fallback only for non-multiples of 4 (rare)
        if x_hat.shape[-1] != self.img_size or x_hat.shape[-2] != self.img_size:
            x_hat = F.interpolate(x_hat, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return x_hat