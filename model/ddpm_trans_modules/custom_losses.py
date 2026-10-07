import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import VGG19_Weights





class VGGPerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.vgg = vgg[:16].eval()  # Use the first 16 layers
        for param in self.vgg.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return F.mse_loss(x_features, y_features)

class ContentPerceptualLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(ContentPerceptualLoss, self).__init__()
        self.huber_loss = nn.HuberLoss()  # Huber Loss
        self.perceptual_loss = VGGPerceptualLoss()  # Perceptual Loss
        self.alpha = alpha  # Weight for Huber Loss
        self.beta = beta    # Weight for Perceptual Loss

    def forward(self, input, target):
        # Compute Huber Loss
        huber = self.huber_loss(input, target)
        
        # Compute Perceptual Loss
        perceptual = self.perceptual_loss(input, target)

        # Combine the losses
        return self.alpha * huber + self.beta * perceptual
    
class L1PerceptualLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(L1PerceptualLoss, self).__init__()
        self.l1_loss = nn.L1Loss()  # L1 Loss
        self.perceptual_loss = VGGPerceptualLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        # Compute L1 Loss
        l1 = self.l1_loss(input, target)
        
        # Compute Perceptual Loss
        perceptual = self.perceptual_loss(input, target)

        # Combine the losses
        return self.alpha * l1 + self.beta * perceptual
    

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = None  # Delay creation of the window to handle device placement

    def create_window(self, window_size, channel=3, device="cpu"):
        _1D_window = torch.hann_window(window_size, periodic=False).unsqueeze(1)
        window = _1D_window @ _1D_window.T
        window /= window.sum()
        self.window = window.view(1, 1, window_size, window_size).expand(channel, 1, window_size, window_size).to(device)
        self.channel = channel

    def forward(self, img1, img2):
        if img1.size(1) != self.channel or self.window is None:
            # Initialize or update the window if channel changes or window is not created yet
            self.create_window(self.window_size, channel=img1.size(1), device=img1.device)

        # Ensure window is on the same device as input images
        self.window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim = ssim_map.mean() if self.size_average else ssim_map.mean([1, 2, 3])
        
        return 1 - ssim  # return 1 - SSIM as loss


class ContentPerceptualSSIMLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(ContentPerceptualSSIMLoss, self).__init__()
        self.huber_loss = nn.HuberLoss()  # Huber Loss
        self.perceptual_loss = VGGPerceptualLoss()  # VGG Perceptual Loss
        self.ssim_loss = SSIMLoss(window_size=11)  # SSIM Loss
        self.alpha = alpha  # Weight for Huber Loss
        self.beta = beta    # Weight for Perceptual Loss
        self.gamma = gamma  # Weight for SSIM Loss

    def forward(self, input, target):
        # Compute Huber Loss
        huber = self.huber_loss(input, target)
        
        # Compute Perceptual Loss
        perceptual = self.perceptual_loss(input, target)
        
        # Compute SSIM Loss (1 - SSIM for loss form)
        ssim = self.ssim_loss(input, target)

        # Combine the losses
        return self.alpha * huber + self.beta * perceptual + self.gamma * ssim

