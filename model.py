import sys
sys.path.append("PerceptualSimilarity/")
import os
import utils
import torch
import numpy as np
from torch import nn
import torchgeometry
from kornia import color
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import unet_parts as UNet
from torchvision import transforms

def rgb_to_hsi(rgb):
    rgb = rgb / 255.0  # Normalize RGB values to [0, 1]
    
    # Calculate Intensity
    I = np.mean(rgb, axis=2)
    
    # Calculate Saturation
    min_rgb = np.min(rgb, axis=2)
    S = 1 - (min_rgb / (I + 1e-10))  # Avoid division by zero

    # Calculate Hue
    H = np.zeros(rgb.shape[0:2])
    num = 0.5 * ((rgb[..., 0] - rgb[..., 1]) + (rgb[..., 0] - rgb[..., 2]))
    den = np.sqrt((rgb[..., 0] - rgb[..., 1])**2 + (rgb[..., 0] - rgb[..., 2]) * (rgb[..., 1] - rgb[..., 2]))
    
    theta = np.arccos(num / (den + 1e-10))  # Avoid division by zero
    H[min_rgb == rgb[..., 1]] = theta[min_rgb == rgb[..., 1]]
    H[min_rgb == rgb[..., 2]] = 2 * np.pi - theta[min_rgb == rgb[..., 2]]
    H[min_rgb == rgb[..., 0]] = 0
    H = H / (2 * np.pi)  # Normalize Hue to [0, 1]
    return np.stack((H, S, I), axis=-1)

def hsi_to_rgb(hsi):
    H = hsi[..., 0] * 2 * np.pi  # Convert Hue back to [0, 2*pi]
    S = hsi[..., 1]
    I = hsi[..., 2]

    # Initialize RGB channels
    R = np.zeros(hsi.shape[0:2])
    G = np.zeros(hsi.shape[0:2])
    B = np.zeros(hsi.shape[0:2])

    for i in range(hsi.shape[0]):
        for j in range(hsi.shape[1]):
            if H[i, j] < 2 * np.pi / 3:
                R[i, j] = I[i, j] * (1 + S[i, j] * np.cos(H[i, j]) / np.cos(np.pi / 3 - H[i, j]))
                G[i, j] = I[i, j] * (1 + S[i, j] * (1 - np.cos(H[i, j]) / np.cos(np.pi / 3 - H[i, j])))
                B[i, j] = I[i, j] * (1 - S[i, j])
            elif H[i, j] < 4 * np.pi / 3:
                H[i, j] -= 2 * np.pi / 3
                R[i, j] = I[i, j] * (1 - S[i, j])
                G[i, j] = I[i, j] * (1 + S[i, j] * np.cos(H[i, j]) / np.cos(np.pi / 3 - H[i, j]))
                B[i, j] = I[i, j] * (1 + S[i, j] * (1 - np.cos(H[i, j]) / np.cos(np.pi / 3 - H[i, j])))
            else:
                H[i, j] -= 4 * np.pi / 3
                R[i, j] = I[i, j] * (1 + S[i, j] * (1 - np.cos(H[i, j]) / np.cos(np.pi / 3 - H[i, j])))
                G[i, j] = I[i, j] * (1 - S[i, j])
                B[i, j] = I[i, j] * (1 + S[i, j] * np.cos(H[i, j]) / np.cos(np.pi / 3 - H[i, j]))

    # Clip RGB values to [0, 255] and convert to uint8
    R = np.clip(R * 255, 0, 255).astype(np.uint8)
    G = np.clip(G * 255, 0, 255).astype(np.uint8)
    B = np.clip(B * 255, 0, 255).astype(np.uint8)
    return np.stack((R, G, B), axis=-1)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, inputs):
        # Convert inputs from RGB to HSI if inputs are images
        if inputs.ndim == 4:  # (N, C, H, W) format
            inputs = rgb_to_hsi(inputs)
        outputs = self.linear(inputs)
        if self.activation == 'relu':
            outputs = nn.ReLU(inplace=True)(outputs)
        return outputs

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding=(kernel_size - 1) // 2)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation == 'relu':
            outputs = nn.ReLU(inplace=True)(outputs)
        return outputs

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class StegaStampEncoder(nn.Module):
    def __init__(self):
        super(StegaStampEncoder, self).__init__()
        self.secret_dense = Dense(100, 7500, activation='relu', kernel_initializer='he_normal')

        self.conv1 = Conv2D(3, 32)
        self.conv2 = Conv2D(32, 32, strides=2)
        self.conv3 = Conv2D(32, 64, strides=2)
        self.conv4 = Conv2D(64, 128, strides=2)
        self.conv5 = Conv2D(128, 256, strides=2)
        self.up6 = Conv2D(256, 128)
        self.conv6 = Conv2D(256, 128)
        self.up7 = Conv2D(128, 64)
        self.conv7 = Conv2D(128, 64)
        self.up8 = Conv2D(64, 32)
        self.conv8 = Conv2D(64, 32)
        self.up9 = Conv2D(32, 32)
        self.conv9 = Conv2D(70, 32)
        self.residual = Conv2D(32, 3, kernel_size=1, activation=None)

    def forward(self, inputs):
        secrect, image = inputs
        image = rgb_to_hsi(image)  # Convert RGB to HSI
        secrect = secrect - 0.5
        image = image - 0.5

        secrect = self.secret_dense(secrect)
        secrect = secrect.reshape(-1, 3, 50, 50)
        secrect_enlarged = nn.Upsample(scale_factor=(8, 8))(secrect)

        inputs = torch.cat([secrect_enlarged, image], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.res
class StegaStampEncoderUnet(nn.Module):
    def __init__(self, bilinear=False):
        super(StegaStampEncoderUnet, self).__init__()
        self.secret_dense = Dense(100, 7500, activation='relu', kernel_initializer='he_normal')

        self.conv1 = nn.Conv2d(6, 6, 3, padding=1)  # Changed padding to 1 for 3x3 conv
        self.inc = UNet.DoubleConv(6, 64)
        self.down1 = UNet.Down(64, 128)
        self.down2 = UNet.Down(128, 256)
        self.DoubleConv = UNet.DoubleConv(256, 512)
        factor = 2 if bilinear else 1
        self.up1 = UNet.Up(512, 256 // factor, bilinear)
        self.up2 = UNet.Up(256, 128 // factor, bilinear)
        self.up3 = UNet.Up(128, 64 // factor, bilinear)
        self.outc = UNet.OutConv(64, 6)
        self.conv2 = nn.Conv2d(6, 3, kernel_size=3, padding=1)  # Changed kernel size and padding
        self.sig = nn.Sigmoid()

    def forward(self, inputs):
        secrect, image = inputs
        image = rgb_to_hsi(image)  # Convert RGB to HSI
        secrect = secrect - 0.5
        image = image - 0.5
        secrect = self.secret_dense(secrect)
        secrect = secrect.reshape(-1, 3, 50, 50)
        image = nn.functional.interpolate(image, scale_factor=(1 / 8, 1 / 8))
        inputs = torch.cat([secrect, image], dim=1)
        conv1 = self.conv1(inputs)
        x1 = self.inc(conv1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.DoubleConv(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        x = self.conv2(x)
        secrect_enlarged = nn.Upsample(scale_factor=(8, 8))(x)
        secrect_enlarged = self.sig(secrect_enlarged)
        return hsi_to_rgb(secrect_enlarged)


class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()
        self.localization = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(320000, 128, activation='relu'),
            nn.Linear(128, 6)
        )
        self.localization[-1].weight.data.fill_(0)
        self.localization[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, image):
        image = rgb_to_hsi(image)  # Convert RGB to HSI
        theta = self.localization(image)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        transformed_image = F.grid_sample(image, grid, align_corners=False)
        return hsi_to_rgb(transformed_image)



class StegaStampDecoder(nn.Module):
    def __init__(self, secret_size=100):
        super(StegaStampDecoder, self).__init__()
        self.secret_size = secret_size
        self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Use nn.Conv2d instead of Conv2D
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(21632, 512),  # Use nn.Linear instead of Dense
            nn.Linear(512, secret_size)  # Use nn.Linear instead of Dense
        )

    def forward(self, image):
        image = rgb_to_hsi(image)  # Convert RGB to HSI
        image = image - 0.5
        transformed_image = self.stn(image)
        decoded_output = self.decoder(transformed_image)
        return hsi_to_rgb(torch.sigmoid(decoded_output))


class StegaStampDecoderUnet(nn.Module):
    def __init__(self, secret_size=100):
        super(StegaStampDecoderUnet, self).__init__()
        self.secret_size = secret_size
        self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(21632, 512),
            nn.Linear(512, secret_size)
        )

    def forward(self, image):
        image = rgb_to_hsi(image)  # Convert RGB to HSI
        image = image - 0.5
        transformed_image = self.stn(image)
        decoded_output = self.decoder(transformed_image)
        return hsi_to_rgb(torch.sigmoid(decoded_output))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, image):
        image = rgb_to_hsi(image)  # Convert RGB to HSI
        x = image - 0.5
        x = self.model(x)
        output = torch.mean(x)
        residual = x  # Keep the residual for later use
        return output, hsi_to_rgb(residual)  # Return output and converted residual

def transform_net(encoded_image, args, global_step):
    sh = encoded_image.size()
    ramp_fn = lambda ramp: np.min([global_step / ramp, 1.])

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = utils.get_rnd_brightness_torch(rnd_bri, rnd_hue, args.batch_size)  # [batch_size, 3, 1, 1]
    jpeg_quality = 100. - torch.rand(1).item() * ramp_fn(args.jpeg_quality_ramp) * (100. - args.jpeg_quality)
    rnd_noise = torch.rand(1).item() * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand(1).item() * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    # Blur
    N_blur = 7
    f = utils.random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
                                 wmin_line=3)
    if args.cuda:
        f = f.cuda()
    encoded_image = F.conv2d(encoded_image, f, bias=None, padding=int((N_blur - 1) / 2))

    # Noise
    noise = torch.normal(mean=0, std=rnd_noise, size=encoded_image.size(), dtype=torch.float32)
    if args.cuda:
        noise = noise.cuda()
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # Contrast & Brightness
    contrast_scale = torch.FloatTensor(encoded_image.size(0)).uniform_(contrast_params[0], contrast_params[1])
    contrast_scale = contrast_scale.view(encoded_image.size(0), 1, 1, 1)
    if args.cuda:
        contrast_scale = contrast_scale.cuda()
        rnd_brightness = rnd_brightness.cuda()
    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # Saturation
    sat_weight = torch.FloatTensor([.3, .6, .1]).view(1, 3, 1, 1)
    if args.cuda:
        sat_weight = sat_weight.cuda()
    encoded_image_lum = torch.mean(encoded_image * sat_weight, dim=1, keepdim=True)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    # JPEG
    encoded_image = encoded_image.view([-1, 3, 400, 400])
    if not args.no_jpeg:
        encoded_image = utils.jpeg_compress_decompress(encoded_image, rounding=utils.round_only_at_0,
                                                       quality=jpeg_quality)

    return encoded_image


def get_secret_acc(secret_true, secret_pred):
    if secret_pred.device.type == 'cuda':
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - (torch.sum((correct_pred - secret_pred.size(1)) != 0).item() / correct_pred.size(0))
    bit_acc = torch.sum(correct_pred).item() / secret_pred.numel()
    return bit_acc, str_acc


def build_model(encoder, decoder, discriminator, lpips_fn, secret_input, image_input, l2_edge_gain,
                borders, secret_size, M, loss_scales, yuv_scales, args, global_step, writer):
    test_transform = transform_net(image_input, args, global_step)

    input_warped = torchgeometry.warp_perspective(image_input, M[:, 1, :, :], dsize=(400, 400), flags='bilinear')
    print("Line 325: input_warped min: {:.4f}, max: {:.4f}".format(input_warped.min().item(), input_warped.max().item()))

    mask_warped = torchgeometry.warp_perspective(torch.ones_like(input_warped), M[:, 1, :, :], dsize=(400, 400), flags='bilinear')
    print("Line 328: mask_warped min: {:.4f}, max: {:.4f}".format(mask_warped.min().item(), mask_warped.max().item()))

    input_warped += (1 - mask_warped) * image_input
    print("Line 331: input_warped after addition min: {:.4f}, max: {:.4f}".format(input_warped.min().item(), input_warped.max().item()))

    residual_warped = encoder((secret_input, input_warped))
    encoded_warped = residual_warped + input_warped

    residual = torchgeometry.warp_perspective(residual_warped, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
    print("Line 337: residual min: {:.4f}, max: {:.4f}".format(residual.min().item(), residual.max().item()))

    if borders == 'no_edge':
        encoded_image = image_input + residual
    elif borders == 'black':
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        print("Line 344: encoded_image (black) min: {:.4f}, max: {:.4f}".format(encoded_image.min().item(), encoded_image.max().item()))
        input_unwarped = torchgeometry.warp_perspective(image_input, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        print("Line 346: input_unwarped (black) min: {:.4f}, max: {:.4f}".format(input_unwarped.min().item(), input_unwarped.max().item()))
    elif borders.startswith('random'):
        mask = torchgeometry.warp_perspective(torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        encoded_image = residual_warped + input_unwarped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        print("Line 351: encoded_image (random) min: {:.4f}, max: {:.4f}".format(encoded_image.min().item(), encoded_image.max().item()))
        input_unwarped = torchgeometry.warp_perspective(input_warped, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        print("Line 353: input_unwarped (random) min: {:.4f}, max: {:.4f}".format(input_unwarped.min().item(), input_unwarped.max().item()))
    elif borders == 'white':
        mask = torchgeometry.warp_perspective(torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        print("Line 358: encoded_image (white) min: {:.4f}, max: {:.4f}".format(encoded_image.min().item(), encoded_image.max().item()))
        input_unwarped = torchgeometry.warp_perspective(input_warped, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        print("Line 360: input_unwarped (white) min: {:.4f}, max: {:.4f}".format(input_unwarped.min().item(), input_unwarped.max().item()))
    elif borders == 'image':
        mask = torchgeometry.warp_perspective(torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        print("Line 365: encoded_image (image) min: {:.4f}, max: {:.4f}".format(encoded_image.min().item(), encoded_image.max().item()))
        encoded_image += (1 - mask) * torch.roll(image_input, 1, 0)

    if borders == 'no_edge':
        D_output_real, _ = discriminator(image_input)
        D_output_fake, _ = discriminator(encoded_image)
        D_loss = torch.mean(D_output_fake - D_output_real)
    else:
        D_output_real, _ = discriminator(image_input)
        D_output_fake, _ = discriminator(encoded_image)
        D_loss = torch.mean(D_output_fake - D_output_real)

    # Log D_loss
    writer.add_scalar('D_loss', D_loss.item(), global_step)

    # Compute LPIPS loss
    lpips_loss = lpips_fn(test_transform, image_input)
    writer.add_scalar('LPIPS', lpips_loss.item(), global_step)

    # Total loss
    total_loss = D_loss + lpips_loss
    return total_loss	