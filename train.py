import os
import yaml
import random
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image, ImageOps
from torch import optim
import torch
import lpips
import time
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fastkanconv import FastKANConvLayer  # Import the Kan-based convolutional network
from kan_unet import KANU_Net  # Import the Kan-based U-Net model
from dataset import StegaData  # Assuming the dataset remains unchanged
import utils

# Constants
CHECKPOINT_MARK_1 = 10_000
CHECKPOINT_MARK_2 = 1500
IMAGE_SIZE = 400

# Helper function for logging
def infoMessage0(string):
    print(f'[-----]: {string}')

# Load settings from yaml config
infoMessage0('opening settings file')
with open('cfg/setting.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

if not os.path.exists(args.checkpoints_path):
    os.makedirs(args.checkpoints_path)

if not os.path.exists(args.saved_models):
    os.makedirs(args.saved_models)

args.min_loss = float('inf')
args.min_secret_loss = float('inf')

# Main training function
def main():
    # Seed settings for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    log_path = os.path.join(args.logs_path, str(args.exp_name))
    writer = SummaryWriter(log_path)
    
    # Load dataset
    infoMessage0('Loading data')
    dataset = StegaData(args.train_path, args.secret_size, size=(IMAGE_SIZE, IMAGE_SIZE))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Define models using KANU_Net for both encoder and decoder
    encoder = KANU_Net(n_channels=3, n_classes=3)  # Assuming 3 input/output channels for RGB images
    decoder = KANU_Net(n_channels=args.secret_size, n_classes=3)  # Secret size as input, RGB output

    discriminator = FastKANConvLayer(in_channels=3, out_channels=1)  # Replace Discriminator with FastKANConvNet
    lpips_alex = lpips.LPIPS(net="alex", verbose=False)
    
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        infoMessage0('Using CUDA')
        encoder, decoder, discriminator, lpips_alex = encoder.cuda(), decoder.cuda(), discriminator.cuda(), lpips_alex.cuda()

    # Optimizers for different model parts
    d_vars = discriminator.parameters()
    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    # Training loop variables
    total_steps = len(dataset) // args.batch_size + 1
    global_step = 0
    start_time = time.time()

    while global_step < args.num_steps:
        for image_input, secret_input in dataloader:
            step_start_time = time.time()
            
            if args.cuda:
                image_input, secret_input = image_input.cuda(), secret_input.cuda()

            # Set random transformations and loss scaling factors
            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp, args.secret_loss_scale)

            global_step += 1
            
            # Apply the encoder and decoder models
            image_output = encoder(image_input)
            decoded_secret = decoder(secret_input)

            # Calculate losses
            loss, secret_loss, D_loss, bit_acc, str_acc = utils.build_model(
                encoder, decoder, discriminator, lpips_alex, secret_input, image_input, args.l2_edge_gain,
                args.borders, args.secret_size, None, [l2_loss_scale, 0, secret_loss_scale, 0], 
                [args.y_scale, args.u_scale, args.v_scale], args, global_step, writer
            )

            # Optimization steps
            if global_step < args.no_im_loss_steps:
                optimize_secret_loss.zero_grad()
                secret_loss.backward()
                optimize_secret_loss.step()
            else:
                optimize_loss.zero_grad()
                loss.backward()
                optimize_loss.step()
                if not args.no_gan:
                    optimize_dis.zero_grad()
                    D_loss.backward()
                    optimize_dis.step()

            # Logging and checkpoint saving
            step_time = time.time() - step_start_time
            total_time_elapsed = time.time() - start_time
            steps_remaining = args.num_steps - global_step
            eta_seconds = (total_time_elapsed / global_step) * steps_remaining if global_step > 0 else 0
            eta = timedelta(seconds=int(eta_seconds))
            
            if global_step % 10 == 0:
                writer.add_scalars('Loss values', {'loss': loss.item(), 'secret loss': secret_loss.item(), 'D_loss loss': D_loss.item()})
            if global_step % 100 == 0:
                print(f"Step: {global_step}, Time per Step: {step_time:.2f} seconds, ETA: {eta}, Loss = {loss:.4f}")
            
            if global_step % CHECKPOINT_MARK_1 == 0:
                torch.save(encoder.state_dict(), os.path.join(args.saved_models, "encoder.pth"))
                torch.save(decoder.state_dict(), os.path.join(args.saved_models, "decoder.pth"))

            if global_step > CHECKPOINT_MARK_2:
                if loss < args.min_loss:
                    args.min_loss = loss
                    torch.save(encoder.state_dict(), os.path.join(args.checkpoints_path, "encoder_best_total_loss.pth"))
                    torch.save(decoder.state_dict(), os.path.join(args.checkpoints_path, "decoder_best_total_loss.pth"))
            if global_step > CHECKPOINT_MARK_1:
                if secret_loss < args.min_secret_loss:
                    args.min_secret_loss = secret_loss
                    torch.save(encoder.state_dict(), os.path.join(args.checkpoints_path, "encoder_best_secret_loss.pth"))
                    torch.save(decoder.state_dict(), os.path.join(args.checkpoints_path, "decoder_best_secret_loss.pth"))

    writer.close()
    torch.save(encoder.state_dict(), os.path.join(args.saved_models, "encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(args.saved_models, "decoder.pth"))

if __name__ == '__main__':
    main()

