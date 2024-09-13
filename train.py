import os
import yaml
import random
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image
from torch import optim
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import lpips
import time
from datetime import timedelta

import model
import utils
from dataset import Dataset  # Make sure this matches your dataset import
from kan_unet import KANU_Net  # Import KAN-based U-Net

CHECKPOINT_MARK_1 = 10_000
CHECKPOINT_MARK_2 = 1500
IMAGE_SIZE = 224  # Match this to your dataset size

def infoMessage0(string):
    print(f'[-----]: {string}')

infoMessage0('Opening settings file')
with open('cfg/setting.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

if not os.path.exists(args.checkpoints_path):
    os.makedirs(args.checkpoints_path)

if not os.path.exists(args.saved_models):
    os.makedirs(args.saved_models)

args.min_loss = float('inf')
args.min_secret_loss = float('inf')

def main():
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    log_path = os.path.join(args.logs_path, str(args.exp_name))
    writer = SummaryWriter(log_path)
    infoMessage0('Loading data')
    dataset = Dataset(args.train_path, args.secret_size, size=(IMAGE_SIZE, IMAGE_SIZE))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    encoder = KANU_Net(n_channels=3, n_classes=1)  # Adjust channels/classes as needed
    decoder = model.StegaStampDecoder(secret_size=args.secret_size)  # Assuming this is needed
    discriminator = model.Discriminator()  # Assuming this is needed
    lpips_alex = lpips.LPIPS(net="alex", verbose=False)

    args.cuda = torch.cuda.is_available()  # Check CUDA availability
    if args.cuda:
        infoMessage0('CUDA = True')
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator = discriminator.cuda()
        lpips_alex.cuda()

    d_vars = discriminator.parameters()
    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    total_steps = len(dataset) // args.batch_size + 1
    global_step = 0

    start_time = time.time()

    while global_step < args.num_steps:
        for image_input, secret_input in dataloader:  # Use the dataloader directly
            step_start_time = time.time()
            
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()

            no_im_loss = global_step < args.no_im_loss_steps
            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp,
                                    args.secret_loss_scale)

            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran

            global_step += 1
            Ms = torch.eye(3, 3)
            Ms = torch.stack((Ms, Ms), 0)
            Ms = torch.stack((Ms, Ms, Ms, Ms), 0)
            if args.cuda:
                Ms = Ms.cuda()

            image_output = encoder((secret_input, image_input))  # Pass secret_input and image_input as tuple

            loss_scales = [l2_loss_scale, 0, secret_loss_scale, 0]
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]
            loss, secret_loss, D_loss, bit_acc, str_acc = model.build_model(
                encoder, decoder, discriminator, lpips_alex,
                secret_input, image_input,
                args.l2_edge_gain, args.borders,
                args.secret_size, Ms, loss_scales,
                yuv_scales, args, global_step, writer
            )
            if no_im_loss:
                optimize_secret_loss.zero_grad()
                secret_loss.backward()
                optimize_secret_loss.step()
            else:
                optimize_loss.zero_grad()
                loss.backward()
                optimize_loss.step()
                if not args.no_gan:
                    optimize_dis.zero_grad()
                    optimize_dis.step()

            step_time = time.time() - step_start_time
            total_time_elapsed = time.time() - start_time
            steps_remaining = args.num_steps - global_step
            eta_seconds = (total_time_elapsed / global_step) * steps_remaining if global_step > 0 else 0
            eta = timedelta(seconds=int(eta_seconds))
            
            if global_step % 10 == 0:
                writer.add_scalars('Loss values', {'loss': loss.item(), 'secret loss': secret_loss.item(),
                                                   'D_loss': D_loss.item()})
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
