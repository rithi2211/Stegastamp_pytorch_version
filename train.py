import os
import yaml
import random
import model  # This should now refer to your KANU_Net model
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image, ImageOps
from torch import optim
import utils
from dataset import StegaData
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import lpips
import time
from datetime import datetime, timedelta
from kanu_net import KANU_Net  # Import the KAN-based U-Net model

CHECKPOINT_MARK_1 = 10_000
CHECKPOINT_MARK_2 = 1500
IMAGE_SIZE = 400

def infoMessage0(string):
    print(f'[-----]: {string}')

infoMessage0('opening settings file')
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
    dataset = StegaData(args.train_path, args.secret_size, size=(IMAGE_SIZE, IMAGE_SIZE))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Replace StegaStampEncoder with KANU_Net for the new U-Net model
    kanu_net = KANU_Net(n_channels=3, n_classes=1, bilinear=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    discriminator = model.Discriminator()  # If you're keeping the discriminator
    lpips_alex = lpips.LPIPS(net="alex", verbose=False)
    
    args.cuda = torch.cuda.is_available()  # Check CUDA availability
    if args.cuda:
        infoMessage0('cuda = True')
        kanu_net = kanu_net.cuda()
        discriminator = discriminator.cuda()
        lpips_alex.cuda()

    d_vars = discriminator.parameters()
    g_vars = [{'params': kanu_net.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

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

            # Forward pass using the KANU_Net
            image_output = kanu_net(image_input)

            loss_scales = [l2_loss_scale, 0, secret_loss_scale, 0]
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]
            loss, secret_loss, D_loss, bit_acc, str_acc = model.build_model(kanu_net, discriminator, lpips_alex,
                                                                            secret_input, image_input,
                                                                            args.l2_edge_gain, args.borders,
                                                                            args.secret_size, Ms, loss_scales,
                                                                            yuv_scales, args, global_step, writer)
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
                                                   'D_loss loss': D_loss.item()})
            if global_step % 100 == 0:
                print(f"Step: {global_step}, Time per Step: {step_time:.2f} seconds, ETA: {eta}, Loss = {loss:.4f}")
            
            if global_step % CHECKPOINT_MARK_1 == 0:
                torch.save(kanu_net, os.path.join(args.saved_models, "kanu_net.pth"))

            if global_step > CHECKPOINT_MARK_2:
                if loss < args.min_loss:
                    args.min_loss = loss
                    torch.save(kanu_net, os.path.join(args.checkpoints_path, "kanu_net_best_total_loss.pth"))
            if global_step > CHECKPOINT_MARK_1:
                if secret_loss < args.min_secret_loss:
                    args.min_secret_loss = secret_loss
                    torch.save(kanu_net, os.path.join(args.checkpoints_path, "kanu_net_best_secret_loss.pth"))

    writer.close()
    torch.save(kanu_net, os.path.join(args.saved_models, "kanu_net.pth"))

if __name__ == '__main__':
    main()
