import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast  # PyTorch AMP
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from cp_dataset_test import CPDatasetTest
from networks import ConditionGenerator, VGGLoss, save_checkpoint, make_grid, make_grid_3d
from network_generator import SPADEGenerator, MultiscaleDiscriminator, GANLoss, Projected_GANs_Loss, set_requires_grad
from sync_batchnorm import DataParallelWithCallback
from utils import create_network
import sys
from tqdm import tqdm
import numpy as np
from torch.utils.data import Subset
from torchvision.transforms import transforms
import eval_models as models
import torchgeometry as tgm
from pg_modules.discriminator import ProjectedDiscriminator
import cv2

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"[*] Checkpoint not found at {checkpoint_path}")
        return
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    # Adjust conv_img.weight and conv_img.bias for shape mismatch
    if 'conv_img.weight' in state_dict and state_dict['conv_img.weight'].shape[0] == 4:
        state_dict['conv_img.weight'] = state_dict['conv_img.weight'][:3, :, :, :]  # Truncate to 3 channels
        state_dict['conv_img.bias'] = state_dict['conv_img.bias'][:3]  # Truncate to 3 channels
    model.load_state_dict(state_dict, strict=False)
    print(f"[*] Load Success! log: {model.load_state_dict(state_dict, strict=False)}")

def remove_overlap(seg_out, warped_cm):
    assert len(warped_cm.shape) == 4
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use PyTorch AMP')

    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=768)
    parser.add_argument("--fine_height", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, help='condition generator checkpoint')
    parser.add_argument('--gen_checkpoint', type=str, default='', help='gen checkpoint')
    parser.add_argument('--dis_checkpoint', type=str, default='', help='dis checkpoint')

    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=1000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('--resume', action='store_true', help='resume training from the last checkpoint')
    
    # test
    parser.add_argument("--lpips_count", type=int, default=1000)
    parser.add_argument("--test_datasetting", default="paired")
    parser.add_argument("--test_dataroot", default="./data/")
    parser.add_argument("--test_data_list", default="test_pairs.txt")

    # Hyper-parameters
    parser.add_argument('--G_lr', type=float, default=0.00005, help='initial learning rate for adam')  # Reduced
    parser.add_argument('--D_lr', type=float, default=0.0002, help='initial learning rate for adam')   # Reduced

    # SEAN-related hyper-parameters
    parser.add_argument('--GMM_const', type=float, default=None, help='constraint for GMM module')
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of input label classes without unknown class')
    parser.add_argument('--gen_semantic_nc', type=int, default=7, help='# of input label classes without unknown class')
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='weight for image-level l1 loss')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
    
    # D
    parser.add_argument('--n_layers_D', type=int, default=3, help='# layers in each discriminator')
    parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
    parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
    
    # G & D arch-related
    parser.add_argument("--composition_mask", action='store_true', help='use composition mask')

    # Training
    parser.add_argument('--occlusion', action='store_true')
    # tocg
    parser.add_argument('--cond_G_ngf', type=int, default=96)
    parser.add_argument("--cond_G_input_width", type=int, default=192)
    parser.add_argument("--cond_G_input_height", type=int, default=256)
    parser.add_argument('--cond_G_num_layers', type=int, default=5)
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")

    # New arguments for selective layer freezing and last layer control
    parser.add_argument('--freeze_tocg_layers', type=int, default=0, help='number of layers to freeze in tocg from the start')
    parser.add_argument('--freeze_gen_layers', type=int, default=0, help='number of layers to freeze in generator from the start')
    parser.add_argument('--last_layer_mode', type=str, default='train', choices=['train', 'half', 'freeze'],
                        help='Mode for the last layer: train (full training), half (half parameters frozen), freeze (fully frozen)')

    opt = parser.parse_args()

    # Set GPU IDs
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
        "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
        % (opt.batch_size, len(opt.gpu_ids))

    return opt

def apply_layer_freezing(model, num_layers_to_freeze, last_layer_mode):
    """Apply selective layer freezing and handle the last layer based on mode."""
    children = list(model.named_children())
    total_layers = len(children)
    
    # Freeze specified layers from the start
    for i, (name, module) in enumerate(children):
        if i < num_layers_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
    
    # Handle the last layer based on mode
    if total_layers > 0 and last_layer_mode != 'train':
        last_name, last_module = children[-1]
        if last_layer_mode == 'freeze':
            for param in last_module.parameters():
                param.requires_grad = False
        elif last_layer_mode == 'half':
            # Freeze half of the parameters in the last layer
            params = list(last_module.parameters())
            half_idx = len(params) // 2
            for param in params[:half_idx]:
                param.requires_grad = False
            for param in params[half_idx:]:
                param.requires_grad = True

def train(opt, train_loader, test_loader, tocg, generator, discriminator, model):
    """
    Train Generator and Condition Generator
    """
    # Model
    tocg.cuda()
    tocg.train()
    generator.train()
    discriminator.train()
    if not opt.composition_mask:
        discriminator.feature_network.requires_grad_(False)
    discriminator.cuda()
    model.eval()

    # Apply layer freezing
    apply_layer_freezing(tocg, opt.freeze_tocg_layers, opt.last_layer_mode)
    apply_layer_freezing(generator, opt.freeze_gen_layers, opt.last_layer_mode)

    # Criterion
    criterionGAN = None
    if opt.composition_mask:
        criterionGAN = GANLoss('hinge', tensor=torch.cuda.FloatTensor)
    else:
        criterionGAN = Projected_GANs_Loss(tensor=torch.cuda.FloatTensor)
    
    criterionL1 = nn.L1Loss()
    criterionFeat = nn.L1Loss()
    criterionVGG = VGGLoss()

    # Optimizer
    optimizer_gen = torch.optim.Adam(
        list(generator.parameters()) + list(tocg.parameters()),
        lr=opt.G_lr, betas=(0.0, 0.9)
    )
    scheduler_gen = torch.optim.lr_scheduler.LambdaLR(optimizer_gen, lr_lambda=lambda step: 1.0 -
            max(0, step * 1000 + opt.load_step - opt.keep_step) / float(opt.decay_step + 1))
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=opt.D_lr, betas=(0.0, 0.9))
    scheduler_dis = torch.optim.lr_scheduler.LambdaLR(optimizer_dis, lr_lambda=lambda step: 1.0 -
            max(0, step * 1000 + opt.load_step - opt.keep_step) / float(opt.decay_step + 1))

    # Initialize PyTorch AMP
    scaler = GradScaler(enabled=opt.fp16)

    if len(opt.gpu_ids) > 0:
        tocg = DataParallelWithCallback(tocg, device_ids=opt.gpu_ids)
        generator = DataParallelWithCallback(generator, device_ids=opt.gpu_ids)
        discriminator = DataParallelWithCallback(discriminator, device_ids=opt.gpu_ids)
        criterionGAN = DataParallelWithCallback(criterionGAN, device_ids=opt.gpu_ids)
        criterionFeat = DataParallelWithCallback(criterionFeat, device_ids=opt.gpu_ids)
        criterionVGG = DataParallelWithCallback(criterionVGG, device_ids=opt.gpu_ids)
        criterionL1 = DataParallelWithCallback(criterionL1, device_ids=opt.gpu_ids)
        
    upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss = gauss.cuda()

    checkpoint_path = os.path.join(opt.checkpoint_dir, opt.name, 'checkpoint.pth')
    if opt.resume:
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            opt.load_step = checkpoint['step']
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            tocg.load_state_dict(checkpoint['tocg_state_dict'])
            optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
            optimizer_dis.load_state_dict(checkpoint['optimizer_dis_state_dict'])
            scheduler_gen.load_state_dict(checkpoint['scheduler_gen_state_dict'])
            scheduler_dis.load_state_dict(checkpoint['scheduler_dis_state_dict'])
        else:
            print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")

    for step in tqdm(range(opt.load_step, opt.keep_step + opt.decay_step)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # Input
        agnostic = inputs['agnostic'].cuda()
        parse_GT = inputs['parse'].cuda()
        pose = inputs['densepose'].cuda()
        parse_cloth = inputs['parse_cloth'].cuda()
        parse_agnostic = inputs['parse_agnostic'].cuda()
        pcm = inputs['pcm'].cuda()
        cm = inputs['cloth_mask']['paired'].cuda()
        c_paired = inputs['cloth']['paired'].cuda()
        
        # Target
        im = inputs['image'].cuda()

        # Warping Cloth
        with autocast(enabled=opt.fp16):
            pre_clothes_mask_down = F.interpolate(cm, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
            input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
            clothes_down = F.interpolate(c_paired, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
            densepose_down = F.interpolate(pose, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
            
            input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
            input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

            flow_list_taco, fake_segmap, warped_cloth_paired_taco, warped_clothmask_paired_taco, flow_list_tvob, warped_cloth_paired_tvob, warped_clothmask_paired_tvob = tocg(input1, input2)
            
            warped_clothmask_paired_taco_onehot = torch.FloatTensor((warped_clothmask_paired_taco.detach().cpu().numpy() > 0.5).astype(float)).cuda()
            
            cloth_mask = torch.ones_like(fake_segmap)
            cloth_mask[:,3:4, :, :] = warped_clothmask_paired_taco
            fake_segmap = fake_segmap * cloth_mask
                
            N, _, iH, iW = c_paired.shape
            N, flow_iH, flow_iW, _ = flow_list_tvob[-1].shape

            flow_tvob = F.interpolate(flow_list_tvob[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
            flow_tvob_norm = torch.cat([flow_tvob[:, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_tvob[:, :, :, 1:2] / ((flow_iH - 1.0) / 2.0)], 3)

            grid = make_grid(N, iH, iW)
            grid_3d = make_grid_3d(N, iH, iW)

            warped_grid_tvob = grid + flow_tvob_norm
            warped_cloth_tvob = F.grid_sample(c_paired, warped_grid_tvob, padding_mode='border')
            warped_clothmask_tvob = F.grid_sample(cm, warped_grid_tvob, padding_mode='border')

            flow_taco = F.interpolate(flow_list_taco[-1].permute(0, 4, 1, 2, 3), size=(2,iH,iW), mode='trilinear').permute(0, 2, 3, 4, 1)
            flow_taco_norm = torch.cat([flow_taco[:, :, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_taco[:, :, :, :, 1:2] / ((flow_iH - 1.0) / 2.0), flow_taco[:, :, :, :, 2:3]], 4)
            warped_cloth_tvob = warped_cloth_tvob.unsqueeze(2)
            warped_cloth_paired_taco = F.grid_sample(torch.cat((warped_cloth_tvob, torch.zeros_like(warped_cloth_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
            warped_cloth_paired_taco = warped_cloth_paired_taco[:,:,0,:,:]

            warped_clothmask_tvob = warped_clothmask_tvob.unsqueeze(2)
            warped_clothmask_taco = F.grid_sample(torch.cat((warped_clothmask_tvob, torch.zeros_like(warped_clothmask_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
            warped_clothmask_taco = warped_clothmask_taco[:,:,0,:,:]

            fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear'))
            fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

            if opt.occlusion:
                warped_clothmask_taco = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask_taco)
                warped_cloth_paired_taco = warped_cloth_paired_taco * warped_clothmask_taco + torch.ones_like(warped_cloth_paired_taco) * (1-warped_clothmask_taco)
                warped_cloth_paired_taco = warped_cloth_paired_taco.detach()
                
            old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
            old_parse.scatter_(1, fake_parse, 1.0)

            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    parse[:, i] += old_parse[:, label]
                    
            parse = parse.detach()

            # Train the generator and tocg
            G_losses = {}
            if opt.composition_mask:
                output_paired_rendered, output_paired_comp = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)
                output_paired_comp1 = output_paired_comp * warped_clothmask_taco
                output_paired_comp = parse[:,2:3,:,:] * output_paired_comp1
                output_paired = warped_cloth_paired_taco * output_paired_comp + output_paired_rendered * (1 - output_paired_comp)

                fake_concat = torch.cat((parse, output_paired_rendered), dim=1)
                real_concat = torch.cat((parse, im), dim=1)
                pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))

                pred_fake = []
                pred_real = []
                for p in pred:
                    pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                    pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
                
                G_losses['GAN'] = criterionGAN(pred_fake, True, for_discriminator=False)

                num_D = len(pred_fake)
                GAN_Feat_loss = torch.cuda.FloatTensor(len(opt.gpu_ids)).zero_()
                for i in range(num_D):
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):
                        unweighted_loss = criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * opt.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss

                G_losses['VGG'] = criterionVGG(output_paired, im) * opt.lambda_vgg + criterionVGG(output_paired_rendered, im) * opt.lambda_vgg
                G_losses['L1'] = criterionL1(output_paired_rendered, im) * opt.lambda_l1 + criterionL1(output_paired, im) * opt.lambda_l1
                G_losses['Composition_Mask'] = torch.mean(torch.abs(1 - output_paired_comp))

                loss_gen = sum(G_losses.values()).mean()

            else:
                set_requires_grad(discriminator, False)
                output_paired = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)

                try:
                    pred_fake, feats_fake = discriminator(output_paired)
                    pred_real, feats_real = discriminator(im)
                except Exception as e:
                    print(f"Error in discriminator forward pass: {e}")
                    raise

                G_losses['GAN'] = criterionGAN(pred_fake, True, for_discriminator=False) * 0.5

                num_D = len(feats_fake)
                GAN_Feat_loss = torch.cuda.FloatTensor(len(opt.gpu_ids)).zero_()
                for i in range(num_D):
                    num_intermediate_outputs = len(feats_fake[i])
                    for j in range(num_intermediate_outputs):
                        unweighted_loss = criterionFeat(feats_fake[i][j], feats_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * opt.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss

                G_losses['VGG'] = criterionVGG(output_paired, im) * opt.lambda_vgg
                G_losses['L1'] = criterionL1(output_paired, im) * opt.lambda_l1

                loss_gen = sum(G_losses.values()).mean()

        optimizer_gen.zero_grad()
        with autocast(enabled=opt.fp16):
            loss_gen = sum(G_losses.values()).mean()
        scaler.scale(loss_gen).backward()
        torch.nn.utils.clip_grad_norm_(list(generator.parameters()) + list(tocg.parameters()), max_norm=1.0)
        scaler.step(optimizer_gen)

        # Train the discriminator
        D_losses = {}
        with autocast(enabled=opt.fp16):
            if opt.composition_mask:
                with torch.no_grad():            
                    output_paired_rendered, output_comp = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)
                    output_comp1 = output_comp * warped_clothmask_taco
                    output_comp = parse[:,2:3,:,:] * output_comp1
                    output = warped_cloth_paired_taco * output_comp + output_paired_rendered * (1 - output_comp)
                    output_comp = output_comp.detach()
                    output = output.detach()
                    output_comp.requires_grad_()
                    output.requires_grad_()

                fake_concat = torch.cat((parse, output_paired_rendered), dim=1)
                real_concat = torch.cat((parse, im), dim=1)
                pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))

                pred_fake = []
                pred_real = []
                for p in pred:
                    pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                    pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
                
                D_losses['D_Fake'] = criterionGAN(pred_fake, False, for_discriminator=True)
                D_losses['D_Real'] = criterionGAN(pred_real, True, for_discriminator=True)
                
                loss_dis = sum(D_losses.values()).mean()

            else:
                set_requires_grad(discriminator, True)
                discriminator.module.feature_network.requires_grad_(False)

                with torch.no_grad():
                    output = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)
                    output = output.detach()
                    output.requires_grad_()

                try:
                    pred_fake, _ = discriminator(output)
                    pred_real, _ = discriminator(im)
                except Exception as e:
                    print(f"Error in discriminator forward pass: {e}")
                    raise

                D_losses['D_Fake'] = criterionGAN(pred_fake, False, for_discriminator=True)
                D_losses['D_Real'] = criterionGAN(pred_real, True, for_discriminator=True)

                loss_dis = sum(D_losses.values()).mean()

        optimizer_dis.zero_grad()
        scaler.scale(loss_dis).backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        scaler.step(optimizer_dis)
        scaler.update()
        
        if not opt.composition_mask:
            set_requires_grad(discriminator, False)

        if (step+1) % 100 == 0:
            a_0 = im.cuda()[0]
            b_0 = output.cuda()[0]
            c_0 = warped_cloth_paired_taco.cuda()[0]
            combine = torch.cat((a_0, b_0, c_0), dim=2)
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite('sample_fs_toig/'+str(step)+'.jpg',bgr)

        # Evaluate the generator
        if (step + 1) % opt.lpips_count == 0:
            generator.eval()
            tocg.eval()
            T2 = transforms.Compose([transforms.Resize((128, 128))])
            lpips_list = []
            avg_distance = 0.0
            
            with torch.no_grad():
                print("LPIPS")
                for i in tqdm(range(500)):
                    inputs = test_loader.next_batch()
                    agnostic = inputs['agnostic'].cuda()
                    parse_GT = inputs['parse'].cuda()
                    pose = inputs['densepose'].cuda()
                    parse_cloth = inputs['parse_cloth'].cuda()
                    parse_agnostic = inputs['parse_agnostic'].cuda()
                    pcm = inputs['pcm'].cuda()
                    cm = inputs['cloth_mask']['paired'].cuda()
                    c_paired = inputs['cloth']['paired'].cuda()
                    im = inputs['image'].cuda()
                                
                    with autocast(enabled=opt.fp16):
                        pre_clothes_mask_down = F.interpolate(cm, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
                        input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
                        clothes_down = F.interpolate(c_paired, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
                        densepose_down = F.interpolate(pose, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
                        
                        input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
                        input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

                        flow_list_taco, fake_segmap, warped_cloth_paired_taco, warped_clothmask_paired_taco, flow_list_tvob, warped_cloth_paired_tvob, warped_clothmask_paired_tvob = tocg(input1, input2)
                        
                        warped_clothmask_paired_taco_onehot = torch.FloatTensor((warped_clothmask_paired_taco.detach().cpu().numpy() > 0.5).astype(float)).cuda()
                        
                        cloth_mask = torch.ones_like(fake_segmap)
                        cloth_mask[:,3:4, :, :] = warped_clothmask_paired_taco
                        fake_segmap = fake_segmap * cloth_mask
                            
                        N, _, iH, iW = c_paired.shape
                        N, flow_iH, flow_iW, _ = flow_list_tvob[-1].shape

                        flow_tvob = F.interpolate(flow_list_tvob[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                        flow_tvob_norm = torch.cat([flow_tvob[:, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_tvob[:, :, :, 1:2] / ((flow_iH - 1.0) / 2.0)], 3)

                        grid = make_grid(N, iH, iW)
                        grid_3d = make_grid_3d(N, iH, iW)

                        warped_grid_tvob = grid + flow_tvob_norm
                        warped_cloth_tvob = F.grid_sample(c_paired, warped_grid_tvob, padding_mode='border')
                        warped_clothmask_tvob = F.grid_sample(cm, warped_grid_tvob, padding_mode='border')

                        flow_taco = F.interpolate(flow_list_taco[-1].permute(0, 4, 1, 2, 3), size=(2, iH, iW), mode='trilinear').permute(0, 2, 3, 4, 1)
                        flow_taco_norm = torch.cat([flow_taco[:, :, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_taco[:, :, :, :, 1:2] / ((flow_iH - 1.0) / 2.0), flow_taco[:, :, :, :, 2:3]], 4)
                        warped_cloth_tvob = warped_cloth_tvob.unsqueeze(2)
                        warped_cloth_paired_taco = F.grid_sample(torch.cat((warped_cloth_tvob, torch.zeros_like(warped_cloth_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
                        warped_cloth_paired_taco = warped_cloth_paired_taco[:,:,0,:,:]

                        warped_clothmask_tvob = warped_clothmask_tvob.unsqueeze(2)
                        warped_clothmask_taco = F.grid_sample(torch.cat((warped_clothmask_tvob, torch.zeros_like(warped_clothmask_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
                        warped_clothmask_taco = warped_clothmask_taco[:,:,0,:,:]

                        fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear'))
                        fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

                        if opt.occlusion:
                            warped_clothmask_taco = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask_taco)
                            warped_cloth_paired_taco = warped_cloth_paired_taco * warped_clothmask_taco + torch.ones_like(warped_cloth_paired_taco) * (1-warped_clothmask_taco)
                            warped_cloth_paired_taco = warped_cloth_paired_taco.detach()
                            
                        old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
                        old_parse.scatter_(1, fake_parse, 1.0)

                        labels = {
                            0:  ['background',  [0]],
                            1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                            2:  ['upper',       [3]],
                            3:  ['hair',        [1]],
                            4:  ['left_arm',    [5]],
                            5:  ['right_arm',   [6]],
                            6:  ['noise',       [12]]
                        }
                        parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
                        for i in range(len(labels)):
                            for label in labels[i][1]:
                                parse[:, i] += old_parse[:, label]
                                
                        parse = parse.detach()
                    
                        if opt.composition_mask:
                            output_paired_rendered, output_paired_comp = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)
                            output_paired_comp1 = output_paired_comp * warped_clothmask_taco
                            output_paired_comp = parse[:,2:3,:,:] * output_paired_comp1
                            output_paired = warped_cloth_paired_taco * output_paired_comp + output_paired_rendered * (1 - output_paired_comp)
                        else:
                            output_paired = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)

                        avg_distance += model.forward(T2(im), T2(output_paired))
                    
            avg_distance = avg_distance / 500
            print(f"LPIPS: {avg_distance}")
            generator.train()
            tocg.train()

        if (step + 1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print("step: %8d, time: %.3f, G_loss: %.4f, G_adv_loss: %.4f, D_loss: %.4f, D_fake_loss: %.4f, D_real_loss: %.4f"
                  % (step + 1, t, loss_gen.item(), G_losses['GAN'].mean().item(), loss_dis.item(),
                     D_losses['D_Fake'].mean().item(), D_losses['D_Real'].mean().item()), flush=True)

        if (step + 1) % opt.save_count == 0:
            checkpoint = {
                'step': step + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'tocg_state_dict': tocg.state_dict(),
                'optimizer_gen_state_dict': optimizer_gen.state_dict(),
                'optimizer_dis_state_dict': optimizer_dis.state_dict(),
                'scheduler_gen_state_dict': scheduler_gen.state_dict(),
                'scheduler_dis_state_dict': scheduler_dis.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

        if (step + 1) % 1000 == 0:
            scheduler_gen.step()
            scheduler_dis.step()

def main():
    opt = get_opt()
    print(opt)
    print("Start to train %s!" % opt.name)

    os.makedirs('sample_fs_toig', exist_ok=True)
    os.makedirs(os.path.join(opt.checkpoint_dir, opt.name), exist_ok=True)

    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)
    
    opt.batch_size = 1
    opt.dataroot = opt.test_dataroot
    opt.datamode = 'test'
    opt.data_list = opt.test_data_list
    test_dataset = CPDatasetTest(opt)
    test_dataset = Subset(test_dataset, np.arange(500))
    test_loader = CPDataLoader(opt, test_dataset)
    
    input1_nc = 4  # Match checkpoint expectation
    input2_nc = opt.semantic_nc + 3
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=13, ngf=opt.cond_G_ngf, norm_layer=nn.BatchNorm2d, num_layers=opt.cond_G_num_layers)
    load_checkpoint(tocg, opt.tocg_checkpoint)

    generator = SPADEGenerator(opt, 3+3+3)  # 9 channels: agnostic + pose + warped cloth
    generator.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        generator.cuda()
    generator.init_weights(opt.init_type, opt.init_variance)

    discriminator = None
    if opt.composition_mask:
        discriminator = create_network(MultiscaleDiscriminator, opt)
    else:
        discriminator = ProjectedDiscriminator(interp224=False)

    model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)

    if opt.gen_checkpoint and os.path.exists(opt.gen_checkpoint):
        load_checkpoint(generator, opt.gen_checkpoint)
    if opt.dis_checkpoint and os.path.exists(opt.dis_checkpoint):
        load_checkpoint(discriminator, opt.dis_checkpoint)

    train(opt, train_loader, test_loader, tocg, generator, discriminator, model)

    save_checkpoint(generator, os.path.join(opt.checkpoint_dir, opt.name, 'gen_model_final.pth'))
    save_checkpoint(discriminator, os.path.join(opt.checkpoint_dir, opt.name, 'dis_model_final.pth'))
    save_checkpoint(tocg, os.path.join(opt.checkpoint_dir, opt.name, 'tocg_model_final.pth'))

    print("Finished training %s!" % opt.name)

if __name__ == "__main__":
    main()