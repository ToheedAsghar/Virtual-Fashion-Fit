# train_generator.py (Modified for Fine-tuning, Freezing, LPIPS)

import torch
import torch.nn as nn
from torch.nn import functional as F

import argparse
import os
import time
import copy # For deepcopying opt

# Use the modified CPDataset for both training and testing
from cp_dataset import CPDataset, CPDataLoader 

from networks import ConditionGenerator, VGGLoss, load_checkpoint, save_checkpoint, make_grid, make_grid_3d
from network_generator import SPADEGenerator, MultiscaleDiscriminator, GANLoss, Projected_GANs_Loss, set_requires_grad

from utils import create_network 
import sys
from tqdm import tqdm

import numpy as np
from torch.utils.data import Subset, DataLoader # Standard DataLoader
from torchvision import transforms as T_torchvision 
import eval_models as models
import torchgeometry as tgm # Keep for GaussianBlur if tgm is available

from pg_modules.discriminator import ProjectedDiscriminator
import cv2

# Apex for FP16 training (optional)
try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    print("Warning: Apex not found. FP16 training will be disabled.")


def remove_overlap(seg_out, warped_cm):
    assert len(warped_cm.shape) == 4
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, default="shalwar_qameez_gen_finetune")
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=4) 
    parser.add_argument('--fp16', action='store_true', help='Use NVIDIA Apex AMP for mixed-precision training.')

    # Dataset parameters
    parser.add_argument("--dataroot", default="./data/", help="Root for all datasets")
    parser.add_argument("--train_datamode", default="train", help="Mode for training dataset (should be 'train')")
    parser.add_argument("--train_data_list", default="train_pairs.txt", help="Training data list file relative to dataroot")
    parser.add_argument("--fine_width", type=int, default=192, help="Target width for SPADEGenerator output")
    parser.add_argument("--fine_height", type=int, default=256, help="Target height for SPADEGenerator output")
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_shalwar', help='Directory to save checkpoints')
    parser.add_argument('--tocg_checkpoint', type=str, required=True, help='Path to pre-trained ConditionGenerator (TOCG) checkpoint')
    parser.add_argument('--gen_checkpoint', type=str, default='', help='Path to pre-trained SPADEGenerator checkpoint for fine-tuning or resuming')
    parser.add_argument('--dis_checkpoint', type=str, default='', help='Path to pre-trained Discriminator checkpoint for fine-tuning or resuming')
    parser.add_argument('--resume', action='store_true', help='Resume training from YOUR last saved session checkpoint in checkpoint_dir/name')

    # Training schedule
    parser.add_argument("--load_step", type=int, default=0, help="Manual starting step (used by resume, otherwise 0)")
    parser.add_argument("--keep_step", type=int, default=100000, help="Total steps to train for (initial phase)")
    parser.add_argument("--decay_step", type=int, default=100000, help="Steps for learning rate decay")
    parser.add_argument("--shuffle", action='store_true', default=True, help='Shuffle training data')
    
    # Fine-tuning specific parameters
    parser.add_argument('--finetune_G_lr', type=float, default=0.00005, help='Generator learning rate for fine-tuning')
    parser.add_argument('--finetune_D_lr', type=float, default=0.0001, help='Discriminator learning rate for fine-tuning')
    parser.add_argument('--freeze_G_steps', type=int, default=0, help="Number of initial steps to freeze early G layers (0 to disable)")
    parser.add_argument('--freeze_G_layer_prefixes', type=str, default="head_0,G_middle_0,up_0", 
                        help='Comma-separated prefixes of SPADEGenerator layer names to freeze (e.g., "head_0,G_middle_0")')

    # LPIPS Evaluation (Validation)
    parser.add_argument("--lpips_count", type=int, default=1000, help="Frequency (in steps) of LPIPS evaluation on validation set")
    parser.add_argument("--test_dataroot", default="./data/", help="Root for validation/test dataset")
    parser.add_argument("--test_data_list", default="test_pairs.txt", help="Validation/Test data list file relative to test_dataroot")

    # Model Hyperparameters (SPADEGenerator, Discriminator, TOCG)
    parser.add_argument('--semantic_nc', type=int, default=13, help='Number of semantic classes from TOCG output / for parse remapping')
    parser.add_argument('--gen_semantic_nc', type=int, default=7, help='Number of semantic classes for SPADEGenerator input parse map')
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='Normalization for SPADEGenerator')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='Normalization for Discriminator')
    parser.add_argument('--ngf', type=int, default=64, help='Number of generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='Number of discriminator filters in first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most', help="Upsampling layers in SPADEGenerator")
    parser.add_argument('--init_type', type=str, default='xavier', help='Network initialization type')
    parser.add_argument('--init_variance', type=float, default=0.02, help='Variance for network initialization')

    parser.add_argument('--no_ganFeat_loss', action='store_true', help='Disable GAN Feature matching loss')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='Weight for L1 loss')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='Weight for GAN Feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='Weight for VGG perceptual loss')
    
    parser.add_argument('--n_layers_D', type=int, default=3, help='Number of layers in each NLayerDiscriminator scale')
    parser.add_argument('--num_D', type=int, default=2, help='Number of scales in MultiscaleDiscriminator')
    parser.add_argument("--composition_mask", action='store_true', help='Use composition mask output from SPADEGenerator (implies MultiscaleDiscriminator)')

    parser.add_argument('--occlusion', action='store_true', help="Enable occlusion handling for warped cloth")
    
    # TOCG (ConditionGenerator) specific parameters (must match the pre-trained TOCG model)
    parser.add_argument('--cond_G_ngf', type=int, default=96, help="NGF for ConditionGenerator")
    parser.add_argument("--cond_G_input_width", type=int, default=192, help="Input width for ConditionGenerator")
    parser.add_argument("--cond_G_input_height", type=int, default=256, help="Input height for ConditionGenerator")
    parser.add_argument('--cond_G_num_layers', type=int, default=5, help="Number of layers in ConditionGenerator")
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1", help="Warp feature strategy for ConditionGenerator")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu", help="Output layer type for ConditionGenerator")
    
    parser.add_argument("--display_count", type=int, default=100, help="Frequency (in steps) to display/log training progress")
    parser.add_argument("--save_count", type=int, default=1000, help="Frequency (in steps) to save checkpoints")

    opt = parser.parse_args()

    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
        "Batch size %d is wrong. It must be a multiple of # GPUs %d." % (opt.batch_size, len(opt.gpu_ids))
    
    if opt.fp16 and not APEX_AVAILABLE:
        print("Warning: --fp16 was specified, but Apex is not available. Disabling FP16.")
        opt.fp16 = False
        
    return opt

def set_generator_requires_grad(generator_model, layer_prefixes_to_freeze, freeze_status):
    """
    Sets requires_grad for generator layers.
    If freeze_status is True, layers matching prefixes are frozen (requires_grad=False).
    Otherwise, all generator layers are unfrozen (requires_grad=True).
    """
    if not layer_prefixes_to_freeze: # If empty, means don't freeze any specific layers
        for param in generator_model.parameters():
            param.requires_grad = True
        return

    if freeze_status: # Freeze specified layers
        print(f"Freezing generator layers with prefixes: {layer_prefixes_to_freeze}")
        for name, param in generator_model.named_parameters():
            should_freeze_this_param = any(name.startswith(prefix) for prefix in layer_prefixes_to_freeze)
            param.requires_grad = not should_freeze_this_param
            # if not param.requires_grad:
            #     print(f"  Froze: {name}")
    else: # Unfreeze all layers
        print("Unfreezing all generator layers.")
        for param in generator_model.parameters():
            param.requires_grad = True

# Note: `model` argument renamed to `perceptual_loss_model` to avoid conflict if `model` is used elsewhere
def train(opt, train_loader, val_loader, tocg, generator, discriminator, perceptual_loss_model,
          initial_optimizer_gen, initial_optimizer_dis, initial_scheduler_gen, initial_scheduler_dis): # Pass initial optimizers/schedulers
    
    tocg.cuda() # TOCG is always on GPU
    tocg.eval() # TOCG is pre-trained and fixed

    # Models to GPU
    generator.cuda()
    discriminator.cuda()
    perceptual_loss_model.eval() # LPIPS model also in eval

    # Optimizers and Schedulers (use passed-in ones, might be re-initialized for freezing)
    optimizer_gen = initial_optimizer_gen
    optimizer_dis = initial_optimizer_dis
    scheduler_gen = initial_scheduler_gen
    scheduler_dis = initial_scheduler_dis
    
    # FP16/AMP Initialization (moved here as optimizers might change)
    # This needs to be handled carefully if optimizers are re-initialized for freezing
    amp_handles = None
    generator_amp = generator
    discriminator_amp = discriminator

    if opt.fp16 and APEX_AVAILABLE:
        # We will re-initialize AMP if optimizer_gen changes due to freezing
        # For now, initialize with the initial optimizers
        models_to_amp = [generator, discriminator]
        optimizers_to_amp = [optimizer_gen, optimizer_dis]
        # Remove .module if DataParallel was already applied, Apex works with the base model
        models_to_amp_unwrapped = [(m.module if isinstance(m, nn.DataParallel) else m) for m in models_to_amp]

        amp_handles = amp.initialize(
            models_to_amp_unwrapped, optimizers_to_amp, opt_level='O1', num_losses=2
        )
        # amp.initialize returns models and optimizers. We need to reassign them.
        generator_amp, discriminator_amp = amp_handles[0] # Models returned by amp
        optimizer_gen, optimizer_dis = amp_handles[1]    # Optimizers returned by amp
        print("APEX AMP Initialized for G and D.")


    if len(opt.gpu_ids) > 0:
        # Apply DataParallel after potential AMP model unwrapping/rewrapping
        if not isinstance(generator_amp, nn.DataParallel):
             generator_display = nn.DataParallel(generator_amp, device_ids=opt.gpu_ids)
        else: # Already wrapped (e.g. if AMP returns it wrapped, though unlikely)
             generator_display = generator_amp
        
        if not isinstance(discriminator_amp, nn.DataParallel):
             discriminator_display = nn.DataParallel(discriminator_amp, device_ids=opt.gpu_ids)
        else:
             discriminator_display = discriminator_amp
    else:
        generator_display = generator_amp
        discriminator_display = discriminator_amp

    # Criterion (Losses)
    criterionGAN_cls = Projected_GANs_Loss if not opt.composition_mask else GANLoss
    gan_mode_arg = 'hinge' if opt.composition_mask else None # Projected_GANs_Loss doesn't use gan_mode string
    
    tensor_type = torch.cuda.HalfTensor if opt.fp16 and APEX_AVAILABLE else torch.cuda.FloatTensor
    if gan_mode_arg:
        criterionGAN = criterionGAN_cls(gan_mode_arg, tensor=tensor_type)
    else: # For Projected_GANs_Loss
        criterionGAN = criterionGAN_cls(tensor=tensor_type)

    criterionL1 = nn.L1Loss()
    criterionFeat = nn.L1Loss()
    criterionVGG = VGGLoss()
    
    # Gaussian blur for fake_parse_gauss
    if 'tgm' in sys.modules: # Check if torchgeometry was imported
        gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).cuda()
    else: # Fallback if tgm not available or causes issues
        print("Warning: torchgeometry (tgm) not available. GaussianBlur for parse map will be skipped.")
        gauss = lambda x: x # Identity function

    # Training loop
    train_iter = iter(train_loader)
    current_train_step = opt.load_step
    
    # For selective layer freezing
    prev_G_frozen_status = None
    layer_prefixes_to_freeze_list = [p.strip() for p in opt.freeze_G_layer_prefixes.split(',') if p.strip()] if opt.freeze_G_steps > 0 else []

    # Main training loop
    for step_idx in tqdm(range(opt.load_step, opt.keep_step + opt.decay_step), initial=opt.load_step, total=opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        
        try:
            inputs = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader) # Restart iterator for new epoch
            inputs = next(train_iter)

        # --- Selective Layer Freezing Logic ---
        if opt.freeze_G_steps > 0 and layer_prefixes_to_freeze_list:
            currently_freeze_G = (current_train_step < opt.freeze_G_steps)
            if prev_G_frozen_status is None or currently_freeze_G != prev_G_frozen_status:
                print(f"Step {current_train_step}: Changing G freeze status. Now freezing: {currently_freeze_G}")
                
                # Determine which generator model to pass (AMP wrapped or original)
                base_generator_model = generator_amp.module if isinstance(generator_amp, nn.DataParallel) else generator_amp
                
                set_generator_requires_grad(base_generator_model, layer_prefixes_to_freeze_list, currently_freeze_G)
                
                # Re-initialize optimizer_gen to only include parameters with requires_grad=True
                current_G_lr = opt.finetune_G_lr if (opt.gen_checkpoint or opt.resume) else opt.G_lr # Use fine-tune LR if applicable
                
                print(f"Re-initializing optimizer_gen. Current G LR: {current_G_lr}")
                optimizer_gen = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, base_generator_model.parameters()), 
                    lr=current_G_lr, 
                    betas=(0.0, 0.9) 
                )
                # Re-initialize scheduler_gen for the new optimizer
                # Make sure LambdaLR uses current_train_step correctly for its calculation
                scheduler_gen = torch.optim.lr_scheduler.LambdaLR(optimizer_gen, 
                    lr_lambda=lambda s: 1.0 - max(0, s - opt.keep_step) / float(opt.decay_step + 1) if opt.decay_step > 0 else 1.0,
                    last_epoch=current_train_step -1 # LambdaLR uses step-based last_epoch
                )

                if opt.fp16 and APEX_AVAILABLE:
                    print("Re-initializing Apex AMP for Generator due to freezing change.")
                    # Re-initialize AMP only for the generator part if D's optimizer didn't change
                    # This is complex with AMP. A simpler way is to re-init AMP for all,
                    # or ensure the AMP list passed to initialize correctly reflects updated optimizers.
                    # For now, let's assume we might need to re-init for G and D if G changes.
                    models_to_amp_unwrapped = [
                        base_generator_model, 
                        (discriminator_amp.module if isinstance(discriminator_amp, nn.DataParallel) else discriminator_amp)
                    ]
                    optimizers_to_amp = [optimizer_gen, optimizer_dis] # Use updated optimizer_gen
                    
                    amp_handles = amp.initialize(
                        models_to_amp_unwrapped, optimizers_to_amp, opt_level='O1', num_losses=2, verbosity=0
                    )
                    generator_amp, discriminator_amp = amp_handles[0]
                    optimizer_gen, optimizer_dis = amp_handles[1] # Get potentially updated optimizers from AMP
                    # Re-wrap with DataParallel if needed (if amp returns base models)
                    if len(opt.gpu_ids) > 0:
                        generator_display = nn.DataParallel(generator_amp, device_ids=opt.gpu_ids) if not isinstance(generator_amp, nn.DataParallel) else generator_amp
                        discriminator_display = nn.DataParallel(discriminator_amp, device_ids=opt.gpu_ids) if not isinstance(discriminator_amp, nn.DataParallel) else discriminator_amp
                    else:
                        generator_display = generator_amp
                        discriminator_display = discriminator_amp


                prev_G_frozen_status = currently_freeze_G
                print(f"Optimizer_gen and scheduler_gen re-initialized. Freeze status applied.")

        # --- Unpack inputs (Same as before, ensure keys match CPDataset output) ---
        agnostic_cond = inputs['agnostic'].cuda()
        densepose_cond = inputs['densepose'].cuda() 
        cloth_mask_paired_tocg = inputs['cloth_mask']['paired'].cuda()
        cloth_paired_tocg = inputs['cloth']['paired'].cuda()
        parse_agnostic_tocg = inputs['parse_agnostic'].cuda() 
        densepose_tocg = inputs['densepose'].cuda() 
        im_target = inputs['image'].cuda()

        # --- TOCG Forward Pass (Warping - same as before) ---
        with torch.no_grad():
            cm_tocg_down = F.interpolate(cloth_mask_paired_tocg, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
            parse_agnostic_tocg_down = F.interpolate(parse_agnostic_tocg, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
            cloth_tocg_down = F.interpolate(cloth_paired_tocg, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear', align_corners=False)
            densepose_tocg_down = F.interpolate(densepose_tocg, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear', align_corners=False)
            
            tocg_input1 = torch.cat([cloth_tocg_down, cm_tocg_down], 1)
            tocg_input2 = torch.cat([parse_agnostic_tocg_down, densepose_tocg_down], 1)
            
            all_tocg_outputs = tocg(tocg_input1, tocg_input2)
            # Assuming tocg returns: flow_list_taco, fake_segmap, warped_cloth_taco, warped_mask_taco, flow_list_tvob, ...
            warped_cloth_taco_lowres = all_tocg_outputs[2]
            warped_mask_taco_lowres = all_tocg_outputs[3]
            fake_segmap_tocg_lowres = all_tocg_outputs[1]

            # Upsample directly from TOCG output (simplification, original re-warped with flows)
            # For more accuracy, implement the full re-warping using flow_list_taco and flow_list_tvob
            # as sketched in the previous response and present in original train_generator.py.
            # This simplification might be okay if cond_G_input_height/width is not too small.
            warped_cloth_for_spade = F.interpolate(warped_cloth_taco_lowres, size=(opt.fine_height, opt.fine_width), mode='bilinear', align_corners=False)
            warped_mask_for_spade = F.interpolate(warped_mask_taco_lowres, size=(opt.fine_height, opt.fine_width), mode='nearest')

            fake_segmap_tocg_upsampled = F.interpolate(fake_segmap_tocg_lowres, size=(opt.fine_height, opt.fine_width), mode='bilinear', align_corners=False)
            fake_parse_gauss = gauss(fake_segmap_tocg_upsampled)
            fake_parse_labels = fake_parse_gauss.argmax(dim=1, keepdim=True)

            if opt.occlusion:
                softmax_fake_parse_gauss = F.softmax(fake_parse_gauss, dim=1)
                warped_mask_for_spade = remove_overlap(softmax_fake_parse_gauss, warped_mask_for_spade) # Use updated warped_mask_for_spade
                warped_cloth_for_spade = warped_cloth_for_spade * warped_mask_for_spade + \
                                         torch.ones_like(warped_cloth_for_spade) * (1 - warped_mask_for_spade)
            warped_cloth_for_spade = warped_cloth_for_spade.detach()

            # Create `parse_for_spade` (opt.gen_semantic_nc channels) from `fake_parse_labels` (opt.semantic_nc classes)
            clamped_fake_parse_labels = torch.clamp(fake_parse_labels.long(), 0, opt.semantic_nc - 1)
            old_parse_onehot = F.one_hot(clamped_fake_parse_labels.squeeze(1), num_classes=opt.semantic_nc).permute(0,3,1,2).float()
            
            spade_input_labels_map_config = { 
                0:['background',[0]], 1:['paste',[2,4,7,8,9,10,11]], 2:['upper',[3]], 
                3:['hair',[1]], 4:['left_arm',[5]], 5:['right_arm',[6]], 6:['noise',[12]]
            }
            parse_for_spade = torch.FloatTensor(fake_parse_labels.size(0), opt.gen_semantic_nc, opt.fine_height, opt.fine_width).zero_().cuda()
            for i_spade_map_cfg in range(opt.gen_semantic_nc):
                if i_spade_map_cfg in spade_input_labels_map_config:
                    for label_idx_from_old_parse in spade_input_labels_map_config[i_spade_map_cfg][1]:
                        if 0 <= label_idx_from_old_parse < opt.semantic_nc:
                            parse_for_spade[:, i_spade_map_cfg] += old_parse_onehot[:, label_idx_from_old_parse]
            parse_for_spade = parse_for_spade.detach()
        
        spade_input_concat = torch.cat((agnostic_cond, densepose_cond, warped_cloth_for_spade), dim=1)

        # --- SPADEGenerator Training (Generator model is now generator_display due to DataParallel/AMP) ---
        set_requires_grad(discriminator_display, False)
        optimizer_gen.zero_grad()

        G_losses = {}
        generated_image = None # Define to ensure it's available for visualization
        if opt.composition_mask:
            output_rendered, output_comp_alpha = generator_display(spade_input_concat, parse_for_spade)
            output_comp_intermediate = output_comp_alpha * warped_mask_for_spade 
            output_final_comp_mask = parse_for_spade[:,2:3,:,:] * output_comp_intermediate
            generated_image = warped_cloth_for_spade * output_final_comp_mask + output_rendered * (1 - output_final_comp_mask)
            
            pred_input_fake_G = torch.cat((parse_for_spade, output_rendered), dim=1)
            pred_input_real_G = torch.cat((parse_for_spade, im_target), dim=1)
            pred_fake_outputs_G = discriminator_display(torch.cat((pred_input_fake_G, pred_input_real_G), dim=0))
            
            pred_fake_G_list, pred_real_G_list = [], []
            for p_out_list in pred_fake_outputs_G:
                pred_fake_G_list.append([tensor[:tensor.size(0)//2] for tensor in p_out_list])
                pred_real_G_list.append([tensor[tensor.size(0)//2:] for tensor in p_out_list])

            G_losses['GAN'] = criterionGAN(pred_fake_G_list, True, for_discriminator=False)
            if not opt.no_ganFeat_loss:
                GAN_Feat_loss = torch.tensor(0.0, device=im_target.device)
                for i_gan_feat in range(len(pred_fake_G_list)):
                    for j_gan_feat in range(len(pred_fake_G_list[i_gan_feat]) -1):
                        GAN_Feat_loss += criterionFeat(pred_fake_G_list[i_gan_feat][j_gan_feat], pred_real_G_list[i_gan_feat][j_gan_feat].detach())
                G_losses['GAN_Feat'] = GAN_Feat_loss * (opt.lambda_feat / (len(pred_fake_G_list) if len(pred_fake_G_list)>0 else 1.0) )

            G_losses['VGG'] = (criterionVGG(generated_image, im_target) + criterionVGG(output_rendered, im_target)) * opt.lambda_vgg
            G_losses['L1'] = (criterionL1(generated_image, im_target) + criterionL1(output_rendered, im_target)) * opt.lambda_l1
            G_losses['Composition_Mask_Smoothness'] = torch.mean(torch.abs(1 - output_comp_alpha))
        else: 
            generated_image = generator_display(spade_input_concat, parse_for_spade)
            pred_fake_G, feats_fake_G = discriminator_display(generated_image)
            with torch.no_grad(): _, feats_real_G = discriminator_display(im_target)
            G_losses['GAN'] = criterionGAN(pred_fake_G, True, for_discriminator=False) * 0.5
            if not opt.no_ganFeat_loss:
                GAN_Feat_loss = torch.tensor(0.0, device=im_target.device)
                for i_gan_feat in range(len(feats_fake_G)):
                    for j_gan_feat in range(len(feats_fake_G[i_gan_feat])):
                         GAN_Feat_loss += criterionFeat(feats_fake_G[i_gan_feat][j_gan_feat], feats_real_G[i_gan_feat][j_gan_feat].detach())
                G_losses['GAN_Feat'] = GAN_Feat_loss * (opt.lambda_feat / (len(feats_fake_G) if len(feats_fake_G)>0 else 1.0))
            G_losses['VGG'] = criterionVGG(generated_image, im_target) * opt.lambda_vgg
            G_losses['L1'] = criterionL1(generated_image, im_target) * opt.lambda_l1
        
        loss_gen_total = sum(val for val in G_losses.values() if val is not None) # Sum valid losses
        
        if opt.fp16 and APEX_AVAILABLE:
            # AMP loss ID for G is 0
            with amp.scale_loss(loss_gen_total, optimizer_gen, loss_id=0) as scaled_loss_gen:
                scaled_loss_gen.backward()
        else:
            loss_gen_total.backward()
        optimizer_gen.step()

        # --- Discriminator Training ---
        set_requires_grad(discriminator_display, True)
        if not opt.composition_mask: # ProjectedDiscriminator specific
            base_discriminator_model = discriminator_display.module if isinstance(discriminator_display, nn.DataParallel) else discriminator_display
            if hasattr(base_discriminator_model, 'feature_network'):
                base_discriminator_model.feature_network.requires_grad_(False)
        
        optimizer_dis.zero_grad()
        D_losses = {}
        with torch.no_grad():
            if opt.composition_mask:
                output_rendered_detached, _ = generator_display(spade_input_concat, parse_for_spade)
                # No need to reconstruct full `generated_image_detached` if D only sees `output_rendered_detached`
                pred_input_fake_D = torch.cat((parse_for_spade, output_rendered_detached.detach()), dim=1)
            else:
                generated_image_detached = generator_display(spade_input_concat, parse_for_spade).detach()
                pred_input_fake_D = generated_image_detached
        
        pred_input_real_D = torch.cat((parse_for_spade, im_target), dim=1) if opt.composition_mask else im_target

        if opt.composition_mask:
            pred_outputs_D = discriminator_display(torch.cat((pred_input_fake_D, pred_input_real_D), dim=0))
            pred_fake_D_list, pred_real_D_list = [], []
            for p_out_list in pred_outputs_D:
                pred_fake_D_list.append([tensor[:tensor.size(0)//2] for tensor in p_out_list])
                pred_real_D_list.append([tensor[tensor.size(0)//2:] for tensor in p_out_list])
            D_losses['D_Fake'] = criterionGAN(pred_fake_D_list, False, for_discriminator=True)
            D_losses['D_Real'] = criterionGAN(pred_real_D_list, True, for_discriminator=True)
        else:
            pred_fake_D, _ = discriminator_display(pred_input_fake_D)
            pred_real_D, _ = discriminator_display(pred_input_real_D)
            D_losses['D_Fake'] = criterionGAN(pred_fake_D, False, for_discriminator=True)
            D_losses['D_Real'] = criterionGAN(pred_real_D, True, for_discriminator=True)

        loss_dis_total = sum(D_losses.values())
        if opt.fp16 and APEX_AVAILABLE:
            # AMP loss ID for D is 1
            with amp.scale_loss(loss_dis_total, optimizer_dis, loss_id=1) as scaled_loss_dis:
                scaled_loss_dis.backward()
        else:
            loss_dis_total.backward()
        optimizer_dis.step()
        
        current_train_step +=1

        # --- Logging, Visualization, LPIPS, Checkpointing (use current_train_step) ---
        if (current_train_step) % opt.display_count == 0:
            # Visualization
            if generated_image is not None : # Ensure generated_image was created
                a_0 = im_target.cuda()[0] 
                b_0 = generated_image.cuda()[0] 
                c_0 = warped_cloth_for_spade.cuda()[0]
                
                combine = torch.cat((a_0, b_0, c_0), dim=2)
                cv_img = (combine.permute(1,2,0).detach().cpu().numpy() + 1) / 2.0
                cv_img = np.clip(cv_img, 0, 1)
                rgb_img = (cv_img * 255).astype(np.uint8)
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                
                vis_save_dir = os.path.join('sample_fs_toig', opt.name)
                os.makedirs(vis_save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(vis_save_dir, f'{current_train_step}.jpg'), bgr_img)

            # Logging
            t_iter_display = time.time() - iter_start_time
            log_msg_display = f"step: {current_train_step:7d}, time: {t_iter_display:.2f}s, G_loss: {loss_gen_total.item():.3f}, "
            if 'GAN' in G_losses and G_losses['GAN'] is not None: log_msg_display += f"G_adv: {G_losses['GAN'].item():.3f}, "
            if 'GAN_Feat' in G_losses and G_losses['GAN_Feat'] is not None: log_msg_display += f"G_feat: {G_losses['GAN_Feat'].item():.3f}, "
            if 'VGG' in G_losses and G_losses['VGG'] is not None: log_msg_display += f"G_vgg: {G_losses['VGG'].item():.3f}, "
            if 'L1' in G_losses and G_losses['L1'] is not None: log_msg_display += f"G_L1: {G_losses['L1'].item():.3f}, "
            if 'Composition_Mask_Smoothness' in G_losses and G_losses['Composition_Mask_Smoothness'] is not None: log_msg_display += f"G_CompAlpha: {G_losses['Composition_Mask_Smoothness'].item():.3f}, "
            log_msg_display += f"D_loss: {loss_dis_total.item():.3f}"
            if 'D_Fake' in D_losses and D_losses['D_Fake'] is not None: log_msg_display += f", D_fake: {D_losses['D_Fake'].item():.3f}"
            if 'D_Real' in D_losses and D_losses['D_Real'] is not None: log_msg_display += f", D_real: {D_losses['D_Real'].item():.3f}"
            print(log_msg_display, flush=True)

        # LPIPS Evaluation on Validation Set
        if (current_train_step) % opt.lpips_count == 0 and current_train_step > 0:
            generator_display.eval() # Use generator_display which might be DataParallel wrapped
            lpips_resize_transform = T_torchvision.Compose([T_torchvision.Resize((128, 128), interpolation=T_torchvision.InterpolationMode.BICUBIC)])
            avg_lpips_distance = 0.0
            num_lpips_samples = 0
            
            print(f"\nCalculating LPIPS on validation set at step {current_train_step}...")
            val_iter_lpips = iter(val_loader) # Use val_loader now
            max_lpips_batches = min(500 // val_loader.batch_size if val_loader.batch_size > 0 else 500, len(val_loader))

            for _ in tqdm(range(max_lpips_batches), desc="LPIPS Eval"):
                try:
                    lpips_inputs = next(val_iter_lpips)
                except StopIteration: break
                
                agnostic_cond_lpips = lpips_inputs['agnostic'].cuda()
                densepose_cond_lpips = lpips_inputs['densepose'].cuda()
                cloth_mask_paired_tocg_lpips = lpips_inputs['cloth_mask']['paired'].cuda()
                cloth_paired_tocg_lpips = lpips_inputs['cloth']['paired'].cuda()
                parse_agnostic_tocg_lpips = lpips_inputs['parse_agnostic'].cuda()
                densepose_tocg_lpips = lpips_inputs['densepose'].cuda()
                im_target_lpips = lpips_inputs['image'].cuda()

                with torch.no_grad():
                    # TOCG forward for LPIPS (same logic as training to get warped_cloth and parse_for_spade)
                    cm_tocg_down_lpips = F.interpolate(cloth_mask_paired_tocg_lpips, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
                    parse_agnostic_tocg_down_lpips = F.interpolate(parse_agnostic_tocg_lpips, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
                    cloth_tocg_down_lpips = F.interpolate(cloth_paired_tocg_lpips, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear', align_corners=False)
                    densepose_tocg_down_lpips = F.interpolate(densepose_tocg_lpips, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear', align_corners=False)
                    tocg_input1_lpips = torch.cat([cloth_tocg_down_lpips, cm_tocg_down_lpips], 1)
                    tocg_input2_lpips = torch.cat([parse_agnostic_tocg_down_lpips, densepose_tocg_down_lpips], 1)
                    all_tocg_outputs_lpips = tocg(tocg_input1_lpips, tocg_input2_lpips)
                    
                    warped_cloth_taco_lowres_lpips = all_tocg_outputs_lpips[2]
                    warped_mask_taco_lowres_lpips = all_tocg_outputs_lpips[3]
                    fake_segmap_tocg_lowres_lpips = all_tocg_outputs_lpips[1]

                    warped_cloth_for_spade_lpips = F.interpolate(warped_cloth_taco_lowres_lpips, size=(opt.fine_height, opt.fine_width), mode='bilinear', align_corners=False)
                    warped_mask_for_spade_lpips = F.interpolate(warped_mask_taco_lowres_lpips, size=(opt.fine_height, opt.fine_width), mode='nearest')
                    
                    fake_segmap_tocg_upsampled_lpips = F.interpolate(fake_segmap_tocg_lowres_lpips, size=(opt.fine_height, opt.fine_width), mode='bilinear', align_corners=False)
                    fake_parse_gauss_lpips = gauss(fake_segmap_tocg_upsampled_lpips)
                    fake_parse_labels_lpips = fake_parse_gauss_lpips.argmax(dim=1, keepdim=True)

                    if opt.occlusion:
                        softmax_fake_parse_gauss_lpips = F.softmax(fake_parse_gauss_lpips, dim=1)
                        warped_mask_for_spade_lpips = remove_overlap(softmax_fake_parse_gauss_lpips, warped_mask_for_spade_lpips)
                        warped_cloth_for_spade_lpips = warped_cloth_for_spade_lpips * warped_mask_for_spade_lpips + \
                                                 torch.ones_like(warped_cloth_for_spade_lpips) * (1 - warped_mask_for_spade_lpips)
                    
                    clamped_fake_parse_labels_lpips = torch.clamp(fake_parse_labels_lpips.long(), 0, opt.semantic_nc - 1)
                    old_parse_onehot_lpips = F.one_hot(clamped_fake_parse_labels_lpips.squeeze(1), num_classes=opt.semantic_nc).permute(0,3,1,2).float()
                    parse_for_spade_lpips = torch.FloatTensor(fake_parse_labels_lpips.size(0), opt.gen_semantic_nc, opt.fine_height, opt.fine_width).zero_().cuda()
                    for i_spade_map_cfg_lpips in range(opt.gen_semantic_nc): # spade_input_labels_map_config from outer scope
                         if i_spade_map_cfg_lpips in spade_input_labels_map_config:
                            for label_idx_from_old_parse_lpips in spade_input_labels_map_config[i_spade_map_cfg_lpips][1]:
                                if 0 <= label_idx_from_old_parse_lpips < opt.semantic_nc:
                                    parse_for_spade_lpips[:, i_spade_map_cfg_lpips] += old_parse_onehot_lpips[:, label_idx_from_old_parse_lpips]
                    
                    spade_input_concat_lpips = torch.cat((agnostic_cond_lpips, densepose_cond_lpips, warped_cloth_for_spade_lpips.detach()), dim=1)
                    
                    if opt.composition_mask:
                        output_rendered_lpips, output_comp_alpha_lpips = generator_display(spade_input_concat_lpips, parse_for_spade_lpips.detach())
                        output_comp_intermediate_lpips = output_comp_alpha_lpips * warped_mask_for_spade_lpips
                        output_final_comp_mask_lpips = parse_for_spade_lpips[:,2:3,:,:] * output_comp_intermediate_lpips
                        generated_image_lpips = warped_cloth_for_spade_lpips * output_final_comp_mask_lpips + \
                                                output_rendered_lpips * (1 - output_final_comp_mask_lpips)
                    else:
                        generated_image_lpips = generator_display(spade_input_concat_lpips, parse_for_spade_lpips.detach())
                    
                    lpips_val_batch = perceptual_loss_model.forward(lpips_resize_transform(im_target_lpips), 
                                                                    lpips_resize_transform(generated_image_lpips))
                    avg_lpips_distance += lpips_val_batch.sum().item()
                    num_lpips_samples += im_target_lpips.size(0)
            
            if num_lpips_samples > 0: avg_lpips_distance /= num_lpips_samples
            print(f"Validation LPIPS @ step {current_train_step}: {avg_lpips_distance:.4f} (over {num_lpips_samples} samples)")
            generator_display.train() # Set G back to train mode

        # Save Checkpoint
        if (current_train_step) % opt.save_count == 0 and current_train_step > 0:
            # Checkpoint path for this fine-tuning session
            session_checkpoint_path = os.path.join(opt.checkpoint_dir, opt.name, 'generator_checkpoint.pth')
            checkpoint_data = {
                'step': current_train_step,
                'generator_state_dict': (generator_amp.module if isinstance(generator_amp, nn.DataParallel) else generator_amp).state_dict(), # Save base model state
                'discriminator_state_dict': (discriminator_amp.module if isinstance(discriminator_amp, nn.DataParallel) else discriminator_amp).state_dict(),
                'optimizer_gen_state_dict': optimizer_gen.state_dict(),
                'optimizer_dis_state_dict': optimizer_dis.state_dict(),
                'scheduler_gen_state_dict': scheduler_gen.state_dict(),
                'scheduler_dis_state_dict': scheduler_dis.state_dict(),
            }
            if opt.fp16 and APEX_AVAILABLE and amp_handles is not None : # amp_handles from amp.initialize
                checkpoint_data['amp'] = amp.state_dict()
            
            torch.save(checkpoint_data, session_checkpoint_path)
            print(f"Saved session checkpoint at step {current_train_step} to {session_checkpoint_path}")

        # Step Schedulers
        # LambdaLR scheduler steps are typically called after optimizer.step() each iteration,
        # or per epoch. Original code stepped every 1000 iterations.
        # Let's adjust to step based on current_train_step for LambdaLR's last_epoch.
        if (current_train_step +1) % 1000 ==0 : # Match original, but careful with LambdaLR's expectation
             pass # Schedulers are now updated based on current_train_step when optimizer is re-init for freezing
                  # Or, if not freezing, they step based on initial setup.
                  # A more standard way is to call scheduler.step() each iteration if it's that kind of scheduler
                  # or after each "epoch" concept.
                  # For LambdaLR using total steps, it's often updated each iteration.
        scheduler_gen.step() # Step based on current optimizer state, after G step
        scheduler_dis.step() # Step based on current optimizer state, after D step


def main():
    opt = get_opt()
    print("--- Parsed Options ---")
    for k, v in vars(opt).items(): print(f"{k}: {v}")
    print("----------------------")
    print(f"Starting training/fine-tuning for: {opt.name}!")

    os.makedirs(os.path.join('sample_fs_toig', opt.name), exist_ok=True)
    os.makedirs(os.path.join(opt.checkpoint_dir, opt.name), exist_ok=True)

    # --- Training Dataset ---
    train_opt = copy.deepcopy(opt)
    train_opt.datamode = opt.train_datamode 
    train_opt.data_list = opt.train_data_list
    train_opt.shuffle = opt.shuffle 
    train_dataset = CPDataset(train_opt)
    # Use standard DataLoader, CPDataLoader was just a wrapper
    train_loader = DataLoader(train_dataset, batch_size=train_opt.batch_size, shuffle=train_opt.shuffle,
                              num_workers=train_opt.workers, pin_memory=True, drop_last=True)

    # --- Validation Dataset (for LPIPS) ---
    val_opt = copy.deepcopy(opt)
    val_opt.dataroot = opt.test_dataroot 
    val_opt.datamode = 'test' # Tells CPDataset to use base transforms
    val_opt.data_list = opt.test_data_list
    val_opt.batch_size = opt.batch_size # Can be different
    val_opt.shuffle = False
    val_dataset_full = CPDataset(val_opt)
    num_val_samples = min(500, len(val_dataset_full)) # Limit val samples for LPIPS
    val_indices = np.random.choice(len(val_dataset_full), num_val_samples, replace=False) if len(val_dataset_full) > num_val_samples else np.arange(len(val_dataset_full))
    val_dataset_subset = Subset(val_dataset_full, val_indices)
    val_loader = DataLoader(val_dataset_subset, batch_size=val_opt.batch_size, shuffle=False,
                            num_workers=val_opt.workers, pin_memory=True, drop_last=False)
    print(f"Using {len(val_dataset_subset)} samples for LPIPS validation.")

    # --- TOCG Model ---
    tocg_input1_nc = 3 + 1 
    tocg_input2_nc = opt.semantic_nc + 3 
    tocg = ConditionGenerator(opt, input1_nc=tocg_input1_nc, input2_nc=tocg_input2_nc, 
                              output_nc=opt.semantic_nc, ngf=opt.cond_G_ngf, 
                              norm_layer=nn.BatchNorm2d, num_layers=opt.cond_G_num_layers)
    
    # --- SPADEGenerator ---
    spade_input_nc = 3 + 3 + 3 
    generator = SPADEGenerator(opt, spade_input_nc)
    generator.init_weights(opt.init_type, opt.init_variance) # Initialize first

    # --- Discriminator ---
    discriminator = None
    if opt.composition_mask:
        opt_for_D = copy.deepcopy(opt) 
        opt_for_D.input_nc = opt.gen_semantic_nc + 3 
        discriminator = create_network(MultiscaleDiscriminator, opt_for_D)
    else:
        discriminator = ProjectedDiscriminator(interp224=False, input_nc=3)
    # Discriminator also needs weight initialization if not loaded from checkpoint
    # Assuming create_network or ProjectedDiscriminator handles its own init if needed.

    # --- LPIPS Model ---
    perceptual_loss_model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available())

    # --- Optimizers and Schedulers (Initial Setup) ---
    # Use fine-tuning LRs if starting from pre-trained, otherwise standard LRs
    # This logic will be refined by the resume/finetune checkpoint loading
    current_G_lr = opt.finetune_G_lr if (opt.gen_checkpoint and not opt.resume) else opt.G_lr
    current_D_lr = opt.finetune_D_lr if (opt.dis_checkpoint and not opt.resume) else opt.D_lr
    
    optimizer_gen = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=current_G_lr, betas=(0.0, 0.9))
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=current_D_lr, betas=(0.0, 0.9))
    
    # Schedulers: LambdaLR expects last_epoch to be "number of steps - 1"
    # If opt.load_step is from a resumed session, it's correct.
    # If starting fresh fine-tuning (opt.load_step=0), last_epoch should be -1.
    last_epoch_val = opt.load_step -1 if opt.load_step > 0 else -1

    scheduler_gen = torch.optim.lr_scheduler.LambdaLR(optimizer_gen, 
        lr_lambda=lambda s: 1.0 - max(0, s - opt.keep_step) / float(opt.decay_step + 1) if opt.decay_step > 0 else 1.0,
        last_epoch=last_epoch_val)
    scheduler_dis = torch.optim.lr_scheduler.LambdaLR(optimizer_dis, 
        lr_lambda=lambda s: 1.0 - max(0, s - opt.keep_step) / float(opt.decay_step + 1) if opt.decay_step > 0 else 1.0,
        last_epoch=last_epoch_val)


    # --- Checkpoint Loading (Handles resume vs. fine-tune from pre-trained) ---
    # TOCG first, as it's always loaded and fixed
    if not opt.tocg_checkpoint or not os.path.exists(opt.tocg_checkpoint):
        print(f"ERROR: TOCG checkpoint not found or not specified: {opt.tocg_checkpoint}"); sys.exit(1)
    print(f"Loading fixed TOCG from: {opt.tocg_checkpoint}")
    load_checkpoint(tocg, opt.tocg_checkpoint) # load_checkpoint should handle .cuda()

    session_checkpoint_path = os.path.join(opt.checkpoint_dir, opt.name, 'generator_checkpoint.pth')
    if opt.resume:
        if os.path.exists(session_checkpoint_path):
            print(f"Resuming YOUR fine-tuning session from: {session_checkpoint_path}")
            checkpoint = torch.load(session_checkpoint_path, map_location=f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu')
            opt.load_step = checkpoint['step']
            
            # Load models (handle DataParallel-saved models if necessary)
            gen_state_dict = checkpoint['generator_state_dict']
            dis_state_dict = checkpoint['discriminator_state_dict']
            # If model was saved with DataParallel, keys might have 'module.' prefix
            # generator.load_state_dict(gen_state_dict) # Or strip 'module.' if needed
            # discriminator.load_state_dict(dis_state_dict)
            # A robust way:
            def load_model_state(model, state_dict):
                try: model.load_state_dict(state_dict, strict=True)
                except RuntimeError: # Try stripping 'module.'
                    print("Attempting to load state_dict by stripping 'module.' prefix...")
                    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    model.load_state_dict(new_state_dict, strict=True)
            
            load_model_state(generator, gen_state_dict)
            load_model_state(discriminator, dis_state_dict)

            optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
            optimizer_dis.load_state_dict(checkpoint['optimizer_dis_state_dict'])
            scheduler_gen.load_state_dict(checkpoint['scheduler_gen_state_dict'])
            scheduler_dis.load_state_dict(checkpoint['scheduler_dis_state_dict'])
            
            if opt.fp16 and APEX_AVAILABLE and 'amp' in checkpoint:
                # AMP state loading needs to happen AFTER models and optimizers are on GPU
                # and optimizers are correctly associated with model parameters.
                # amp.load_state_dict(checkpoint['amp']) # This will be handled inside train() after amp.initialize
                # For now, just store it to be loaded after amp init in train()
                # This is tricky. Simpler: AMP re-initializes and learns scaling factors again.
                print("AMP state from checkpoint will be used if Apex is re-initialized with these optimizers.")
            print(f"Resumed from your step {opt.load_step}. Schedulers' last_epoch updated.")
        else:
            print(f"WARNING: --resume specified, but session checkpoint {session_checkpoint_path} not found.")
            opt.load_step = 0 # Start fresh or from author's pre-trained
            if opt.gen_checkpoint and os.path.exists(opt.gen_checkpoint):
                print(f"Loading AUTHOR'S pre-trained Generator for fine-tuning from: {opt.gen_checkpoint}")
                load_checkpoint(generator, opt.gen_checkpoint)
            if opt.dis_checkpoint and os.path.exists(opt.dis_checkpoint):
                print(f"Loading AUTHOR'S pre-trained Discriminator for fine-tuning from: {opt.dis_checkpoint}")
                load_checkpoint(discriminator, opt.dis_checkpoint)
    else: # Not resuming, start fine-tuning from author's weights if provided
        opt.load_step = 0
        print("Starting new training/fine-tuning session (opt.load_step = 0).")
        if opt.gen_checkpoint and os.path.exists(opt.gen_checkpoint):
            print(f"Loading AUTHOR'S pre-trained Generator for fine-tuning from: {opt.gen_checkpoint}")
            load_checkpoint(generator, opt.gen_checkpoint)
        else: print("No pre-trained Generator provided, using initialized weights.")
        
        if opt.dis_checkpoint and os.path.exists(opt.dis_checkpoint):
            print(f"Loading AUTHOR'S pre-trained Discriminator for fine-tuning from: {opt.dis_checkpoint}")
            load_checkpoint(discriminator, opt.dis_checkpoint)
        else: print("No pre-trained Discriminator provided, using initialized weights.")
        
        # Re-set optimizers with fine-tuning LRs if starting from author's weights
        print(f"Setting G_lr={opt.finetune_G_lr}, D_lr={opt.finetune_D_lr} for fine-tuning.")
        optimizer_gen = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=opt.finetune_G_lr, betas=(0.0, 0.9))
        optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=opt.finetune_D_lr, betas=(0.0, 0.9))
        scheduler_gen = torch.optim.lr_scheduler.LambdaLR(optimizer_gen, 
            lr_lambda=lambda s: 1.0 - max(0, s - opt.keep_step) / float(opt.decay_step + 1) if opt.decay_step > 0 else 1.0, last_epoch=-1)
        scheduler_dis = torch.optim.lr_scheduler.LambdaLR(optimizer_dis, 
            lr_lambda=lambda s: 1.0 - max(0, s - opt.keep_step) / float(opt.decay_step + 1) if opt.decay_step > 0 else 1.0, last_epoch=-1)


    # Ensure models are on GPU before passing to train function if not already handled by load_checkpoint
    # generator.cuda()
    # discriminator.cuda()

    # Call train function
    train(opt, train_loader, val_loader, tocg, generator, discriminator, perceptual_loss_model,
          optimizer_gen, optimizer_dis, scheduler_gen, scheduler_dis) # Pass optimizers and schedulers

    # Save final models
    final_gen_path = os.path.join(opt.checkpoint_dir, opt.name, 'gen_model_final.pth')
    final_dis_path = os.path.join(opt.checkpoint_dir, opt.name, 'dis_model_final.pth')
    # Save base model if DataParallel or AMP wrapped
    save_checkpoint(generator.module if isinstance(generator, nn.DataParallel) else generator, final_gen_path)
    save_checkpoint(discriminator.module if isinstance(discriminator, nn.DataParallel) else discriminator, final_dis_path)
    print(f"Saved final generator to {final_gen_path}")
    print(f"Saved final discriminator to {final_dis_path}")

    print(f"Finished training/fine-tuning for: {opt.name}!")

if __name__ == "__main__":
    main()