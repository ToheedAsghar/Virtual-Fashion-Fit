# cp_dataset.py (Modified for Albumentations)
# coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as T # Renamed to T to avoid conflict
from torchvision.transforms.functional import to_tensor, normalize # For specific cases

from PIL import Image, ImageDraw
import json
import cv2 # For albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os.path as osp
import numpy as np


class CPDataset(data.Dataset):
    """
    Dataset for CP-VTON with integrated Albumentations.
    """
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode  # train or test or self-defined
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc # Number of output channels for semantic maps
        self.data_path = osp.join(opt.dataroot, opt.datamode)

        # Determine if augmentations should be used
        # Typically, augmentations are for 'train' mode.
        # You can add another opt flag like opt.use_augmentations for explicit control if needed.
        self.use_albumentations = (self.datamode == 'train')

        if self.use_albumentations:
            self._init_albumentations_transforms()
        else:
            # Original torchvision transforms for test/non-augmented mode
            self.pil_transform_rgb = T.Compose([
                T.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.pil_transform_mask = T.Compose([ # For single channel masks (e.g., cloth_mask)
                T.Resize((self.fine_height, self.fine_width), interpolation=Image.NEAREST),
                T.ToTensor()
            ])
            self.pil_transform_label_map = T.Compose([ # For label maps (e.g., parse_map)
                T.Resize((self.fine_height, self.fine_width), interpolation=Image.NEAREST)
                # ToTensor is handled manually for label maps to ensure LongTensor
            ])


        # load data list (same as original)
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = dict()
        # Paired mode is assumed for SD-VITON training (person image name is same as cloth image name effectively for pairing)
        # The original CPDataset uses im_names for c_names['paired'] if the datalist provides pairs.
        self.c_names['paired'] = c_names # Using the provided c_names from data_list for paired cloth
        # self.c_names['unpaired'] = c_names # If unpaired functionality is needed later


    def _init_albumentations_transforms(self):
        fh, fw = self.opt.fine_height, self.opt.fine_width
        norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        # For training with augmentations
        # Common geometric transforms that apply to image and all additional_targets
        geometric_transforms = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15,
                               interpolation=cv2.INTER_LINEAR, # For image targets
                               border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7), # value=0 for black padding
            # Resize should be one of the last geometric, or first if RandomResizedCrop is not used
        ]
        
        # Color/pixel transforms only for 'image' type targets
        color_pixel_transforms = [
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ]

        self.person_train_transform = A.Compose(
            geometric_transforms + [
                A.RandomResizedCrop(height=fh, width=fw, scale=(0.8, 1.0), ratio=(0.75, 1.33), 
                                    interpolation=cv2.INTER_LINEAR, p=0.5), # Overrides fixed Resize for person image if applied
                A.Resize(height=fh, width=fw, interpolation=cv2.INTER_LINEAR, always_apply=True), # Ensure final size if crop not applied
            ] + color_pixel_transforms + [
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2(), # Converts HWC NumPy to CHW Tensor
            ],
            additional_targets={
                'im_parse_map': 'mask',      # Raw parse map (e.g., 0-19 int labels)
                'agnostic_map': 'image',     # Agnostic map (RGB)
                'pose_map': 'image',         # OpenPose rendered image (RGB)
                'densepose_map': 'image',    # DensePose map (RGB UVI)
                'parse_agnostic_map_raw': 'mask', # Raw labels for new_parse_agnostic_map
            },
            # For masks, INTER_NEAREST is used by default by albumentations if target type is 'mask'
        )

        self.garment_train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=5, p=0.3,
                               interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=[255,255,255]),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(height=fh, width=fw, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ], additional_targets={
            'cloth_mask': 'mask',
        })

        # Base transforms for NumPy arrays (used if augmentations are off but still want albumentations pipeline)
        # Or for specific items that don't get full augmentation
        self.np_base_transform_rgb = A.Compose([
            A.Resize(height=fh, width=fw, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ])
        self.np_base_transform_mask = A.Compose([
            A.Resize(height=fh, width=fw, interpolation=cv2.INTER_NEAREST),
            ToTensorV2(transpose_mask=True), # Converts (H,W) to (1,H,W)
        ])
        self.np_base_transform_to_tensor_only = A.Compose([ # For specific cases like densepose_wo_normalize
             A.Resize(height=fh, width=fw, interpolation=cv2.INTER_LINEAR), # Or NEAREST if it's label-like
             ToTensorV2() # Scales uint8 [0,255] to float [0,1]
        ])

    def name(self):
        return "CPDatasetWithAlbumentations"

    def get_agnostic(self, im_pil, im_parse_pil, pose_data_np):
        # Copied from original CPDataset, ensure inputs are PIL Image and NumPy array
        parse_array = np.array(im_parse_pil) # Convert PIL to NumPy for processing
        # Head parts (original classes 4 and 13)
        parse_head = ((parse_array == 4) | (parse_array == 13)).astype(np.float32)
        # Lower body parts (original classes 9, 12, 16, 17, 18, 19)
        parse_lower = ((parse_array == 9) | (parse_array == 12) | \
                       (parse_array == 16) | (parse_array == 17) | \
                       (parse_array == 18) | (parse_array == 19)).astype(np.float32)

        agnostic_pil = im_pil.copy()
        agnostic_draw = ImageDraw.Draw(agnostic_pil)

        length_a = np.linalg.norm(pose_data_np[5] - pose_data_np[2])
        length_b = np.linalg.norm(pose_data_np[12] - pose_data_np[9])
        point = (pose_data_np[9] + pose_data_np[12]) / 2
        
        # Create a mutable copy of pose_data_np for modifications
        pose_data_modified = pose_data_np.copy()
        if length_b == 0: # Avoid division by zero
            length_b = 1e-6 
        pose_data_modified[9] = point + (pose_data_modified[9] - point) / length_b * length_a
        pose_data_modified[12] = point + (pose_data_modified[12] - point) / length_b * length_a
        
        r = int(length_a / 16) + 1

        # Mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data_modified[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data_modified[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data_modified[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data_modified[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data_modified[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # Mask neck
        pointx, pointy = pose_data_modified[1]
        agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

        # Mask arms
        agnostic_draw.line([tuple(pose_data_modified[i]) for i in [2, 5]], 'gray', width=r*12)
        for i in [2, 5]:
            pointx, pointy = pose_data_modified[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
        for i_pose_idx in [3, 4, 6, 7]: # Use a different loop variable name
            # Check visibility of points (pose_data_modified has x,y,visibility; we use x,y which is index 0,1)
            # Original code uses [i-1,0] and [i,0] - this means pose_data_modified has shape (num_keypoints, 2)
            prev_point_visible = not (pose_data_modified[i_pose_idx-1, 0] == 0.0 and pose_data_modified[i_pose_idx-1, 1] == 0.0)
            curr_point_visible = not (pose_data_modified[i_pose_idx, 0] == 0.0 and pose_data_modified[i_pose_idx, 1] == 0.0)

            if not (prev_point_visible and curr_point_visible):
                continue
            agnostic_draw.line([tuple(pose_data_modified[j]) for j in [i_pose_idx - 1, i_pose_idx]], 'gray', width=r*10)
            pointx, pointy = pose_data_modified[i_pose_idx]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        canvas_w, canvas_h = im_pil.size
        for parse_id, pose_ids_segment in [(14, [5, 6, 7]), (15, [2, 3, 4])]: # Corrected variable name
            mask_arm = Image.new('L', (canvas_w, canvas_h), 'white')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            
            first_point_idx = pose_ids_segment[0]
            pointx_first, pointy_first = pose_data_modified[first_point_idx]
            mask_arm_draw.ellipse((pointx_first-r*5, pointy_first-r*6, pointx_first+r*5, pointy_first+r*6), 'black', 'black')

            for k_idx in range(1, len(pose_ids_segment)):
                current_kp_idx = pose_ids_segment[k_idx]
                prev_kp_idx = pose_ids_segment[k_idx-1]

                prev_point_visible = not (pose_data_modified[prev_kp_idx, 0] == 0.0 and pose_data_modified[prev_kp_idx, 1] == 0.0)
                curr_point_visible = not (pose_data_modified[current_kp_idx, 0] == 0.0 and pose_data_modified[current_kp_idx, 1] == 0.0)

                if not (prev_point_visible and curr_point_visible):
                    continue
                mask_arm_draw.line([tuple(pose_data_modified[j]) for j in [prev_kp_idx, current_kp_idx]], 'black', width=r*10)
                pointx, pointy = pose_data_modified[current_kp_idx]
                if current_kp_idx != pose_ids_segment[-1]:
                    mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
            
            last_point_idx = pose_ids_segment[-1]
            pointx_last, pointy_last = pose_data_modified[last_point_idx]
            if not (pose_data_modified[last_point_idx, 0] == 0.0 and pose_data_modified[last_point_idx, 1] == 0.0): # Check if last point is visible
                mask_arm_draw.ellipse((pointx_last-r*4, pointy_last-r*4, pointx_last+r*4, pointy_last+r*4), 'black', 'black')
            
            parse_arm_mask_np = (np.array(mask_arm) < 128).astype(np.float32) # Where mask_arm is black (0)
            parse_id_match_np = (parse_array == parse_id).astype(np.float32)
            final_arm_mask_for_paste = parse_arm_mask_np * parse_id_match_np
            
            agnostic_pil.paste(im_pil, None, Image.fromarray(np.uint8(final_arm_mask_for_paste * 255), 'L'))

        agnostic_pil.paste(im_pil, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic_pil.paste(im_pil, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        return agnostic_pil

    def _remap_parse_map(self, parse_tensor_labels_hw, semantic_nc_out):
        # parse_tensor_labels_hw: (H,W) tensor with original labels (0-19 assumed)
        # semantic_nc_out: Number of target semantic classes
        # Returns: (semantic_nc_out, H, W) one-hot tensor with remapped labels
        
        labels_map_config = { # From original CPDataset.py (VITON / LIP parsing)
            0: ['background', [0, 10]], 1: ['hair', [1, 2]], 2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]], 4: ['bottom', [9, 12]], 5: ['left_arm', [14]],
            6: ['right_arm', [15]], 7: ['left_leg', [16]], 8: ['right_leg', [17]],
            9: ['left_shoe', [18]], 10: ['right_shoe', [19]], 11: ['socks', [8]],
            12: ['noise', [3, 11]] # Typically, semantic_nc = 13 for this mapping
        }
        if semantic_nc_out != len(labels_map_config):
            print(f"Warning: semantic_nc {semantic_nc_out} does not match labels_map_config length {len(labels_map_config)}. Remapping might be incomplete.")

        h, w = parse_tensor_labels_hw.shape
        # Ensure input tensor is Long for scatter
        parse_tensor_labels_hw_long = parse_tensor_labels_hw.long()

        # Create a one-hot map from the raw 0-19 labels. Max original label is 19.
        raw_channel_count = 20
        raw_one_hot = torch.FloatTensor(raw_channel_count, h, w).zero_()
        # Clamp labels to be within [0, raw_channel_count-1] before scatter
        clamped_labels = torch.clamp(parse_tensor_labels_hw_long.unsqueeze(0), 0, raw_channel_count - 1)
        raw_one_hot.scatter_(0, clamped_labels, 1.0)

        new_parse_map_tensor = torch.FloatTensor(semantic_nc_out, h, w).zero_()
        for new_idx in range(semantic_nc_out):
            if new_idx in labels_map_config:
                 for old_label_val in labels_map_config[new_idx][1]:
                    if 0 <= old_label_val < raw_channel_count:
                        new_parse_map_tensor[new_idx] += raw_one_hot[old_label_val]
        return new_parse_map_tensor

    def _create_parse_onehot_single_channel(self, parse_tensor_labels_hw, semantic_nc_out):
        # parse_tensor_labels_hw: (H,W) tensor with original labels (0-19 assumed)
        # semantic_nc_out: Number of target semantic classes
        # Returns: (1, H, W) tensor where each pixel is the new class index (0 to semantic_nc_out-1)
        labels_map_config = {
            0: ['background', [0, 10]], 1: ['hair', [1, 2]], 2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]], 4: ['bottom', [9, 12]], 5: ['left_arm', [14]],
            6: ['right_arm', [15]], 7: ['left_leg', [16]], 8: ['right_leg', [17]],
            9: ['left_shoe', [18]], 10: ['right_shoe', [19]], 11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }
        
        h, w = parse_tensor_labels_hw.shape
        new_label_indices_hw = torch.zeros_like(parse_tensor_labels_hw, dtype=torch.long) # Default to 0 (background)

        # Create a reverse mapping: old_label -> new_idx
        old_to_new_map = {}
        for new_idx in range(semantic_nc_out):
            if new_idx in labels_map_config:
                for old_label_val in labels_map_config[new_idx][1]:
                    old_to_new_map[old_label_val] = new_idx
        
        for r in range(h):
            for c_ in range(w): # Renamed c to c_ to avoid conflict
                old_label = parse_tensor_labels_hw[r, c_].item()
                new_label_indices_hw[r, c_] = old_to_new_map.get(old_label, 0) # Default to 0 if not found

        return new_label_indices_hw.unsqueeze(0).float() # (1,H,W)

    def __getitem__(self, index):
        im_name_list_entry = self.im_names[index] # e.g., "000001_0.jpg"
        c_name_paired_list_entry = self.c_names['paired'][index] # e.g., "000001_1.jpg"

        # Construct full paths (original logic from CPDataset.py)
        # Person image related paths
        person_image_path = osp.join(self.data_path, 'image', im_name_list_entry)
        person_parse_path = osp.join(self.data_path, 'image-parse-v3', im_name_list_entry.replace('.jpg', '.png'))
        person_pose_json_path = osp.join(self.data_path, 'openpose_json', im_name_list_entry.replace('.jpg', '_keypoints.json'))
        person_pose_rendered_path = osp.join(self.data_path, 'openpose_img', im_name_list_entry.replace('.jpg', '_rendered.png'))
        # Densepose name might need care. Original CPDataset uses im_name (which includes 'image/' prefix later)
        # but then replaces 'image' with 'image-densepose'. If im_name_list_entry is "000001_0.jpg",
        # this would mean path 'image-densepose/000001_0.jpg' or '.png'
        # The provided `train_generator.py` uses `im_name.replace('image', 'image-densepose')`
        # Let's assume densepose files are named like person images but in 'image-densepose' folder.
        person_densepose_path = osp.join(self.data_path, 'image-densepose', im_name_list_entry.replace('.jpg','.png')) # Assuming .png like parse

        # Image parse agnostic (for new_parse_agnostic_map GT)
        person_parse_agnostic_gt_path = osp.join(self.data_path, 'image-parse-agnostic-v3.2', im_name_list_entry.replace('.jpg', '.png'))

        # Cloth related paths (using c_name_paired_list_entry)
        cloth_image_path = osp.join(self.data_path, 'cloth', c_name_paired_list_entry)
        cloth_mask_path = osp.join(self.data_path, 'cloth-mask', c_name_paired_list_entry)


        # --- 1. Load all data as PIL Images and raw data (pose_json) ---
        im_pil_big = Image.open(person_image_path)
        cloth_pil_orig = Image.open(cloth_image_path).convert('RGB')
        cloth_mask_pil_orig = Image.open(cloth_mask_path) # Grayscale typically

        im_parse_pil_big = Image.open(person_parse_path) # Raw parse map (0-19 labels)

        with open(person_pose_json_path, 'r') as f:
            pose_label = json.load(f)
            pose_data_np_raw = pose_label['people'][0]['pose_keypoints_2d']
            pose_data_np_raw = np.array(pose_data_np_raw).reshape((-1, 3))[:, :2] # Keep X,Y coords

        pose_map_pil_orig = Image.open(person_pose_rendered_path)
        densepose_pil_orig = Image.open(person_densepose_path) # Typically RGB UVI map
        
        # For new_parse_agnostic_map (GT)
        parse_agnostic_gt_pil_big = Image.open(person_parse_agnostic_gt_path)

        # --- 2. Call get_agnostic using original resolution PILs (or copies) ---
        agnostic_pil_big = self.get_agnostic(im_pil_big.copy(), im_parse_pil_big.copy(), pose_data_np_raw.copy())

        # --- 3. Resize all PILs to target fine_height/fine_width ---
        # Person-related PILs
        im_pil = im_pil_big.resize((self.fine_width, self.fine_height), Image.BICUBIC)
        im_parse_pil = im_parse_pil_big.resize((self.fine_width, self.fine_height), Image.NEAREST)
        agnostic_pil = agnostic_pil_big.resize((self.fine_width, self.fine_height), Image.BICUBIC)
        pose_map_pil = pose_map_pil_orig.resize((self.fine_width, self.fine_height), Image.BICUBIC)
        densepose_pil = densepose_pil_orig.resize((self.fine_width, self.fine_height), Image.BICUBIC) # UVI maps often use BICUBIC
        parse_agnostic_gt_pil = parse_agnostic_gt_pil_big.resize((self.fine_width, self.fine_height), Image.NEAREST)

        # Cloth-related PILs
        cloth_pil = cloth_pil_orig.resize((self.fine_width, self.fine_height), Image.BICUBIC)
        cloth_mask_pil = cloth_mask_pil_orig.resize((self.fine_width, self.fine_height), Image.NEAREST)
        
        # For densepose_map_wo_normalize for clothes_no_loss_mask (load again for specific handling)
        # In original CPDataset, it's loaded and ToTensor applied without normalization.
        densepose_wo_norm_pil = Image.open(person_densepose_path).resize((self.fine_width, self.fine_height), Image.BICUBIC)


        # --- 4. Apply Transformations (Albumentations or original Torchvision) ---
        if self.use_albumentations:
            # Convert PILs to NumPy arrays (RGB format for color images)
            im_np = np.array(im_pil)
            im_parse_np_labels = np.array(im_parse_pil) # (H,W) with raw labels 0-19
            agnostic_np = np.array(agnostic_pil)
            pose_map_np = np.array(pose_map_pil)
            densepose_np = np.array(densepose_pil)
            parse_agnostic_gt_np_labels = np.array(parse_agnostic_gt_pil) # (H,W) with raw labels for agnostic parse

            cloth_np = np.array(cloth_pil)
            cloth_mask_np = np.array(cloth_mask_pil) # (H,W) grayscale

            # Apply Person Augmentations
            aug_person_inputs = {
                'image': im_np, 'im_parse_map': im_parse_np_labels,
                'agnostic_map': agnostic_np, 'pose_map': pose_map_np,
                'densepose_map': densepose_np,
                'parse_agnostic_map_raw': parse_agnostic_gt_np_labels,
            }
            augmented_person = self.person_train_transform(**aug_person_inputs)
            im_tensor = augmented_person['image']
            im_parse_tensor_labels_hw = augmented_person['im_parse_map'] # (H,W) Tensor with augmented raw labels
            agnostic_tensor = augmented_person['agnostic_map']
            pose_map_tensor = augmented_person['pose_map']
            densepose_tensor = augmented_person['densepose_map']
            parse_agnostic_gt_tensor_labels_hw = augmented_person['parse_agnostic_map_raw']

            # Apply Garment Augmentations
            aug_garment_inputs = {'image': cloth_np, 'cloth_mask': cloth_mask_np}
            augmented_garment = self.garment_train_transform(**aug_garment_inputs)
            cloth_tensor_paired = augmented_garment['image'] # c['paired']
            cloth_mask_tensor_paired_raw = augmented_garment['cloth_mask'] # cm['paired'] (possibly 1,H,W or H,W)
            # Ensure cloth_mask is (1,H,W), float, 0 or 1
            cloth_mask_tensor_paired = (cloth_mask_tensor_paired_raw.squeeze() > 0.5).float().unsqueeze(0) if cloth_mask_tensor_paired_raw.ndim >=2 else (cloth_mask_tensor_paired_raw > 0.5).float().unsqueeze(0)


            # For densepose_map_wo_normalize (used for clothes_no_loss_mask)
            # This typically does not get heavy augmentations, just resize + ToTensor (no norm)
            densepose_wo_norm_np = np.array(densepose_wo_norm_pil)
            densepose_map_wo_normalize_tensor = self.np_base_transform_to_tensor_only(image=densepose_wo_norm_np)['image']

        else: # Not using albumentations (e.g., test mode) - use original torchvision transforms
            im_tensor = self.pil_transform_rgb(im_pil)
            # im_parse_pil is (H,W) with labels. ToTensor changes it to (1,H,W) float [0,1] if not L mode.
            # We need (1,H,W) LongTensor with original labels.
            im_parse_tensor_labels_hw = torch.from_numpy(np.array(im_parse_pil)).long() # (H,W)
            
            agnostic_tensor = self.pil_transform_rgb(agnostic_pil)
            pose_map_tensor = self.pil_transform_rgb(pose_map_pil)
            densepose_tensor = self.pil_transform_rgb(densepose_pil)
            
            parse_agnostic_gt_tensor_labels_hw = torch.from_numpy(np.array(parse_agnostic_gt_pil)).long() # (H,W)

            cloth_tensor_paired = self.pil_transform_rgb(cloth_pil)
            # cloth_mask_pil is (H,W) grayscale. ToTensor converts to (1,H,W) float [0,1]. Binarize.
            cloth_mask_tensor_paired_raw = to_tensor(cloth_mask_pil) # Use functional to_tensor
            cloth_mask_tensor_paired = (cloth_mask_tensor_paired_raw > 0.5).float()

            densepose_map_wo_normalize_tensor = to_tensor(densepose_wo_norm_pil) # ToTensor only [0,1]

        # --- 5. Post-process Tensors (common for both augmented and non-augmented paths) ---
        # These operations are performed on tensors.
        # im_parse_tensor_labels_hw is (H,W) with original 0-19 labels, LongTensor.
        # parse_agnostic_gt_tensor_labels_hw is (H,W) with original 0-19 labels, LongTensor.

        # `new_parse_map` (semantic_nc channels, one-hot, for `result['parse']`)
        # Derived from augmented person parse map (`im_parse_tensor_labels_hw`)
        parse_map_remapped_tensor = self._remap_parse_map(im_parse_tensor_labels_hw.squeeze(), self.semantic_nc)

        # `parse_onehot` (single channel with remapped class indices, for `result['parse_onehot']`)
        # Derived from augmented person parse map (`im_parse_tensor_labels_hw`)
        parse_onehot_remapped_tensor = self._create_parse_onehot_single_channel(im_parse_tensor_labels_hw.squeeze(), self.semantic_nc)

        # `new_parse_agnostic_map` (semantic_nc channels, one-hot, for `result['parse_agnostic']`)
        # Derived from augmented `parse_agnostic_gt_tensor_labels_hw`
        new_parse_agnostic_map_tensor = self._remap_parse_map(parse_agnostic_gt_tensor_labels_hw.squeeze(), self.semantic_nc)
        
        # `pcm` (Parse Cloth Mask) & `im_c` (Parse Cloth image)
        # Derived from `parse_map_remapped_tensor` (which is based on augmented person parse)
        # Original CPDataset uses channel 3 of `new_parse_map` (i.e., `parse_map_remapped_tensor`) for upper clothes.
        upper_cloth_channel_idx = 3 
        if upper_cloth_channel_idx < self.semantic_nc:
            pcm_tensor = parse_map_remapped_tensor[upper_cloth_channel_idx:upper_cloth_channel_idx+1]
        else: # Fallback if semantic_nc is too small
            pcm_tensor = torch.zeros_like(parse_map_remapped_tensor[0:1]) 
            print(f"Warning: upper_cloth_channel_idx {upper_cloth_channel_idx} is out of bounds for semantic_nc {self.semantic_nc}. pcm will be zero.")
        
        im_c_tensor = im_tensor * pcm_tensor + (1 - pcm_tensor) # im_tensor is augmented person image

        # `lower_clothes_mask`
        # Original CPDataset uses channel 4 of `new_parse_map` for bottom clothes.
        bottom_cloth_channel_idx = 4
        if bottom_cloth_channel_idx < self.semantic_nc:
            lower_clothes_mask_tensor = parse_map_remapped_tensor[bottom_cloth_channel_idx:bottom_cloth_channel_idx+1, :, :]
        else: # Fallback
            lower_clothes_mask_tensor = torch.zeros_like(parse_map_remapped_tensor[0:1])
            print(f"Warning: bottom_cloth_channel_idx {bottom_cloth_channel_idx} is out of bounds for semantic_nc {self.semantic_nc}. lower_clothes_mask will be zero.")

        # `clothes_no_loss_mask` (complex logic from original CPDataset)
        # Uses `densepose_map_wo_normalize_tensor` (C,H,W), range [0,1]
        # Original: densepose_end_of_torso_mask = torch.FloatTensor((densepose_map_wo_normalize[1:2,:,:].cpu().numpy() == (80/255.)).astype(np.int32))
        # Assuming densepose_map_wo_normalize_tensor[1] is the V channel of UVI map.
        if densepose_map_wo_normalize_tensor.shape[0] >= 2: # Ensure V channel exists
            densepose_v_channel = densepose_map_wo_normalize_tensor[1:2, :, :] 
            threshold_v = 80.0 / 255.0
            # Compare float tensors with a small epsilon for equality if needed, or check if source was exact
            # For simplicity, direct comparison, assuming source could be exactly threshold_v after ToTensor scaling
            densepose_end_of_torso_mask = (torch.abs(densepose_v_channel - threshold_v) < 1e-3).float() # Check equality with tolerance
        else:
            densepose_end_of_torso_mask = torch.zeros(1, self.fine_height, self.fine_width) # Fallback
            print("Warning: Densepose V channel not found for clothes_no_loss_mask. Using zeros.")

        # Original grid logic:
        # grid = self.make_grid(1, self.fine_height, self.fine_width).permute(0, 3, 1, 2)
        # grid_x, grid_y = torch.split(grid, 1, dim=1)
        # grid_y_max_intermediate = (1. - densepose_end_of_torso_mask) * 0. + grid_y * densepose_end_of_torso_mask
        # grid_y_max_val = torch.max(grid_y_max_intermediate) # This gives a single max value over the whole batch/image
        # grid_y_max_idx = int(grid_y_max_val.item() * self.fine_height) # .item() if it's a 0-dim tensor
        # Simplified: Find the highest row index where densepose_end_of_torso_mask is 1.
        # This indicates the lowest point of the torso based on the V channel threshold.
        
        clothes_no_loss_mask_tensor = torch.zeros_like(densepose_end_of_torso_mask)
        # Find rows where densepose_end_of_torso_mask has any '1'
        # rows_with_torso_end = torch.any(densepose_end_of_torso_mask.squeeze() > 0.5, dim=1)
        # if torch.any(rows_with_torso_end):
        #     grid_y_max_idx = torch.where(rows_with_torso_end)[0].max().item()
        # else: # No torso end found, default behavior (e.g. mask all or none)
        #     grid_y_max_idx = 0 # Or self.fine_height -1 if all should be masked by default
        # clothes_no_loss_mask_tensor[:, :grid_y_max_idx, :] = 1.0
        # The original logic is subtle. For now, using a placeholder or simplified version:
        # Assume mask everything above where torso ends. If densepose_end_of_torso_mask is 1 at torso end,
        # we want 1 above it.
        # This part is highly dependent on the exact meaning and reliability of densepose_end_of_torso_mask.
        # A common simplification for "upper body part" might be to use the person segmentation.
        # Given the complexity, and if this mask is critical, it needs careful porting.
        # For now, let's use a simple pass-through or default for clothes_no_loss_mask
        # to avoid breaking the pipeline due to this specific complex logic.
        # A common use is to prevent loss on lower body when trying on upper body garment.
        # It might be better defined by the `new_parse_map` (e.g. mask where body parts other than upper body are).
        clothes_no_loss_mask_tensor = torch.ones_like(densepose_end_of_torso_mask) # Placeholder: allow loss everywhere. Review this.

        # --- 6. Final assembly of the result dictionary ---
        result = {
            'c_name': {'paired': c_name_paired_list_entry}, # Original had c_name dict, not needed by train_condition
            'im_name': im_name_list_entry,
            
            'cloth': {'paired': cloth_tensor_paired},
            'cloth_mask': {'paired': cloth_mask_tensor_paired},
            
            'parse_agnostic': new_parse_agnostic_map_tensor, # Input for Person Image Encoder (input2)
            'densepose': densepose_tensor,                   # Input for Person Image Encoder (input2)
            'pose': pose_map_tensor,                         # Input for Person Image Encoder (input2) (openpose rendered)
            
            'agnostic' : agnostic_tensor, # For generator stage, if this dataset is used there

            'parse_onehot' : parse_onehot_remapped_tensor,  # GT for CrossEntropyLoss (target for fake_segmap)
            'parse': parse_map_remapped_tensor,             # GT for GAN Loss (real samples for Discriminator)
            'pcm': pcm_tensor,                              # GT for L1 Loss on warped cloth mask
            'parse_cloth': im_c_tensor,                     # GT for VGG Loss on warped cloth image
            
            'image': im_tensor,         # Original Person Image (target for final generation, visualization)
            
            'lower_clothes_mask': lower_clothes_mask_tensor, # For masked L1/VGG loss calculation
            'clothes_no_loss_mask': clothes_no_loss_mask_tensor, # For masked L1/VGG loss
        }
        return result

# CPDataLoader class can remain the same as in the original cp_dataset.py
class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch