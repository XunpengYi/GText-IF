import cv2
import torch
import numpy as np
from torch.utils import data
import os
import random
import ast
from utils.transforms import generate_random_crop_pos, generate_random_crop_pos_with_cat_check, random_crop_pad_to_shape, normalize

def random_mirror(rgb, rgb_gt, gt, modal_x, modal_x_gt):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        rgb_gt = cv2.flip(rgb_gt, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)
        modal_x_gt = cv2.flip(modal_x_gt, 1)
    return rgb, rgb_gt, gt, modal_x, modal_x_gt

def random_vflip(rgb, rgb_gt, gt, modal_x, modal_x_gt):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 0)
        rgb_gt = cv2.flip(rgb_gt, 0)
        gt = cv2.flip(gt, 0)
        modal_x = cv2.flip(modal_x, 0)
        modal_x_gt = cv2.flip(modal_x_gt, 0)
    return rgb, rgb_gt, gt, modal_x, modal_x_gt

def random_scale(rgb, rgb_gt, gt, modal_x, modal_x_gt, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    rgb_gt = cv2.resize(rgb_gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x_gt = cv2.resize(modal_x_gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    return rgb, rgb_gt, gt, modal_x, modal_x_gt, scale

def valid_mask_scale(mask, scale):
    sh = int(mask.shape[0] * scale)
    sw = int(mask.shape[1] * scale)
    mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return mask

class TrainPre(object):
    def __init__(self, crop_size, train_scale_array, cat_max_ratio, norm_mean_rgb, norm_std_rgb, norm_mean_modal_x, norm_std_modal_x):
        self.crop_size = crop_size
        self.train_scale_array = train_scale_array
        self.cat_max_ratio = cat_max_ratio
        self.norm_mean_rgb = norm_mean_rgb
        self.norm_std_rgb = norm_std_rgb
        self.norm_mean_modal_x = norm_mean_modal_x
        self.norm_std_modal_x = norm_std_modal_x

    def __call__(self, rgb, rgb_gt, gt, modal_x, modal_x_gt):
        valid_mask = np.ones(gt.shape, dtype=np.uint8)
        rgb, rgb_gt, gt, modal_x, modal_x_gt = random_mirror(rgb, rgb_gt, gt, modal_x, modal_x_gt)

        if self.train_scale_array is not None:
            rgb, rgb_gt, gt, modal_x, modal_x_gt, scale = random_scale(rgb, rgb_gt, gt, modal_x, modal_x_gt, self.train_scale_array)
            valid_mask = valid_mask_scale(valid_mask, scale)      

        rgb = normalize(rgb, self.norm_mean_rgb, self.norm_std_rgb)
        rgb_gt = normalize(rgb_gt, self.norm_mean_rgb, self.norm_std_rgb)
        modal_x = normalize(modal_x, self.norm_mean_modal_x, self.norm_std_modal_x)
        modal_x_gt = normalize(modal_x_gt, self.norm_mean_modal_x, self.norm_std_modal_x)

        # set the crop size as the image size
        crop_size = self.crop_size
        crop_pos = generate_random_crop_pos_with_cat_check(rgb.shape[:2], crop_size, gt, cat_max_ratio=self.cat_max_ratio)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_rgb_gt, _ = random_crop_pad_to_shape(rgb_gt, crop_pos, crop_size, 0)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)
        p_modal_x_gt, _ = random_crop_pad_to_shape(modal_x_gt, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)

        p_valid_mask, _ = random_crop_pad_to_shape(valid_mask, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_rgb_gt = p_rgb_gt.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)
        p_modal_x_gt = p_modal_x_gt.transpose(2, 0, 1)
        
        return p_rgb, p_rgb_gt, p_gt, p_modal_x, p_modal_x_gt, p_valid_mask
    
class ValPre(object):
        def __init__(self, norm_mean_rgb, norm_std_rgb, norm_mean_modal_x, norm_std_modal_x):
            self.norm_mean_rgb = norm_mean_rgb
            self.norm_std_rgb = norm_std_rgb
            self.norm_mean_modal_x = norm_mean_modal_x
            self.norm_std_modal_x = norm_std_modal_x
        
        def __call__(self, rgb, rgb_gt, gt, modal_x, modal_x_gt):
            rgb = normalize(rgb, self.norm_mean_rgb, self.norm_std_rgb)
            rgb_gt = normalize(rgb_gt, self.norm_mean_rgb, self.norm_std_rgb)
            modal_x = normalize(modal_x, self.norm_mean_modal_x, self.norm_std_modal_x)
            modal_x_gt = normalize(modal_x_gt, self.norm_mean_modal_x, self.norm_std_modal_x)

            rgb = rgb.transpose(2, 0, 1)
            rgb_gt = rgb_gt.transpose(2, 0, 1)
            modal_x = modal_x.transpose(2, 0, 1)
            modal_x_gt = modal_x_gt.transpose(2, 0, 1)

            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            rgb_gt = torch.from_numpy(np.ascontiguousarray(rgb_gt)).long()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            modal_x = torch.from_numpy(np.ascontiguousarray(modal_x)).float()
            modal_x_gt = torch.from_numpy(np.ascontiguousarray(modal_x_gt)).long()

            valid_mask = torch.ones_like(gt, dtype=torch.uint8)
            return rgb, rgb_gt, gt, modal_x, modal_x_gt, valid_mask
        
class TestPre(object):
        def __init__(self, norm_mean_rgb, norm_std_rgb, norm_mean_modal_x, norm_std_modal_x):
            self.norm_mean_rgb = norm_mean_rgb
            self.norm_std_rgb = norm_std_rgb
            self.norm_mean_modal_x = norm_mean_modal_x
            self.norm_std_modal_x = norm_std_modal_x
        
        def __call__(self, rgb, rgb_gt, gt, modal_x, modal_x_gt):
            rgb = normalize(rgb, self.norm_mean_rgb, self.norm_std_rgb)
            rgb_gt = normalize(rgb_gt, self.norm_mean_rgb, self.norm_std_rgb)
            modal_x = normalize(modal_x, self.norm_mean_modal_x, self.norm_std_modal_x)
            modal_x_gt = normalize(modal_x_gt, self.norm_mean_modal_x, self.norm_std_modal_x)

            rgb = rgb.transpose(2, 0, 1)
            rgb_gt = rgb_gt.transpose(2, 0, 1)
            modal_x = modal_x.transpose(2, 0, 1)
            modal_x_gt = modal_x_gt.transpose(2, 0, 1)

            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            rgb_gt = torch.from_numpy(np.ascontiguousarray(rgb_gt)).long()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            modal_x = torch.from_numpy(np.ascontiguousarray(modal_x)).float()
            modal_x_gt = torch.from_numpy(np.ascontiguousarray(modal_x_gt)).long()

            valid_mask = torch.ones_like(gt, dtype=torch.uint8)
            return rgb, rgb_gt, gt, modal_x, modal_x_gt, valid_mask

def get_train_loader(config, engine, dataset):
    data_setting = {'rgb_root': config['dataset']['train']['rgb_path'],
                    'rgb_gt_root': config['dataset']['train']['rgb_gt_path'],
                    'x_root':config['dataset']['train']['x_path'],
                    'x_gt_root': config['dataset']['train']['x_gt_path'],
                    'text_root': config['dataset']['train']['text_path'],
                    'label_root': config['dataset']['train']['label_path'],
                    'label_transform': config['dataset']['train']['label_transform'],
                    'class_names': config['class_names'],
                    'text_input_mode': config['dataset']['train']['text_input_mode']}
    train_preprocess = TrainPre(config['dataset']['train']['crop_size'], config['dataset']['train']['train_scale_array'], 
                                config['dataset']['train']['cat_max_ratio'],
                                config['dataset']['train']['mean_rgb'], config['dataset']['train']['std_rgb'],
                                config['dataset']['train']['mean_x'], config['dataset']['train']['std_x'])

    train_dataset = dataset(data_setting, "train", train_preprocess)

    train_data_size = len(train_dataset)

    train_sampler = None
    is_shuffle = True
    batch_size = config['train']['batch_size']

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config['train']['batch_size'] // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config['train']['num_workers'],
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler, train_data_size

def get_val_loader(config, engine, dataset):
    data_setting = {'rgb_root': config['dataset']['val']['rgb_path'],
                    'rgb_gt_root': config['dataset']['val']['rgb_gt_path'],
                    'x_root': config['dataset']['val']['x_path'],
                    'x_gt_root': config['dataset']['val']['x_gt_path'],
                    'text_root': config['dataset']['val']['text_path'],
                    'label_root': config['dataset']['val']['label_path'],
                    'label_transform': config['dataset']['val']['label_transform'],
                    'class_names': config['class_names'],
                    'text_input_mode': config['dataset']['val']['text_input_mode']}

    val_preprocess = ValPre(config['dataset']['train']['mean_rgb'], config['dataset']['train']['std_rgb'],
                            config['dataset']['train']['mean_x'], config['dataset']['train']['std_x'])

    val_sampler = None
    is_shuffle = False
    batch_size = 1

    val_dataset = dataset(data_setting, 'val', val_preprocess)

    val_data_size = len(val_dataset)

    if engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    val_loader = data.DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   num_workers=config['train']['num_workers'],
                                   drop_last=False,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=val_sampler)
    return val_loader, val_sampler, val_data_size

def get_val_rank0_loader(config, engine, dataset):
    data_setting = {'rgb_root': config['dataset']['val']['rgb_path'],
                    'rgb_gt_root': config['dataset']['val']['rgb_gt_path'],
                    'x_root':config['dataset']['val']['x_path'],
                    'x_gt_root': config['dataset']['val']['x_gt_path'],
                    'text_root': config['dataset']['val']['text_path'],
                    'label_root': config['dataset']['val']['label_path'],
                    'label_transform': config['dataset']['val']['label_transform'],
                    'class_names': config['class_names'],
                    'text_input_mode': config['dataset']['val']['text_input_mode']}

    val_preprocess = ValPre(config['dataset']['train']['mean_rgb'], config['dataset']['train']['std_rgb'],
                            config['dataset']['train']['mean_x'], config['dataset']['train']['std_x'])

    is_shuffle = False
    batch_size = 1

    val_dataset = dataset(data_setting, 'val', val_preprocess)

    val_data_size = len(val_dataset)

    val_loader = data.DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   num_workers=config['train']['num_workers'],
                                   drop_last=False,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=None)
    return val_loader, val_data_size