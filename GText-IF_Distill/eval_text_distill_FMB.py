import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset, RGBXDataset_distill
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import TestPre
from utils.script import load_config
import time
import random

logger = get_logger()

def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

def get_class_colors():
    pattale = [
        [0, 0, 0],  # background  0
        [228, 228, 179],  # Road 1
        [133, 57, 181],  # Sidewalk 2
        [177, 162, 67],  # Building 3
        [50, 178, 200],  # Lamp 4
        [199, 45, 132],  # Sign 5
        [84, 172, 66],  # Vegetation 6
        [79, 73, 179],  # Sky 7
        [166, 99, 76],  # Person 8
        [253, 121, 66],  # Car 9
        [91, 165, 137],  # Truck 10
        [152, 97, 155],  # Bus 11
        [140, 153, 105],  # Motorcycle 12
        [158, 215, 222],  # Bicycle 13
        [90, 113, 135],  # Pole 14
    ]
    return pattale

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        rgb = data['rgb']
        modal_x = data['modal_x']
        text = data['text']
        name = data['name']

        _, pred_visual = self.sliding_eval_rgbX(
            rgb, modal_x, text, args.eval_crop_size, device
        )

        if self.save_path is not None:
            fusion_dir = self.save_path + '_fusion'
            ensure_dir(fusion_dir)

            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            pred_visual = unnormalize(pred_visual, mean, std)
            pred_visual = pred_visual.detach().squeeze(0).cpu().numpy()

            visual_img = np.transpose(pred_visual, (1, 2, 0))
            visual_img = np.clip(
                visual_img * 255, 0, 255
            ).astype(np.uint8)
            visual_img = cv2.cvtColor(
                visual_img, cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(os.path.join(fusion_dir, name), visual_img)

            logger.info('Saved the fusion image: ' + name)

        return {}

    def compute_metric(self, results):
        return f'Fusion inference completed: {len(results)} images.\n'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./configs/config_other_data_fus_distill_FMB.yaml", type=str, help='Path to the config file')
    parser.add_argument('--devices', default='0', type=str)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', default=False, action='store_true')

    parser.add_argument('--eval_scale_array', default=[1], type=list, help='')
    parser.add_argument('--eval_flip', default=False, type=bool, help='')
    parser.add_argument('--eval_crop_size', default=None, type=list, help='')

    parser.add_argument('--checkpoint_dir', default="./fus_exp/FMB_train_fus_distill/model", type=str, help='')
    parser.add_argument('--epochs', default='best', type=str)

    exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    log_dir = f"./results/log_GText-IF_FMB_weights_{exp_time}"
    parser.add_argument('--save_path', default=os.path.join(log_dir, "img"))
    
    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    config = load_config(args.config_path)

    seed = config['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    network = segmodel(cfg=config, norm_layer=nn.BatchNorm2d)
    
    data_setting = {'rgb_root': config['dataset']['val']['rgb_path'],
                    'rgb_gt_root': config['dataset']['val']['rgb_gt_path'],
                    'x_root': config['dataset']['val']['x_path'],
                    'x_gt_root': config['dataset']['val']['x_gt_path'],
                    'text_root': config['dataset']['val']['text_path'],
                    'label_root': config['dataset']['val']['label_path'],
                    'label_transform': config['dataset']['val']['label_transform'],
                    'class_names': config['class_names'],
                    'text_input_mode': config['dataset']['val']['text_input_mode']}
    
    test_pre = TestPre(config['dataset']['val']['mean_rgb'], config['dataset']['val']['std_rgb'], 
                       config['dataset']['val']['mean_x'], config['dataset']['val']['std_x'])
    
    dataset = RGBXDataset_distill(data_setting, 'val', test_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset=dataset, class_num=config['model']['num_classes'], norm_mean_rgb=config['dataset']['val']['mean_rgb'],
                                 norm_std_rgb=config['dataset']['val']['std_rgb'], norm_mean_x=config['dataset']['val']['mean_x'],
                                 norm_std_x=config['dataset']['val']['std_x'], network=network,
                                 multi_scales=args.eval_scale_array, is_flip=args.eval_flip,
                                 devices=all_dev, verbose=args.verbose, save_path=args.save_path,
                                 show_image=args.show_image)
        segmentor.run(args.checkpoint_dir, args.epochs)