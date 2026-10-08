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
from dataloader.RGBXDataset import RGBXDataset
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
        [0, 0, 0],  # unlabelled
        [128, 0, 64],  # car
        [0, 64, 64],  # person
        [192, 128, 0],  # bike
        [192, 0, 0],  # curve
        [0, 128, 128],  # car_stop
        [128, 64, 64],  # guardrail
        [128, 128, 192],  # color_cone
        [0, 64, 192],  # bump
    ]
    return pattale

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        rgb = data['rgb']
        label = data['label']
        modal_x = data['modal_x']
        text = data['text']

        name = data['name']
        pred_semantic, pred_visual = self.sliding_eval_rgbX(rgb, modal_x, text, args.eval_crop_size, device)

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        pred_semantic = pred_semantic.detach().cpu().numpy()
        pred_visual = unnormalize(pred_visual, mean, std)
        pred_visual = pred_visual.detach().squeeze(0).cpu().numpy()
        label = label.detach().cpu().numpy()
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config['model']['num_classes'], pred_semantic, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_seg')
            ensure_dir(self.save_path + '_fusion')
            fn = name + '.png'
            class_colors = get_class_colors()
            temp = np.zeros((pred_semantic.shape[0], pred_semantic.shape[1], 3))
            for i in range(self.class_num):
                temp[pred_semantic == i] = class_colors[i]

            # save raw result
            cv2.imwrite(os.path.join(self.save_path+'_seg', fn), temp)
            cv2.imwrite(os.path.join(self.save_path, fn), pred_semantic)

            visual_img = np.transpose(pred_visual, (1, 2, 0))
            visual_img = np.clip(visual_img * 255, 0, 255).astype(np.uint8)
            visual_img = cv2.cvtColor(visual_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.save_path + '_fusion', fn), visual_img)

            logger.info('Saved the image: ' + fn)
            
        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config['model']['num_classes'], config['model']['num_classes']))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)
        return result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./configs/config_MFNet_fus.yaml", type=str, help='Path to the config file')
    parser.add_argument('--devices', default='0', type=str)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', default=False, action='store_true')

    parser.add_argument('--eval_scale_array', default=[1], type=list, help='')
    parser.add_argument('--eval_flip', default=False, type=bool, help='')
    parser.add_argument('--eval_crop_size', default=None, type=list, help='')

    parser.add_argument('--checkpoint_dir', default="./weights/MFNet/model", type=str, help='')
    parser.add_argument('--epochs', default='best', type=str)

    exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    log_dir = f"./results/log_GText-IF_MFNet_{exp_time}"
    parser.add_argument('--save_path', default=os.path.join(log_dir, "img"))
    parser.add_argument('--val_log_file', default=os.path.join(log_dir, f"val_{exp_time}.log"), type=str, help='')
    parser.add_argument('--link_val_log_file', default=os.path.join(log_dir, "val_last.log"), type=str, help='')

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
    
    dataset = RGBXDataset(data_setting, 'val', test_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset=dataset, class_num=config['model']['num_classes'], norm_mean_rgb=config['dataset']['val']['mean_rgb'],
                                 norm_std_rgb=config['dataset']['val']['std_rgb'], norm_mean_x=config['dataset']['val']['mean_x'],
                                 norm_std_x=config['dataset']['val']['std_x'], network=network,
                                 multi_scales=args.eval_scale_array, is_flip=args.eval_flip,
                                 devices=all_dev, verbose=args.verbose, save_path=args.save_path,
                                 show_image=args.show_image)
        segmentor.run(args.checkpoint_dir, args.epochs, args.val_log_file,
                      args.link_val_log_file)