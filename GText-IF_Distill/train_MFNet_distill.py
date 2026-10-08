import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from PIL import Image

from dataloader.dataloader import get_train_loader, get_val_loader, get_val_rank0_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, reduce_value
from utils.script import load_config, save_config_to_yaml
from utils.metric import hist_info, compute_metric
from loss.loss_distill import loss_distill_text
import cv2
import pprint

from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default="./configs/config_MFNet_fus_distill.yaml", type=str, help='Path to the config file')
parser.add_argument('--devices', default='0', help='set data parallel training')
parser.add_argument('--pretrained_weight', default='./weights_waiting_for_distill/MFNet/epoch-best.pth', help='weight path waiting for distill')
parser.add_argument('--port', type=str, default='16002', dest="port", help='port for init_process_group')

logger = get_logger()

os.environ['MASTER_PORT'] = '19502'  #169710

import torchvision.transforms as T
def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return tensor * std + mean

def tensor_to_image(tensor, index, save_path):
    img = tensor[index].cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{save_path}_img{index}.png", img)

def tensor_to_np(tensor):
    img = tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def tensor_to_cat_image(rgbs, rgbs_gt, modal_xs, modal_xs_gt, black_st_one, fus):
    rgbs = tensor_to_np(rgbs.squeeze(0))
    modal_xs = tensor_to_np(modal_xs.squeeze(0))
    rgbs_gt = tensor_to_np(rgbs_gt.squeeze(0))
    modal_xs_gt = tensor_to_np(modal_xs_gt.squeeze(0))
    black_st_one = tensor_to_np(black_st_one.squeeze(0))
    fus = tensor_to_np(fus.squeeze(0))
    img_row1 = np.concatenate((rgbs, modal_xs, black_st_one), axis=1)
    img_row2 = np.concatenate((rgbs_gt, modal_xs_gt, fus), axis=1)

    img = np.concatenate((img_row1, img_row2), axis=0)
    return img

def tensor_to_mask(tensor, index, save_path):
    mask = tensor[index].cpu().to(torch.uint8)
    mask = T.ToPILImage()(mask)
    mask.save(f"{save_path}_label_area_{index}.png")

def save_batch_as_images(rgbs, gts, modal_xs, save_dir, idx, flag):
    mean = [0.5, 0.5, 0.5] 
    std = [0.5, 0.5, 0.5]
    rgbs = unnormalize(rgbs, mean, std)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    modal_xs = unnormalize(modal_xs, mean, std)
    batch_size = rgbs.size(0)
    for i in range(batch_size):
        tensor_to_image(rgbs, i, f"{save_dir}/rgbs_{flag}_{idx}")
        tensor_to_mask(gts, i, f"{save_dir}/gts_{flag}_{idx}")
        tensor_to_image(modal_xs, i, f"{save_dir}/modal_xs_{flag}_{idx}")

def save_masks_as_images(masks, save_dir, idx, prefix="mask"):
    if masks.dim() == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)

    batch_size = masks.size(0)
    for i in range(batch_size):
        mask = masks[i].cpu().numpy()
        mask_img = (mask * 255).astype(np.uint8)

        img_pil = Image.fromarray(mask_img)
        save_path = os.path.join(save_dir, f"{prefix}_{idx}_{i}.jpg")
        img_pil.save(save_path)

def save_val_as_images(rgbs, rgbs_gt, fus, modal_xs, modal_xs_gt, save_dir, name):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    fus = unnormalize(fus.cpu(), mean, std)
    rgbs = unnormalize(rgbs.cpu(), mean, std)
    rgbs_gt = unnormalize(rgbs_gt.cpu(), mean, std)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    modal_xs = unnormalize(modal_xs.cpu(), mean, std)
    modal_xs_gt = unnormalize(modal_xs_gt.cpu(), mean, std)

    black_st_one = torch.zeros_like(fus)

    img = tensor_to_cat_image(rgbs, rgbs_gt, modal_xs, modal_xs_gt, black_st_one, fus)
    cv2.imwrite(f"{save_dir}/img_{name[0]}.jpg", img)

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    
    config = load_config(args.config_path)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    seed = config['seed']
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # data loader
    train_loader, train_sampler, train_data_size = get_train_loader(config, engine, RGBXDataset)
    val_loader, val_data_size = get_val_rank0_loader(config, engine, RGBXDataset)

    niters_per_epoch = len(train_loader)
    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        logger.info('Training data size: {}'.format(train_data_size))
        logger.info('Iters per epoch: {}'.format(niters_per_epoch))
        logger.info('Testing data size: {}'.format(val_data_size))

    exp_dir = 'fus_exp_paper/' + config['name'] + "_" + config['model']['backbone'] + '_{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    model_save_dir = exp_dir + '/model'
    train_img_save_dir = exp_dir + '/model' + '/train_img'
    val_img_save_dir = exp_dir + '/val_img'
    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        if os.path.exists(exp_dir) is False:
            os.makedirs(exp_dir)
            os.makedirs(model_save_dir)
            os.makedirs(train_img_save_dir)
            os.makedirs(val_img_save_dir)

    log_file_path = os.path.join(exp_dir, "training.log")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = exp_dir + '/tb_logger'
        tb = SummaryWriter(log_dir=tb_dir)

    background_id = int(config['background_id'])
    logger.info('Cross_Entropy ignore the background id: {}'.format(background_id))

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        config_dir = exp_dir
        save_config_to_yaml(config, config_dir)

    criterion = loss_distill_text()

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model = segmodel(cfg=config, norm_layer=BatchNorm2d)

    model.init_func()
    base_lr = config['train']['learning_rate']
    
    params_list = []
    params_list = model.text_distill_net.parameters()
    
    if config['train']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config['train']['weight_decay'])
    elif config['train']['optimizer'] == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=0.9, weight_decay=config['train']['weight_decay'])
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config['train']['num_epochs'] * niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config['train']['lr_power'], total_iteration, niters_per_epoch * config['train']['warm_up_epoch'])

    if engine.distributed:
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    
    state_dict = torch.load(args.pretrained_weight, map_location=device)
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    elif 'module' in state_dict.keys():
        state_dict = state_dict['module']
    model.load_state_dict(state_dict, strict=False)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.eval()
    model.text_distill_net.train()
    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        logger.info('begin trainning:')

    best_val_loss = 1e3
    best_epoch = 0
    best_mIoU = 0.0
    best_mIoU_epoch = 0

    for epoch in range(engine.state.epoch, config['train']['num_epochs'] + 1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        sum_loss_seg = 0
        sum_loss_fus = 0

        torch.set_printoptions(threshold=float('inf'))

        for idx in pbar:
            model.text_distill_net.train()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            rgbs = minibatch['rgb']
            modal_xs = minibatch['modal_x']

            text = minibatch['text']

            rgbs = rgbs.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)


            if engine.distributed:
                text_token = model.module.tokenize(text).cuda(non_blocking=True)
            else:
                text_token = model.tokenize(text).cuda(non_blocking=True)

            with torch.no_grad():
                text_feat = model.text_encoder.encode_text_full(text_token)

            # modal_xs: B, 3, H, W     gts_shape: B, H, W
            pred_text = model.text_distill(rgbs, modal_xs)

            loss_distill = criterion(pred_text, text_feat.detach())

            loss = loss_distill

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            # auto reduce the loss, no need all_reduce_tensor manually
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config['train']['num_epochs']) \
                        + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config['train']['num_epochs']) \
                        + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))

            del loss
            pbar.set_description(print_str, refresh=False)
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            tb.add_scalar('train_loss_seg', sum_loss_seg / len(pbar), epoch)
            tb.add_scalar('train_loss_fus', sum_loss_fus / len(pbar), epoch)

        if not engine.distributed or torch.distributed.get_rank() == 0:
            if (epoch >= config['train']['val_start_epoch']) or (epoch == config['train']['num_epochs']):
                model.text_distill_net.eval()
                if epoch % config['train']['save_step'] == 0:
                    engine.save_and_link_checkpoint(model_save_dir)
                        
                logger.info('######## Single GPU Validation ########')
                
                sum_val_loss = 0
                val_img_save_dir_per_epoch = os.path.join(val_img_save_dir, str(epoch))
                if os.path.exists(val_img_save_dir_per_epoch) is False:
                    os.makedirs(val_img_save_dir_per_epoch)

                all_result = []
                bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
                pbar = tqdm(range(len(val_loader)), file=sys.stdout, bar_format=bar_format)
                dataloader = iter(val_loader)
                for idx in pbar:
                    print_str = 'validation {}/{}'.format(idx + 1, len(val_loader))
                    pbar.set_description(print_str, refresh=False)

                    data = dataloader.next()
                    rgb = data['rgb']
                    modal_x = data['modal_x']
                    label = data['label']
                    text = data['text']
                    name = data['name']

                    rgb = rgb.cuda(non_blocking=True)
                    modal_x = modal_x.cuda(non_blocking=True)
                    if engine.distributed:
                        text = model.module.tokenize(text).cuda(non_blocking=True)
                    else:
                        text = model.tokenize(text).cuda(non_blocking=True)

                    with torch.no_grad():
                        if engine.distributed:
                            score_semantic, pred_visual = model.module(rgb, modal_x, text)
                        else:
                            score_semantic, pred_visual = model(rgb, modal_x, text)

                        val_loss_seg = criterion(score_semantic, label.cuda(non_blocking=True).long())

                        score_semantic = torch.exp(score_semantic[0])
                        score_semantic = score_semantic.permute(1, 2, 0)
                        processed_pred = score_semantic.cpu().numpy()
                        pred = processed_pred.argmax(2)
                    result = hist_info(config['model']['num_classes'], pred, label.squeeze().numpy())[0]
                    all_result.append(result)
                    sum_val_loss += val_loss_seg

                    save_val_as_images(rgb, rgb, pred_visual, modal_x, modal_x, val_img_save_dir_per_epoch, name)

                iou, mIoU = compute_metric(all_result, config['model']['num_classes'])

                result_title = f"{'Class Name':<15} | {'IoU':>10}\n" + "-"*30 + "\n"

                result_mIoU = [f"{config['class_names'][i]:<15} | {(iou[i] * 100):>10.3f}%\n" for i in range(config['model']['num_classes'])]
                result_mIoU.append("-" * 30 + "\n")
                result_mIoU.append(f"{'Mean IoU':<15} | {(mIoU[0] * 100):>10.3f}%\n")
                result_output = result_title + "".join(result_mIoU)
                print(result_output)
                print(f"Best mIoU: {(max(mIoU[0], best_mIoU)*100):.3f}     Best Epoch: {best_mIoU_epoch}")

                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Epoch {epoch}:" + "\n")
                    log_file.write(result_output + "\n")
                    log_file.write(f"Best mIoU: {(max(mIoU[0], best_mIoU)*100):.3f}   Best Epoch: {best_mIoU_epoch}" + "\n\n")

                tb.add_scalar('mIoU', mIoU[0], epoch)
                tb.add_scalar('val_loss_seg', sum_val_loss / len(pbar), epoch)
                if mIoU[0] > best_mIoU:
                    best_mIoU = mIoU[0]
                    best_mIoU_epoch = epoch
                    engine.save_checkpoint(os.path.join(model_save_dir, "epoch-best.pth"))

        if engine.distributed:
            torch.distributed.barrier()