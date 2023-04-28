import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
import numpy as np
import argparse
import logging
import os
import pprint
import shutil
import wandb
import sys
from tqdm import tqdm
sys.path.append('./')


import yaml
from datasets.data_builder import build_dataloader, build_custom_dataloader
from easydict import EasyDict
from models.model_helper import build_network
from datasets.custom_exemplar_dataset import build_custom_exemplar_dataloader
from utils.criterion_helper.criterion_helper import _MSELoss 
from utils.dist_helper import setup_distributed
from utils.eval_helper import dump, merge_together, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    to_device,
    AverageMeter
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import build_visualizer, Visualizer


parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument(
    "-c", "--config", type=str, default="/content/SAFECount/experiments/ShanghaiTech/part_A/config.yaml", help="Path of config"
)
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("--local_rank", default=0, help="local rank for dist")
parser.add_argument('--resume', default=False, help='Resume training.')
parser.add_argument('--checkpoint_path', default='/content/checkpoints/model_4.pth')


def train(model, loader, valid_loader, criterion, opt, scheduler, config, scaler, vis_obj, epoch):

    model.train()

    # Can be added to the main function
    model.module.backbone.eval()
    for p in model.module.backbone.parameters():
        p.requires_grad = False

    train_loss_ = AverageMeter()
    weight = config.criterion[0].kwargs.weight
    wandb.log({'Epoch': epoch})
    loop = tqdm(loader, position = 0, leave = True)

    for idx, data in enumerate(loop):

        data = to_device(data, torch.device("cuda"))
        current_lr = scheduler.get_lr()[0]
        train_loss = 0
        opt.zero_grad(set_to_none = True)
        
        with torch.cuda.amp.autocast_mode.autocast():

            pred = model(data)
            train_loss = weight * criterion(pred)

        scaler.scale(train_loss).backward()
        scaler.step(opt)
        scaler.update()

        density = pred['density']
        density_pred = pred['density_pred']
        train_loss_.update(train_loss.detach(), 1)
        gt_count = torch.sum(density).item()
        pred_count = torch.sum(density_pred).item()

        if idx % 500 == 0:

            wandb.log({
                'Loss': train_loss_.avg,
                'LR': current_lr,
                'Difference': abs(gt_count - pred_count)
            })

            text = f'GT: {int(gt_count)} & pred: {int(pred_count)}'
            op, scoremap = vis_obj.vis_result1(data["filename"], data["filename"], data["height"], data["width"], density_pred[0])
            op_gt, scoremap_gt = vis_obj.vis_result1(data["filename"], data["filename"], data["height"], data["width"], density[0])
            op = cv2.putText(op, text, (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,255,255), 2, cv2.LINE_AA)
            w_img = wandb.Image(op)
            ip_img = data["image_np"].cpu().detach().numpy().astype(np.uint8)
            wandb.log({
                'Input': wandb.Image(ip_img),
                'Predicted density': w_img,
                'Scoremap': wandb.Image(scoremap),
                'Target': wandb.Image(op_gt),
                'Target Scoremap': wandb.Image(scoremap_gt)
            })

    
@torch.no_grad()
def val(model, valid_loader, criterion, vis_obj, config):

    print('Validating...')
    model.eval()
    os.makedirs(config.evaluator.eval_dir, exist_ok = True)
    val_loss_ = AverageMeter()
    val_rmse, val_mae = 0, 0
    weight = config.criterion[0].kwargs.weight
    loop = tqdm(valid_loader, position = 0, leave = True)

    for idx, data in enumerate(loop):

        data = to_device(data, torch.device('cuda'))
        val_loss = 0
        pred = model(data)
        val_loss = weight * criterion(pred)
        val_loss_.update(val_loss.detach(), 1)
        density = pred['density']
        density_pred = pred['density_pred']
        gt_count = torch.sum(density).item()
        pred_count = torch.sum(density_pred).item()
        dump(config.evaluator.eval_dir, pred)

        if idx % 200 == 0:
            
            text = f'GT: {int(gt_count)} & pred: {int(pred_count)}'
            op, scoremap = vis_obj.vis_result1(data["filename"], data["filename"], data["height"], data["width"], density_pred[0])
            op_gt, scoremap_gt = vis_obj.vis_result1(data["filename"], data["filename"], data["height"], data["width"], density[0])
            ip_img = data["image_np"].cpu().detach().numpy().astype(np.uint8)
            op = cv2.putText(op, text, (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,255,255), 2, cv2.LINE_AA)
            w_img = wandb.Image(op)
            wandb.log({
                'Validation Loss': val_loss_.avg,
                'Val GT count': gt_count,
                'Val Pred count': pred_count,
                'Valid Ip': wandb.Image(ip_img),
                'Valid Pred Image': w_img,
                'Valid Scoremap': wandb.Image(scoremap),
                'Valid Target': wandb.Image(op_gt),
                'Valid Target Scoremap': wandb.Image(scoremap_gt)
            })

    

    gt_count, pred_count = merge_together(config.evaluator.eval_dir)
    val_mae, val_rmse = performances(gt_count, pred_count)
    shutil.rmtree(config.evaluator.eval_dir)

    wandb.log({
        'Val Average MAE': val_mae,
        'Val Average RMSE': val_rmse
    })

    return val_mae, val_rmse


def main():
    
    os.environ['RANK'] = '0'
    torch.distributed.init_process_group(backend='nccl')
    global args, config, best_mae, best_rmse, visualizer, lr_scale_backbone
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    if (args.evaluate or args.test) and config.get("visualizer", None):
        config.visualizer.vis_dir = os.path.join(
            config.exp_path, config.visualizer.vis_dir
        )
        visualizer = build_visualizer(**config.visualizer)

    config.port = config.get("port", None)
    os.makedirs(config.save_path, exist_ok = True)
    os.makedirs(config.log_path, exist_ok = True)
    os.makedirs(config.visualizer.vis_dir, exist_ok = True)
    if not os.path.exists(config.evaluator.eval_dir):
        os.makedirs(config.evaluator.eval_dir, exist_ok = True)
    random_seed = config.get("random_seed", None)

    if random_seed:
        set_random_seed(random_seed)
    
    if config.dataset.type == 'custom_exempler':
        train_loader = build_custom_exemplar_dataloader(config, True, False)
        valid_loader = build_custom_exemplar_dataloader(config, False, False)
    
    if config.dataset.type == 'custom':
        train_loader = build_custom_dataloader(config, True, False)
        valid_loader = build_custom_dataloader(config, False, False)

    model = build_network(config.net)
    model.cuda()
    model = DDP(model, find_unused_parameters=True)

    if (args.resume):
      if args.checkpoint_path != "":
        ckpt = torch.load(args.checkpoint_path)
        model.load_state_dict(ckpt["state_dict"])
        print("Model Loaded!")
    
    model.module.backbone.eval()
    for p in model.module.backbone.parameters():
        p.requires_grad = False

    parameters = [p for n, p in model.module.named_parameters() if "backbone" not in n]
    opt = get_optimizer(parameters, config.trainer.optimizer)
    scheduler = get_scheduler(opt, config.trainer.lr_scheduler)
    criterion = _MSELoss(**config.criterion[0].kwargs)
    
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    vis_obj = Visualizer(**config.visualizer)
    vis_obj1 = Visualizer(**config.visualizer_test)
    best_mae, best_rmse = float('inf'), float('inf')
    for epoch in range(1, config.trainer.epochs + 1):

        train(model, train_loader, valid_loader, criterion, opt, scheduler, config,scaler, vis_obj, epoch)
        if epoch % 2 == 0:
            mae, rmse = val(model, valid_loader, criterion, vis_obj1, config)
            if (mae < best_mae) or (rmse < best_rmse) or (epoch % 10 == 0):
                torch.save(model.state_dict(), config.saver.save_dir + f"model_{epoch}_rmse_{int(rmse)}.pth")
                best_mae, best_rmse = mae, rmse
        

if __name__ == '__main__':

    wandb.init(project = 'Few Shot Crowd Counting')
    main()
