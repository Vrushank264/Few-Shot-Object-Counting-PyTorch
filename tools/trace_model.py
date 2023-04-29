import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import os
from easydict import EasyDict
import yaml
import argparse
import sys
sys.path.append('./')


from models.model_helper import build_network


def arg_parse():

    parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
    parser.add_argument(
        "-c", "--config", type=str, default="/home/oem/Vrushank/Safecount/Few-Shot-Object-Counting-PyTorch/experiments/FSC147/config.yaml", help="Path of config"
    )
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("--local_rank", default=0, help="local rank for dist")
    parser.add_argument('--resume', default=False, help='Resume training.')
    parser.add_argument('--checkpoint_path', default='/content/checkpoints/model_4.pth')

    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader = yaml.FullLoader))

    return args, config


def load_model(config, ckpt_path):

    os.environ['RANK'] = '0'
    torch.distributed.init_process_group(backend='nccl')
    model = build_network(config.net)
    model.cuda()
    model = DDP(model, find_unused_parameters = True)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def get_data(img_path, bboxes):

    data = torch.randint(0, 255, (1, 512, 512, 3), dtype = torch.uint8, device = 'cuda')
    data = cv2.imread(img_path)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = torch.from_numpy(data[None]).to('cuda')
    boxes = torch.tensor(bboxes, dtype = torch.float64).to('cuda')
    return data, boxes


class ExportModel(nn.Module):

    def __init__(self, model):

        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to('cuda').reshape(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to('cuda').reshape(1,3,1,1)
    
    def forward(self, input, boxes, fp16 = True):

        with torch.cuda.amp.autocast_mode.autocast(enabled = fp16):

            img = input.permute(0, 3, 1, 2).contigouos().half()
            boxes = boxes.contiguous().half()
            img = img.sub_(self.mean).div_(self.std)
            data = {"image": img, "boxes": boxes}
            pred = self.model(data)
            op = torch.sum(pred["density_pred"]).item()
            return int(op)
        
args, config = arg_parse()
model = load_model(config, args.ckpt_path)
exp_model = ExportModel(model)
img, boxes = get_data()

torch.cuda.synchronize()

with torch.no_grad():
    out = exp_model(img, boxes)

torch.cuda.synchronize()
print(out)


            






