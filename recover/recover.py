import argparse
import collections
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
from PIL import Image
from torchvision import transforms
import utils_recover as utils_re
import pandas as pd
import time
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


def get_images(args, hook_for_display, device, num_call, is_first_ipc):
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size
    targets_all = torch.LongTensor(np.arange(args.ncls))
    
    recover_model_name_list = args.teacher_model_list
    
    recover_model_list, BN_hooks, _ = utils_re.load_recover_model(recover_model_name_list, args, device)

    if is_first_ipc:
        start_index = args.start_index
    else:
        start_index = 0
    
    scaler = torch.amp.GradScaler('cuda') # Initialize GradScaler for mixed precision training

    for kk in range(start_index, args.ncls, batch_size):
        start_label = kk
        end_label = min(kk+batch_size, args.ncls)
        print(f"currently processing label from {start_label} to {end_label}")
        targets = targets_all[kk:min(kk+batch_size, args.ncls)].to(device)

        if args.initialisation_method == "Guassian":
            init_input_size = args.input_size_lis[0]
            inputs = torch.randn((targets.shape[0], 3, init_input_size, init_input_size), requires_grad=True, device=device,
                                dtype=torch.float).to(device)
            print("initialisation method: Guassian")
        else:
            inputs, orig_patch = utils_re.initialize_patch_data(kk, min(kk+batch_size, args.ncls), args, num_call)
            print(orig_patch.shape)
            print(f"initialisation method: Patches: {args.patch_diff} ")
        lim_0, lim_1 = args.jitter, args.jitter
        
        iteration_all = sum(args.optimization_budgets)

        lr_scheduler = utils_re.lr_cosine_policy(args.lr, 0, iteration_all)
        
        criterion = nn.CrossEntropyLoss().to(device)
        
        start_time = time.time()
        
        index = 0
        recover_model_counter = 0
        start_input_size = args.input_size_lis[index]
        print(f"---The start input size is {start_input_size}")
        curr_iter = 0
        for iteration_per_layer in args.optimization_budgets:
            optimizer = optim.Adam([{'params': [inputs], 'lr': args.lr}], betas=[0.5, 0.9], eps=1e-8)
            print(f"---Now input size is {start_input_size}")
            print(f"---Now synthetic data size is {inputs.shape[3]}")

            # Start to optimize the synthetic data
            for iteration in range(iteration_per_layer):
                curr_recover_model = recover_model_list[recover_model_counter]
                curr_recover_model_name = recover_model_name_list[recover_model_counter]
                BN_hook = BN_hooks[recover_model_counter]
                recover_model_counter += 1
                recover_model_counter = recover_model_counter % len(recover_model_list)
                
                lr_scheduler(optimizer, curr_iter,curr_iter)
                
                if start_input_size == args.input_size:
                    aug_function = transforms.Compose([
                        transforms.RandomResizedCrop(args.input_size),
                        transforms.RandomHorizontalFlip(),
                    ])
                    
                else:
                    aug_function = transforms.Compose([
                        transforms.RandomResizedCrop(start_input_size),
                        transforms.RandomHorizontalFlip(),
                    ])
                    
                if args.apply_data_augmentation:
                    inputs_jit = aug_function(inputs)
                else:
                    inputs_jit = inputs

                off1 = random.randint(0, lim_0)
                off2 = random.randint(0, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda'):
                    outputs_recover = curr_recover_model(inputs_jit)
                    loss_ce = criterion(outputs_recover, targets)

                    rescale = [args.first_bn_multiplier] + [1. for _ in range(len(BN_hook)-1)]
                    loss_BN = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(BN_hook)])

                    loss = args.r_bn * loss_BN+loss_ce

                scaler.scale(loss).backward()  # Scale the loss for mixed precision
                scaler.step(optimizer)  # Update model parameters
                scaler.update()  # Adjust scaling factor

                inputs.data = utils_re.clip(inputs.data, args)

                if curr_iter % save_every == 0:
                    end_time = time.time()
                    print("------------iteration {}----------".format(curr_iter))
                    print("total loss", loss.item())
                    print(f"Model: {curr_recover_model_name}, CE loss: {loss_ce.item()}, BN loss: {loss_BN.item()}")
                    print(f'time for previous iterations: {end_time-start_time}')
                    print(f'learning rate: {optimizer.param_groups[0]["lr"]}')
                    start_time = time.time()

                    if hook_for_display is not None:
                        hook_for_display(inputs, targets)
                curr_iter += 1
            index += 1
            inputs = inputs.detach()
            if curr_iter == iteration_all:
                continue
            else:
                curr_size = args.input_size_lis[index]
                if curr_size != start_input_size:
                    print("I changed")
                    inputs = F.interpolate(inputs, size=(curr_size,curr_size), mode='bilinear', align_corners=False)
                if curr_size != orig_patch.shape[2]:
                    print("I changed")
                    patch_resize = F.interpolate(orig_patch, size=(curr_size,curr_size), mode='bilinear', align_corners=False)
                else:
                    patch_resize = orig_patch
                alpha = args.alpha  
                inputs = alpha * inputs + (1 - alpha) * patch_resize
                print("Residual added")
            start_input_size = curr_size
            inputs.requires_grad = True

            
        if args.store_best_images:
            best_inputs = inputs.data.clone()
            best_inputs = utils_re.denormalize(best_inputs, args)
            save_images(args, best_inputs, targets, ipc_id)

        optimizer.state = collections.defaultdict(dict)
        torch.cuda.empty_cache()


def save_images(args, images, targets, ipc_id):
    print("save_images call")
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path_png = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store_png = dir_path_png +'/class{:03d}_id{:03d}.png'.format(class_id,ipc_id)

        if not os.path.exists(dir_path_png):
            os.makedirs(dir_path_png)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store_png)


def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def count_multiplications(start: int, target: int) -> int:
    if start <= 0 or target <= 0 or target % start != 0:
        raise ValueError("Needs to be positive integers and target should be divisible by start")
    
    count = 0
    while start < target:
        start *= 2
        count += 1
    
    return count


def parse_args():
    parser = argparse.ArgumentParser("Recover data from pre-trained model using FADRM and FADRM+")
    # Overall Configs
    parser.add_argument('--dataset-name', type=str, required=True, 
                        help='Name of the dataset to recover, currently support, CIFAR-100, Tiny-ImageNet, ImageNet-Nette, ImageWoof, ImageNet-1k')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--apply-data-augmentation', action='store_true',
                        help='whether or not to apply data augmentation')
    parser.add_argument('--start-index', type=int, default=0, 
                        help='start index of the class to recover')
    parser.add_argument('--teacher-model-list', nargs='+', type=str, default=['ResNet18'],
                        help='list of teacher models to recover')
    parser.add_argument('--input-size-lis', nargs='+', type=int)
    parser.add_argument('--optimization-budgets', nargs='+', type=int)
    parser.add_argument('--alpha',type=float)
    parser.add_argument('--pretrained-model-type', type=str, required=True, choices=['offline', 'online'],
                        help='Offline: the models are pre-trained and stored in the model pool directory\
                              Online: the pretrained models are loaded by downloading from the Pytorch Official Models')
    # Verifier Related Configs
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate the synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='MobileNetV2',
                        help="arch name to act as a verifier")
    parser.add_argument('--verifier-weight-path', type=str, default=None,
                        help="path to the verifier model weights")
    # Directory Related Configs
    parser.add_argument('--syn-data-path', type=str, required=True, 
                        help='where to store synthetic data')
    parser.add_argument('--model-pool-dir', type=str, default=None,
                        help='required when pretrained model type is offline')
    parser.add_argument('--patch-dir', type=str, default=None,
                        help='the directory where the patches are stored')
    parser.add_argument('--initialisation-dir', default=None, type=str,
                        help="the directory of the initialisation data specifically for patch initialisation,\
                              it will create a sub folder named exp-name under this directory")
    # Data Saving Related Configs
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store synthetic data')
    parser.add_argument('--store-initialised-images', action='store_true',
                        help='whether to store the initialised images when using patches initialisation')
    # Optimization Related Configs
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=4, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r-bn', type=float, default=0.05,
                        help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first-bn-multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_bn')
    parser.add_argument('--weight-temperature', default=5, type=int, help="The temperature used when calculating the weight")
    # Initialisation Related Configs
    parser.add_argument('--initialisation-method',type=str, default="Guassian", choices=["Guassian", "Patches"],
                        help='initialisation method for the synthetic data')
    parser.add_argument('--patch-diff',type=str, default="medium",
                        help="the difficulty of the patches")
    #IPC (Image Per Class) Related Configs
    parser.add_argument("--ipc-start", default=0, type=int, help="start index of IPC")
    parser.add_argument("--ipc-end", default=50, type=int, help="end index of IPC")
    args = parser.parse_args()

    # verifier model weight path is required if verifier is set to True
    if args.verifier:
        if args.pretrained_model_type == 'offline':
            assert args.verifier_weight_path is not None, "Verifier weight path is required"
    
    # set up the path for the synthetic data
    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)
        
    if args.dataset_name == 'cifar100':
        args.mean_norm = [0.5071, 0.4867, 0.4408]
        args.std_norm = [0.2675, 0.2565, 0.2761]
        args.ncls = 100
        args.jitter = 4
        args.input_size = 32


    elif args.dataset_name == 'imagenet1k':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 1000
        args.jitter = 32
        args.input_size = 224
        

    elif args.dataset_name == 'imagenet-nette':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 10
        args.jitter = 32
        args.input_size = 224

    elif args.dataset_name == 'imagewoof':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 10
        args.jitter = 32
        args.input_size = 224

    elif args.dataset_name == 'tiny_imagenet':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 200
        args.jitter = 4
        args.input_size = 64
        
    else:
        raise ValueError('dataset not supported')
    

    # check if the optimization budgets are aligned with input sizes
    if len(args.input_size_lis) != len(args.optimization_budgets):
        raise ValueError(f"optimization budgets incorrect")
        
    return args


def main_syn(args, device, ipc_id, is_first_ipc=False):
    torch.cuda.empty_cache()
    if args.verifier:
        if args.pretrained_model_type == 'offline':
            verifier_model = utils_re.load_verifier_model(args.verifier_arch, args)
        else:
            verifier_model = utils_re.load_online_model(args.verifier_arch, args)
        verifier_model = verifier_model.to(device)
        hook_for_display = lambda x,y: validate(x, y, verifier_model)
    else:
        hook_for_display = None
    get_images(args, hook_for_display, device, ipc_id, is_first_ipc)


if __name__ == '__main__':
    args = parse_args()
    print(args)

    #set up device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"---The recover process will be performed on device: {device}")

    # loop through the IPCs and generate the synthetic data
    for ipc_id in range(args.ipc_start, args.ipc_end):
        print('ipc = ', ipc_id)
        if ipc_id == args.ipc_start:
            main_syn(args, device, ipc_id, is_first_ipc=True)
        else:
            main_syn(args, device, ipc_id)