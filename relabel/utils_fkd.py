import os
import numpy as np
import torch
import copy
import torch.distributed
import torchvision.models as models
import torchvision
from torchvision.transforms import functional as t_F

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *

class RandomResizedCropWithCoords(torchvision.transforms.RandomResizedCrop):
    def __init__(self, **kwargs):
        super(RandomResizedCropWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords):
        try:
            reference = (coords.any())
        except:
            reference = False
        if not reference:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            coords = (i / img.size[1],
                      j / img.size[0],
                      h / img.size[1],
                      w / img.size[0])
            coords = torch.FloatTensor(coords)
        else:
            i = coords[0].item() * img.size[1]
            j = coords[1].item() * img.size[0]
            h = coords[2].item() * img.size[1]
            w = coords[3].item() * img.size[0]
        return t_F.resized_crop(img, i, j, h, w, self.size,
                                 self.interpolation), coords


class ComposeWithCoords(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(ComposeWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomHorizontalFlipWithRes':
                img, status = t(img, status)
            else:
                img = t(img)
        return img, status, coords


class RandomHorizontalFlipWithRes(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, status):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        if status is not None:
            if status == True:
                return t_F.hflip(img), status
            else:
                return img, status
        else:
            status = False
            if torch.rand(1) < self.p:
                status = True
                return t_F.hflip(img), status
            return img, status


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def get_FKD_info(fkd_path):
    def custom_sort_key(s):
        numeric_part = int(s.split('_')[1].split('.tar')[0])
        return numeric_part
    
    max_epoch = len(os.listdir(fkd_path))
    batch_list = sorted(os.listdir(os.path.join(
        fkd_path, 'epoch_0')), key=custom_sort_key)
    batch_size = torch.load(os.path.join(
        fkd_path, 'epoch_0', batch_list[0]))[1].size()[0]
    last_batch_size = torch.load(os.path.join(
        fkd_path, 'epoch_0', batch_list[-1]))[1].size()[0]
    num_img = batch_size * (len(batch_list) - 1) + last_batch_size

    print('======= FKD: dataset info ======')
    print('path: {}'.format(fkd_path))
    print('num img: {}'.format(num_img))
    print('batch size: {}'.format(batch_size))
    print('max epoch: {}'.format(max_epoch))
    print('================================')
    return max_epoch, batch_size, num_img


class ImageFolder_FKD_MIX(torchvision.datasets.ImageFolder):
    def __init__(self, fkd_path, mode, args_epoch=None, args_bs=None, **kwargs):
        self.fkd_path = fkd_path
        self.mode = mode
        super(ImageFolder_FKD_MIX, self).__init__(**kwargs)
        self.batch_config = None  # [list(coords), list(flip_status)]
        self.batch_config_idx = 0  # index of processing image in this batch
        if self.mode == 'fkd_load':
            max_epoch, batch_size, num_img = get_FKD_info(self.fkd_path)
            if args_epoch > max_epoch:
                raise ValueError(f'`--epochs` should be no more than max epoch.')
            if args_bs != batch_size:
                raise ValueError('`--batch-size` should be same in both saving and loading phase. Please use `--gradient-accumulation-steps` to control batch size in model forward phase.')
            self.img2batch_idx_list = get_img2batch_idx_list(num_img=num_img, batch_size=batch_size, epochs=max_epoch)
            self.epoch = None

    def __getitem__(self, index):
        path, target = self.samples[index]

        if self.mode == 'fkd_save':
            coords_ = None
            flip_ = None
        elif self.mode == 'fkd_load':
            if self.batch_config == None:
                raise ValueError('config is not loaded')
            assert self.batch_config_idx <= len(self.batch_config[0])

            coords_ = self.batch_config[0][self.batch_config_idx]
            flip_ = self.batch_config[1][self.batch_config_idx]

            self.batch_config_idx += 1
        else:
            raise ValueError('mode should be fkd_save or fkd_load')

        sample = self.loader(path)

        if self.transform is not None:
            sample_new, flip_status, coords_status = self.transform(sample, coords_, flip_)
        else:
            flip_status = None
            coords_status = None

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample_new, target, flip_status, coords_status

    def load_batch_config(self, img_idx):
        """Use the `img_idx` to locate the `batch_idx`

        Args:
            img_idx: index of the first image in this batch
        """
        assert self.epoch != None
        batch_idx = self.img2batch_idx_list[self.epoch][img_idx]
        batch_config_path =  os.path.join(self.fkd_path, 'epoch_{}'.format(self.epoch), 'batch_{}.tar'.format(batch_idx))

        config = torch.load(batch_config_path,weights_only=False)
        self.batch_config_idx = 0
        self.batch_config = config[:2]
        return config[2:]

    def set_epoch(self, epoch):
        self.epoch = epoch



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, args, rand_index=None, lam=None, bbox=None):
    if args.mode == 'fkd_save':
        rand_index = torch.randperm(images.size()[0]).cuda()
        lam = np.random.beta(args.cutmix, args.cutmix)
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    elif args.mode == 'fkd_load':
        assert rand_index is not None and lam is not None and bbox is not None
        rand_index = rand_index.cuda()
        lam = lam
        bbx1, bby1, bbx2, bby2 = bbox
    else:
        raise ValueError('mode should be fkd_save or fkd_load')

    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images, args, rand_index=None, lam=None):
    if args.mode == 'fkd_save':
        rand_index = torch.randperm(images.size()[0]).cuda()
        lam = np.random.beta(args.mixup, args.mixup)
    elif args.mode == 'fkd_load':
        assert rand_index is not None and lam is not None
        rand_index = rand_index.cuda()
        lam = lam
    else:
        raise ValueError('mode should be fkd_save or fkd_load')

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None


def mix_aug(images, args, rand_index=None, lam=None, bbox=None):
    if args.mix_type == 'mixup':
        return mixup(images, args, rand_index, lam)
    elif args.mix_type == 'cutmix':
        return cutmix(images, args, rand_index, lam, bbox)
    else:
        return images, None, None, None

def get_img2batch_idx_list(num_img = 50000, batch_size = 1024, seed=42, epochs=300):
    train_dataset = torch.utils.data.TensorDataset(torch.arange(num_img))
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    img2batch_idx_list = []
    for epoch in range(epochs):
        img2batch_idx = {}
        for batch_idx, img_indices in enumerate(batch_sampler):
            img2batch_idx[img_indices[0]] = batch_idx

        img2batch_idx_list.append(img2batch_idx)
    return img2batch_idx_list
 
def load_model(args, model_name):
    orig_name = copy.deepcopy(model_name)
    prefix= model_name.split('_')[0]
    if args.dataset_name == 'imagewoof' or args.dataset_name == 'imagenet-nette':
        if prefix == 'ResNet18':
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, args.ncls)
        elif prefix == 'ResNet50':
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, args.ncls)
        elif prefix == 'Densenet121':
            model = models.densenet121(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, args.ncls)
        elif prefix == 'ShuffleNetV2':
            model = models.shufflenet_v2_x1_0(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, args.ncls)
        elif prefix == 'MobileNetV2':
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, args.ncls)
        elif prefix == 'AlexNet':
            model = models.alexnet(pretrained=False)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, args.ncls)
        else:
            raise ValueError('model_name should be one of ResNet18, ResNet50, Densenet121, ShuffleNetV2, MobileNetV2')
    elif args.dataset_name == 'imagenet1k':
        if prefix == 'ResNet18':
            model = models.resnet18(weights=None)
        elif prefix == 'ResNet50':
            model = models.resnet50(weights=None)
        elif prefix == 'ResNet101':
            model = models.resnet101(weights=None)
        elif prefix == 'Densenet121':
            model = models.densenet121(weights=None)
        elif prefix == 'Densenet169':
            model = models.densenet169(weights=None)
        elif prefix == 'Densenet201':
            model = models.densenet201(weights=None)
        elif prefix == 'Densenet161':
            model = models.densenet161(weights=None)
        elif prefix == 'MobileNetV2':
            model = models.mobilenet_v2(weights=None)
        elif prefix == 'ShuffleNetV2':
            model = models.shufflenet_v2_x0_5(weights=None)
        elif prefix == 'EfficientNet':
            model = models.efficientnet_b0(weights=None)
        elif prefix == 'AlexNet':
            model = models.alexnet(weights=None)
        else:
            raise ValueError('model_name should be one of ResNet18, ResNet50, ResNet101, Densenet121, Densenet169, Densenet201, Densenet161, MobileNetV2')
    else:
        if prefix == 'ResNet18':
            model = ResNet18(args.ncls)
        elif prefix == 'ResNet50':
            model = ResNet50(args.ncls)
        elif prefix == 'ResNet101':
            model = ResNet101(args.ncls)
        elif prefix == 'Densenet121':
            model = DenseNet121(args.ncls)
        elif prefix == 'Densenet169':
            model = DenseNet169(args.ncls)
        elif prefix == 'Densenet201':
            model = DenseNet201(args.ncls)
        elif prefix == 'Densenet161':
            model = DenseNet161(args.ncls)
        elif prefix == 'MobileNetV2':
            model = MobileNetV2(args.ncls)
        elif prefix == 'ShuffleNetV2':
            model = ShuffleNetV2(net_size=0.5, ncls=args.ncls)
        elif prefix == 'ConvNetW128':
            model = conv.ConvNet(channel=3, num_classes=args.ncls, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(args.input_size,args.input_size)) 
        else:
            raise ValueError('model_name should be one of ResNet18, ResNet50, ResNet101, Densenet121, Densenet169, Densenet201, Densenet161, MobileNetV2')
        
    model_weight_path = os.path.join(args.model_pool_dir, orig_name + '.pth')

    def pruning_classifier(model=None, classes=[]):
        try:
            model_named_parameters = [name for name, x in model.named_parameters()]
            for name, x in model.named_parameters():
                if (
                    name == model_named_parameters[-1]
                    or name == model_named_parameters[-2]
                ):
                    x.data = x[classes]
        except:
            print("ERROR in changing the number of classes.")
        return model
    
    if 'conv' in model_name:
        model = pruning_classifier(model, range(args.ncls))
        checkpoint = torch.load(
                model_weight_path, map_location="cpu",weights_only=True
            )
        model.load_state_dict(checkpoint["model"])
    else:
        state_dict = torch.load(model_weight_path, weights_only=True)
        model.load_state_dict(state_dict)
    
    return model


def count_jpg_files(directory):
    jpg_count = 0
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        # 统计所有 .jpg 和 .JPG 文件
        jpg_count += len([file for file in files if file.lower().endswith('.png')])
    
    return jpg_count