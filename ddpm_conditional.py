import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch import optim
from utils import setup_logging, get_data, save_images, CelebASet, get_reference, load_checkpoint, TV
from deit_models import deit_tiny_patch8_32, deit_tiny_patch4_32
from modules import UNet_conditional, EMA, Double_UNet_conditional, Double_UNet_conditional_tiny, UNet_conditional_CrossAttention
from torch.utils.data import DataLoader, Subset, random_split
from TinyImageNet import TinyImageNet
import logging
import pickle
#from torchvision.models import vgg16 as VGG16, VGG16_Weights
#from torchvision.models import vgg11
from torch.utils.tensorboard import SummaryWriter


from torch.cuda.amp import autocast, GradScaler
import argparse
from torchvision.transforms import ToTensor, transforms
import torchvision
from net import CIFAR10CNN, CIFAR10CNNDecoderReLU22, MyResNet18, CIFAR10Res18DecoderLayer2, CIFAR10Res18DecoderLayer3, CIFAR10Res18DecoderMaxpool, \
cut_residual
from time import time
from timm.loss import LabelSmoothingCrossEntropy
import random
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio


# logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

DATASETS = {
    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'teacher_model_path': '',
        'gray': 0,
    },
    'cifar100': {
        'num_classes': 100,
        'img_size': 32,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'teacher_model_path': '',
        'gray': 0,
    },
    'cifar10_gray': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'teacher_model_path': '',
        'gray': 1,
    },
    'fashion_mnist': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.5],
        'std': [0.5],
        'teacher_model_path': '',
    },
        'tiny_imagenet': {
        'num_classes': 165,
        'img_size': 32,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'teacher_model_path': '',
    },
}

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda", args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        if args.use_cos:
            self.beta = self.prepare_cosine_noise_schedule().to(device)
        else:
            self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def prepare_cosine_noise_schedule(self):
        max_beta = 0.999
        alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(self.noise_steps):
            t1 = i / self.noise_steps
            t2 = (i + 1) / self.noise_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas) 

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        # logging.info(f"Sampling {n} new images....")
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, model.c_in, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def sample_condition(self, model, n, labels, target_model_feature_map=None, inv_normalize=None, cfg_scale=3, args=None):
        # logging.info(f"Sampling {n} new images....")
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, args.channel, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels, target_model_feature_map)
                #if cfg_scale > 0:
                #    uncond_predicted_noise = model(x, t, None)
                #    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        #print (f"x.shape in sample:{x.shape}")

        inv_norm_image=(255 * inv_normalize(x)).type(torch.uint8)
        #denorm_images = (x.clamp(-1, 1) + 1) / 2
        #denorm_images = (denorm_images * 255).type(torch.uint8)
        return x, inv_norm_image

def obtain_same_set(args):
    img_mean, img_std = DATASETS[args.dataset_name]['mean'], DATASETS[args.dataset_name]['std']
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(img_mean, img_std)],
        std=[1 / s for s in img_std]
    )
    to_gray = [transforms.Grayscale()]

    if args.dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            *normalize,
        ])

        if DATASETS[args.dataset_name]['gray'] == 1:
            transform = transforms.Compose([*transform,
                                            *to_gray])
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)

    return trainset, testset, inv_normalize


def obtain_2ndhalf_set(args):
    img_mean, img_std = DATASETS[args.dataset_name]['mean'], DATASETS[args.dataset_name]['std']
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(img_mean, img_std)],
        std=[1 / s for s in img_std]
    )
    to_gray = [transforms.Grayscale()]

    if args.dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            *normalize,
        ])

        if DATASETS[args.dataset_name]['gray'] == 1:
            transform = transforms.Compose([*transform,
                                            *to_gray])
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        
        # with open('data/cifar-10-batches-py/subset_indices_2.pkl', 'rb') as f:
        with open('data/cifar-10-batches-py/subset_4_5.pkl', 'rb') as f:
            subset_indices = pickle.load(f)
        trainset = Subset(trainset, subset_indices)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)

    return trainset, testset, inv_normalize

def obtain_diff_set(args):
    img_mean, img_std = DATASETS[args.dataset_name]['mean'], DATASETS[args.dataset_name]['std']
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(img_mean, img_std)],
        std=[1 / s for s in img_std]
    )
    # to_gray = [transforms.Grayscale()]

    if args.dataset_name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            *normalize,
        ])

        num_classes = 10
        trainset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
    
    if args.dataset_name == 'tiny_imagenet':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            *normalize,
        ])

        num_classes = 165
        trainset = TinyImageNet(root='data/tiny-imagenet-200', train=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)

    return trainset, testset, inv_normalize


def train(args):
    print (args)

    # setup_logging(args.run_name)
    set_seed(args.seed)


    device = args.device

    if args.training_setting == 'same_set':
        trainset, testset, inv_normalize = obtain_same_set(args)
        #print (f"testset:{testset}")
    elif args.training_setting == 'same_dist':
        trainset, testset, inv_normalize = obtain_2ndhalf_set(args)
    elif args.training_setting == 'diff_set':
        trainset, testset, inv_normalize = obtain_diff_set(args)

    trainset, _ = random_split(trainset, [args.train_size, len(trainset) - args.train_size])


    if args.use_ddp == 1:
        sampler = DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=False,sampler=sampler)
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True,num_workers=2)


    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.val_batch_size, shuffle=False, num_workers=2)


    if args.target_model=='cifar10cnn':
        target_model=CIFAR10CNN(3)
    elif args.target_model=='mnistcnn':
        target_model=CIFAR10CNN(1)
    elif args.target_model=='resnet18':
        target_model=MyResNet18(pretrained=False, num_classes=10)
        if args.cut_res == 1:
            cut_residual(target_model)
    elif args.target_model=='vit':
        target_model=deit_tiny_patch4_32(num_classes=10)
    # elif args.target_model=='vgg':
    #     raise NotImplmentedError
    # elif args.target_model=='vit':
    #     raise NotImplmentedError



    #print (f"target_model: {target_model}")

    if args.retrain_target:
        train_target(target_model, trainloader, testloader, args)
    else:
        print ("load target model")
        target_model_state_dict, target_optimizer_state_dict = load_checkpoint(args.target_model_path)
        target_model.load_state_dict(target_model_state_dict)


    if args.eval_target_model:
        eval_target(target_model, testloader, args)


    target_model = target_model.to(device)
    for param in target_model.parameters():
        param.requires_grad = False

    # get the dimension of intermediate feature map to create encoder in diffusion
    train_iter_single = iter(trainloader)
    single_batch = next(train_iter_single)
    single_inputs, single_labels = single_batch
    single_inputs=single_inputs.to(device)
    #print (f"single_inputs:{single_inputs.shape}")
    if args.target_model == 'vit':
        target_dim, target_depth=target_model.getLayerDepthandDim(single_inputs, int(args.target_layer))
    else:
        target_dim, target_depth=target_model.getLayerDepthandDim(single_inputs, target_model.layerDict[args.target_layer])
    print (f"target_depth:{target_depth}")
    print (f"target_dim:{target_dim}")

    if args.attacker_model == 'unet_conditional':
        model = UNet_conditional(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
    elif args.attacker_model == 'double_unet_conditional':
        model = Double_UNet_conditional(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
    elif args.attacker_model == 'double_unet_conditional_tiny':
        model = Double_UNet_conditional_tiny(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
    elif args.attacker_model == 'unet_conditional_crossattention':
        model = UNet_conditional_CrossAttention(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)

    # print (f"UNet_conditional.condition_encoder:{model.condition_encoder}")

    if args.use_ddp == 1:
        # initialize distributed env
        dist.init_process_group(backend='nccl')
        # use DDP
        model = DDP(model)
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) 

    # model.load_state_dict(torch.load('saved_models/DDPM_conditional/cifar10-rl22-dbu-max_no-conv-norm_40k/500_ckpt.pt'))

    scaler = GradScaler()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    

    diffusion = Diffusion(img_size=args.image_size, device=device, args=args)
    logger = SummaryWriter(os.path.join("runs", args.run_name))

    l = len(trainloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    results_path = os.path.join("results", args.run_name, args.output_folder)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print("results_path directory created.")

    save_model_path = os.path.join("saved_models", args.run_name, args.output_folder)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
        print("save_model_path directory created.")

    time_begin = time()
    for epoch in range(args.epochs):
        if args.use_ddp == 1:
            sampler.set_epoch(epoch)

        model.train()
        # logging.info(f"Starting epoch {epoch}:")
        print(f"Starting epoch {epoch}:")

        loss_sum= 0
        num_input_sum = 0

        #pbar = tqdm(trainloader)
        #for i, (images, labels) in enumerate(pbar):
        for batch_idx, (images, labels) in enumerate(trainloader):
            if batch_idx > args.train_batch_used:
                break

            images = images.to(device)
            labels = labels.to(device)
            if args.condition == 'intermediate_feature_map':
                if args.target_model == 'vit':
                    target_model_feature_map = target_model.getLayerOutput(images, int(args.target_layer)).clone()
                else:
                    target_model_feature_map = target_model.getLayerOutput(images, target_model.layerDict[args.target_layer]).clone()

                #print (f"\ntarget_model_feature_map.shape:{target_model_feature_map.shape}")


            # prediction = target(images).argmax(dim=1).to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            if args.add_feat_noise:
                target_model_feature_map += torch.rand_like(target_model_feature_map) * args.feat_noise_coef

            # conditional dropout
            if np.random.random() < args.random_exploration_rate:
                labels = None
                if args.condition_method == 'add_t_emb':
                    target_model_feature_map=None
                #prediction = None



            optimizer.zero_grad()
            with autocast():
                if args.class_condition:
                    y=labels
                else:
                    y=None

                if args.condition == 'intermediate_feature_map':
                    predicted_noise = model(x_t, t, y, target_model_feature_map)
                else:
                    predicted_noise = model(x_t, t, y)


                loss = mse(noise, predicted_noise)

                if args.add_hypothetical_feature_map_loss:
                    alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                    

                    hypothetical_x_start= (1 / torch.sqrt(alpha_hat))*(x_t-(torch.sqrt(1.0/alpha_hat - 1)*predicted_noise))
                    

                    #print (f"hypothetical_x_start:{hypothetical_x_start.shape}")

                    hypothetical_model_feature_map = target_model.getLayerOutput(hypothetical_x_start, target_model.layerDict[args.target_layer]).clone()

                    #print (f"hypothetical_model_feature_map:{hypothetical_model_feature_map.shape}")

                    loss=loss+args.add_hypothetical_feature_map_loss_coef * torch.sigmoid(mse(target_model_feature_map, hypothetical_model_feature_map))

                

                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.step_ema(ema_model, model)

            num_input_sum += images.shape[0]
            loss_sum += float(loss.item() * images.shape[0])

            #pbar.set_postfix(MSE=loss.item())

            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + batch_idx)

            if batch_idx % args.print_freq == 0:
                avg_loss = (loss_sum / num_input_sum)
                print(f'[Epoch {epoch + 1}][Train][{batch_idx}] \t Loss: {avg_loss:.4e}')
        
        # print('alpha_hat: ', alpha_hat)
        # print(mse(target_model_feature_map, hypothetical_model_feature_map))

        if epoch % args.eval_and_save_freq == 0:
            # logging.info(f"testing....")
            print(f"testing....")
            model.eval()

            all_ssim_sum, all_psnr_sum, all_mse_sum=0, 0, 0
            ema_all_ssim_sum, ema_all_psnr_sum, ema_all_mse_sum = 0, 0, 0
            num_input_sum = 0

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(testloader):
                # for batch_idx, (images, labels) in enumerate(trainloader):
                    if batch_idx > args.val_batch_used:
                        break

                    images = images.to(device)
                    labels = labels.to(device)
                    if args.condition == 'intermediate_feature_map':
                        if args.target_model == 'vit':
                            target_model_feature_map = target_model.getLayerOutput(images, int(args.target_layer)).clone()
                        else:
                            target_model_feature_map = target_model.getLayerOutput(images, target_model.layerDict[args.target_layer]).clone()

                    # print(labels.shape)
                    sampled_images, inv_norm_image = diffusion.sample_condition(model, n=len(labels), labels=labels, target_model_feature_map=target_model_feature_map, inv_normalize=inv_normalize, args=args)
                    ema_sampled_images, ema_inv_norm_image = diffusion.sample_condition(ema_model, n=len(labels), labels=labels, target_model_feature_map=target_model_feature_map, inv_normalize=inv_normalize, args=args)

                    # compute metrics
                    cur_ssim=ssim(sampled_images, images).cpu().detach().numpy()
                    cur_psnr = psnr(sampled_images, images).cpu().detach().numpy()
                    cur_mse = mse(sampled_images, images).cpu().detach().numpy()

                    all_ssim_sum += float(cur_ssim * images.shape[0])
                    all_psnr_sum += float(cur_psnr * images.shape[0])
                    all_mse_sum += float(cur_mse * images.shape[0])
                    num_input_sum += images.shape[0]



                    ema_cur_ssim=ssim(ema_sampled_images, images)
                    ema_cur_psnr = psnr(ema_sampled_images, images)
                    ema_cur_mse = mse(sampled_images, images)

                    ema_all_ssim_sum += float(ema_cur_ssim * images.shape[0])
                    ema_all_psnr_sum += float(ema_cur_psnr * images.shape[0])
                    ema_all_mse_sum += float(ema_cur_mse * images.shape[0])
                    # if batch_idx % args.print_freq == 0:
                    #     print(f'[Epoch {epoch + 1}][Eval][{batch_idx}] \t ssim {all_ssim_sum / num_input_sum:.6f} \t psnr {all_psnr_sum / num_input_sum:6f}')
                    #     print(f'[Epoch {epoch + 1}][Eval][{batch_idx}] \t ema_ssim {ema_all_ssim_sum / num_input_sum:.6f} \t ema_psnr {ema_all_psnr_sum / num_input_sum:6f}')

                    # save the first batch
                    if batch_idx == 0:
                        print (inv_norm_image.shape)

                        save_images(inv_norm_image, os.path.join("results", args.run_name, args.output_folder, f"{epoch}.jpg"))
                        save_images(ema_inv_norm_image, os.path.join("results", args.run_name, args.output_folder, f"{epoch}_ema.jpg"))



            avg_ssim, avg_psnr, avg_mse = (all_ssim_sum / num_input_sum), (all_psnr_sum / num_input_sum), (all_mse_sum / num_input_sum)

            total_mins = -1 if time_begin is None else (time() - time_begin) / 60
            print(f'[Epoch {epoch + 1}] \t \t ssim {avg_ssim:.6f} \t \t psnr {avg_psnr:.6f} \t \t mse {avg_mse:.6f} Time: {total_mins:.2f}')

            ema_avg_ssim, ema_avg_psnr, ema_avg_mse = (ema_all_ssim_sum / num_input_sum), (ema_all_psnr_sum / num_input_sum), (ema_all_mse_sum / num_input_sum)
            print(f'[Epoch {epoch + 1}] \t \t ema_ssim {ema_avg_ssim:.6f} \t \t ema_psnr {ema_avg_psnr:.6f} \t \t ema_mse {ema_avg_mse:.6f} Time: {total_mins:.2f}')

            torch.save(model.state_dict(),
                       os.path.join("saved_models", args.run_name, args.output_folder, f"{epoch}_ckpt.pt"))
            torch.save(ema_model.state_dict(),
                       os.path.join("saved_models", args.run_name, args.output_folder, f"{epoch}_ema_ckpt.pt"))
            torch.save(optimizer.state_dict(),
                       os.path.join("saved_models", args.run_name, args.output_folder, f"{epoch}_optim.pt"))


def test(args):
    device = args.device
    if args.training_setting == 'same_set':
        trainset, testset, inv_normalize = obtain_same_set(args)
    elif args.training_setting == 'same_dist':
        trainset, testset, inv_normalize = obtain_2ndhalf_set(args)
    elif args.training_setting == 'diff_set':
        trainset, testset, inv_normalize = obtain_diff_set(args)


    # testset, _ = random_split(trainset, [1000, len(trainset) - 1000])
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=10, shuffle=False, num_workers=2)
    diffusion = Diffusion(img_size=args.image_size, device=device, args=args)

    mse = nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure().to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    all_ssim_sum, all_psnr_sum, all_mse_sum=0, 0, 0
    num_input_sum = 0

    train_iter_single = iter(testloader)
    single_inputs, single_labels = next(train_iter_single)
    single_inputs=single_inputs.to(device)

    if args.target_model=='cifar10cnn':
        target_model=CIFAR10CNN(3)
        target_model_state_dict, target_optimizer_state_dict = load_checkpoint('best_model/target_model_cnncifar10_acc74')
        target_model.load_state_dict(target_model_state_dict)
        target_model = target_model.to(device)
        target_dim, target_depth=target_model.getLayerDepthandDim(single_inputs, target_model.layerDict[args.target_layer])
        if args.method == 'baseline':
            if args.training_setting == 'same_set':
                attacker = 'CIFAR10CNNDecoder' + args.target_layer + '.pth'
                model_path = 'best_model/' + attacker
            elif args.training_setting == 'diff_set':
                attacker = 'TinyImageNetDecoder' + args.target_layer + '.pth'
                model_path = 'best_model/' + attacker
            decoderNet = torch.load(model_path, map_location=device).to(device)
        elif args.method == 'ginver':
            if args.training_setting == 'same_set':
                attacker = 'GinverCNNDecoder' + args.target_layer + '.pth'
                model_path = 'best_model/' + attacker
            elif args.training_setting == 'diff_set':
                attacker = 'GinverTinyImageNetCNNDecoder' + args.target_layer + '.pth'
                model_path = 'best_model/' + attacker
            decoderNet = torch.load(model_path, map_location=device).to(device)
        elif args.method == 'ours':
            model_path = 'saved_models/DDPM_conditional/' + args.output_folder + '/700_ckpt.pt'
            if args.attacker_model == 'unet_conditional':
                decoderNet = UNet_conditional(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
            elif args.attacker_model == 'double_unet_conditional':
                decoderNet = Double_UNet_conditional(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
            elif args.attacker_model == 'double_unet_conditional_tiny':
                decoderNet = Double_UNet_conditional_tiny(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
            elif args.attacker_model == 'unet_conditional_crossattention':
                decoderNet = UNet_conditional_CrossAttention(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)

            decoderNet = nn.DataParallel(decoderNet)
            decoderNet.load_state_dict(torch.load(model_path, map_location=device))
        elif args.method == 'eina':
            attacker = 'EinaCNN' + args.target_layer
            if args.training_setting == 'diff_set':
                attacker += '_diff'
            if args.add_feat_noise:
                attacker += '_noise'
            attacker += '.pth'
            model_path = 'best_model/' + attacker
            decoderNet = torch.load(model_path, map_location=device).to(device)
             

    elif args.target_model=='resnet18':
        target_model = MyResNet18(pretrained=False, num_classes=10)
        if args.cut_res == 1:
            cut_residual(target_model)
        target_model_state_dict, target_optimizer_state_dict = load_checkpoint('best_model/target_model_res18_cifar10')
        target_model.load_state_dict(target_model_state_dict)
        target_model = target_model.to(device)
        target_dim, target_depth=target_model.getLayerDepthandDim(single_inputs, target_model.layerDict[args.target_layer])
        if args.method == 'baseline':
            if args.training_setting == 'same_set':
                attacker = 'CIFAR10Res18Decoder' + args.target_layer.capitalize() + '.pth'
                model_path = 'best_model/' + attacker
            elif args.training_setting == 'diff_set':
                attacker = 'TinyImageNetRes18Decoder' + args.target_layer.capitalize() + '.pth'
                model_path = 'best_model/' + attacker
            decoderNet = torch.load(model_path, map_location=device).to(device)
        elif args.method == 'ginver':
            if args.training_setting == 'same_set':
                attacker = 'GinverRes18Decoder' + args.target_layer.capitalize() + '.pth'
                model_path = 'best_model/' + attacker
            elif args.training_setting == 'diff_set':
                attacker = 'GinverTinyImageNetRes18Decoder' + args.target_layer.capitalize() + '.pth'
                model_path = 'best_model/' + attacker
            decoderNet = torch.load(model_path, map_location=device).to(device)
        elif args.method == 'ours':
            model_path = 'saved_models/DDPM_conditional/' + args.output_folder + '/900_ckpt.pt'
            if args.attacker_model == 'unet_conditional':
                decoderNet = UNet_conditional(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
            elif args.attacker_model == 'double_unet_conditional':
                decoderNet = Double_UNet_conditional(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
            elif args.attacker_model == 'double_unet_conditional_tiny':
                decoderNet = Double_UNet_conditional_tiny(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
            elif args.attacker_model == 'unet_conditional_crossattention':
                decoderNet = UNet_conditional_CrossAttention(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
            decoderNet = nn.DataParallel(decoderNet)
            decoderNet.load_state_dict(torch.load(model_path, map_location=device))
        elif args.method == 'eina':
            attacker = 'EinaRes18' + args.target_layer.capitalize()
            if args.training_setting == 'diff_set':
                attacker += '_diff'
            attacker += '.pth'
            model_path = 'best_model/' + attacker
            decoderNet = torch.load(model_path, map_location=device).to(device)

    elif args.target_model=='vit':
        target_model = deit_tiny_patch4_32(num_classes=10)
        target_model_state_dict, target_optimizer_state_dict = load_checkpoint('best_model/target_model_vit4_nokd_cifar10')
        target_model.load_state_dict(target_model_state_dict)
        target_model = target_model.to(device)
        target_dim, target_depth=target_model.getLayerDepthandDim(single_inputs, int(args.target_layer))
        if args.method == 'baseline':
            if args.training_setting == 'same_set':
                attacker = 'CIFAR10VitDecoder' + args.target_layer + '.pth'
                model_path = 'best_model/' + attacker
            elif args.training_setting == 'diff_set':
                attacker = 'TinyImageNetVitDecoder' + args.target_layer + '.pth'
                model_path = 'best_model/' + attacker
            decoderNet = torch.load(model_path, map_location=device).to(device)
        elif args.method == 'ginver':
            if args.training_setting == 'same_set':
                attacker = 'GinverVitDecoder' + args.target_layer + '.pth'
                model_path = 'best_model/' + attacker
            elif args.training_setting == 'diff_set':
                attacker = 'GinverTinyImageNetVitDecoder' + args.target_layer + '.pth'
                model_path = 'best_model/' + attacker
            decoderNet = torch.load(model_path, map_location=device).to(device)
        elif args.method == 'ours':
            model_path = 'saved_models/DDPM_conditional/' + args.output_folder + '/600_ckpt.pt'
            if args.attacker_model == 'unet_conditional':
                decoderNet = UNet_conditional(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
            elif args.attacker_model == 'double_unet_conditional':
                decoderNet = Double_UNet_conditional(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
            elif args.attacker_model == 'double_unet_conditional_tiny':
                decoderNet = Double_UNet_conditional_tiny(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
            elif args.attacker_model == 'unet_conditional_crossattention':
                decoderNet = UNet_conditional_CrossAttention(c_in=args.channel, c_out=args.channel, num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
            decoderNet = nn.DataParallel(decoderNet)
            decoderNet.load_state_dict(torch.load(model_path, map_location=device))
        elif args.method == 'eina':
            attacker = 'EinaVit' + args.target_layer
            if args.training_setting == 'diff_set':
                attacker += '_diff'
            attacker += '.pth'
            model_path = 'best_model/' + attacker
            decoderNet = torch.load(model_path, map_location=device).to(device)
    

    # target_dim, target_depth=target_model.getLayerDepthandDim(single_inputs, target_model.layerDict['ReLU22'])
    # decoderNet = Double_UNet_conditional(num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
    # decoderNet = nn.DataParallel(decoderNet) 
    # decoderNet.load_state_dict(torch.load('saved_models/DDPM_conditional/cifar10-ad_ch-lr3e-5-rl22-dbu-max/600_ckpt.pt'))
    decoderNet.eval()
    target_model.eval()
    print('----------Start Testing----------')
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            if batch_idx > args.val_batch_used:
                break

            images = images.to(device)
            labels = labels.to(device)
            if args.condition == 'intermediate_feature_map':
                if args.target_model == 'vit':
                    target_model_feature_map = target_model.getLayerOutput(images, int(args.target_layer)).clone()
                else:
                    target_model_feature_map = target_model.getLayerOutput(images, target_model.layerDict[args.target_layer]).clone()
                
            if args.method == 'ours':
                sampled_images, inv_norm_image = diffusion.sample_condition(decoderNet, n=len(labels), labels=labels, target_model_feature_map=target_model_feature_map, inv_normalize=inv_normalize, args=args)
            else:
                sampled_images = decoderNet(target_model_feature_map)

            # compute metrics
            cur_ssim=ssim(sampled_images, images).cpu().detach().numpy()
            cur_psnr = psnr(sampled_images, images).cpu().detach().numpy()
            cur_mse = mse(sampled_images, images).cpu().detach().numpy()

            all_ssim_sum += float(cur_ssim * images.shape[0])
            all_psnr_sum += float(cur_psnr * images.shape[0])
            all_mse_sum += float(cur_mse * images.shape[0])

            num_input_sum += images.shape[0]
            if batch_idx % 100 == 0:
                print('cur_batch:', batch_idx)

        print(target_model_feature_map.shape)
        avg_ssim, avg_psnr, avg_mse = (all_ssim_sum / num_input_sum), (all_psnr_sum / num_input_sum), (all_mse_sum / num_input_sum)
        print(f'ssim {avg_ssim:.6f} \t \t psnr {avg_psnr:.6f} \t \t mse {avg_mse:.6f}')




def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_target_epoch(device, trainloader, model, optimizer, epoch, args, scaler, teacher_model=None):
    model.train()
    loss_sum, acc1_num_sum = 0, 0
    num_input_sum = 0
    criterion = LabelSmoothingCrossEntropy()
    for batch_idx, (images, labels) in enumerate(trainloader):
        #if batch_idx>args.train_batch_used:
        #    break

        images, labels = images.to(device), labels.to(device)

        with autocast():
            output = model(images)
            loss = criterion(output, labels)

        acc1 = accuracy(output, labels)
        num_input_sum += images.shape[0]
        loss_sum += float(loss.item() * images.shape[0])
        acc1_num_sum += float(acc1[0] * images.shape[0])

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc1_num_sum / num_input_sum)
            print(f'[Epoch {epoch + 1}][Train][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


def validate_target(device, testloader, model, epoch, args, time_begin):
    model.eval()
    loss_sum, acc_num_sum = 0, 0
    num_input_sum = 0
    criterion = LabelSmoothingCrossEntropy()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            #if batch_idx > args.val_batch_used:
            #    break

            images, labels = images.to(device), labels.to(device)

            with autocast():
                output = model(images)
                loss = criterion(output, labels)

            acc1 = accuracy(output, labels)
            num_input_sum += images.shape[0]
            loss_sum += float(loss.item() * images.shape[0])
            acc_num_sum += float(acc1[0] * images.shape[0])

            if batch_idx % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
                print(f'[Epoch {epoch + 1}][Eval][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


    avg_loss, avg_acc = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'[Epoch {epoch + 1}] \t \t Top-1 {avg_acc:6.2f} \t \t Time: {total_mins:.2f}')

    return avg_acc


def train_target(target_model, trainloader, testloader, args):
    device = args.device

    if torch.cuda.device_count() > 1:
        target_model = nn.DataParallel(target_model)

    folder_path = './best_model'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    target_model = target_model.to(device)
    optimizer = optim.AdamW(target_model.parameters(), lr=args.lr)
    scaler = GradScaler()
    time_begin = time()
    best_val_acc = 0

    # logging.info(f'Training target model...')
    print(f'Training target model...')
    for epoch in range(args.retrain_target_epoch):
        train_target_epoch(device, trainloader, target_model, optimizer, epoch, args, scaler)

        val_acc=validate_target(device, testloader, target_model, epoch, args, time_begin)

        if val_acc > best_val_acc:
            print (f"save model")

            best_val_acc = val_acc

            save_model_name=f'./best_model/target_model_{args.target_model}_{args.dataset_name}'

            torch.save(
                {
                    "model": target_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                save_model_name,
            )


def eval_target(target_model, testloader, args):
    device = args.device

    if torch.cuda.device_count() > 1:
        target_model = nn.DataParallel(target_model)


    target_model = target_model.to(device)

    scaler = GradScaler()
    time_begin = time()


    val_acc = validate_target(device, testloader, target_model, 0, args, time_begin)



def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # type: ignore
    os.environ["PYTHONHASHSEED"] = str(seed_value)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def init_parser(parser):
    parser.add_argument('--run_name', type=str, default='DDPM_conditional')
    parser.add_argument('--method', type=str, default='ours')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--training_setting',
                        type=str,
                        choices=['same_set', 'same_dist','diff_set'],
                        default='same_set')

    parser.add_argument('--dataset_name',
                        type=str,
                        choices=['cifar10', 'cifar100', 'fashion_mnist', 'tiny_imagenet'],
                        default='cifar10')

    parser.add_argument('--target_model',
                        type=str,
                        choices=['cifar10cnn', 'vgg', 'vit', 'mnistcnn', 'resnet18'],
                        default='cifar10cnn')
    
    parser.add_argument('--attacker_model',
                        type=str,
                        choices=['unet_conditional', 'double_unet_conditional', 'double_unet_conditional_tiny', 'unet_conditional_crossattention'],
                        default='unet_conditional')

    parser.add_argument('--target_layer', type=str, default='ReLU22') # if general: train a general model


    parser.add_argument('--output_folder',
                        type=str,
                        default='cifar10-linear')

    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=3e-6, type=float)
    parser.add_argument('--random_exploration_rate', default=0, type=float) # in the case add_channel condition, this has to be 0


    parser.add_argument('--time_dim', default=512, type=int)
    parser.add_argument('--class_condition', type=str2bool, default=True)
    parser.add_argument('--class_condition_coef', default=1, type=float) # to balance the weight of class_condition and feature_map_condition


    parser.add_argument('--condition',
                        type=str.lower,
                        choices=['intermediate_feature_map', 'logits', 'grad'],
                        default='intermediate_feature_map')
    parser.add_argument('--encoder',
                        type=str.lower,
                        choices=['small_inverse_model', 'linear', 'None'],
                        default='small_inverse_model')
    parser.add_argument('--feature_map_condition_coef', default=1,
                        type=float)  # to balance the weight of class_condition and feature_map_condition


    parser.add_argument('--condition_method',
                        type=str.lower,
                        choices=['add_t_emb', 'add_channel', 'cross_attn'],
                        default='add_channel')
    
    parser.add_argument('--add_hypothetical_feature_map_loss', type=str2bool, default=False)
    parser.add_argument('--add_hypothetical_feature_map_loss_coef', default=1, type=float)

    parser.add_argument('--use_cos', type=str2bool, default=False)
    parser.add_argument('--conv-norm', type=str2bool, default=True)
    parser.add_argument('--encoder-norm', type=str2bool, default=True)
    parser.add_argument('--attention-norm', type=str2bool, default=True)

    parser.add_argument('--add_channel_number', default=3, type=int)
    parser.add_argument('--add_t_emb_structure', choices=['linear', 'tcnn', 'attn'], default='linear', type=str)
    parser.add_argument('--binary_arg_example', type=str2bool, default=False)  # True
    parser.add_argument('--condition_encoder_size', default=1, type=int) # default=1 means 1 dilation layers


    parser.add_argument('--use_ddp', default=0, type=int)

    # target model
    parser.add_argument('--retrain_target', type=str2bool, default=False)
    parser.add_argument('--retrain_target_epoch', default=200, type=int)
    parser.add_argument('--eval_target_model', type=str2bool, default=False)
    parser.add_argument('--target_model_path',
                        type=str,
                        default='./best_model/target_model_cnncifar10_acc74')



    parser.add_argument('--eval_and_save_freq', default=100, type=int)
    parser.add_argument('--print_freq', default=100, type=int)

    # -1 for debug
    parser.add_argument('--train_batch_used', default=100000, type=int)
    # used 100 data for testing, otherwise too slow
    parser.add_argument('--val_batch_size', default=10, type=int)
    parser.add_argument('--val_batch_used', default=10, type=int)
    parser.add_argument('--test_only', default=0, type=int)
    parser.add_argument('--cut_res', default=0, type=int)
    parser.add_argument('--train_size', default=50000, type=int)
    parser.add_argument('--add_feat_noise', type=str2bool, default=False)
    parser.add_argument('--feat_noise_coef', default=0.1, type=float)
    return parser


def launch():
    set_seed()
    parser = argparse.ArgumentParser(description='DDPM_conditional')
    parser = init_parser(parser)
    args = parser.parse_args()
    if args.dataset_name == 'cifar10':
        args.num_classes = 10
        args.channel = 3

    elif args.dataset_name == 'cifar100':
        args.num_classes = 100
        args.channel = 3
    elif args.dataset_name == 'fashion_mnist':
        args.num_classes = 10
        args.channel = 1

    elif args.dataset_name == 'tiny_imagenet':
        args.num_classes = 165
        args.channel = 3
    #else:
    #    raise NotImplmentedError


    # args.dataset_path = r"./data/" + args.dataset_name
    if args.test_only == 1:
        test(args)
    else:
        train(args)
        test(args)
    # train_target(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # diffusion = Diffusion(img_size=64, device=device)
    # # dataset = CelebASet()
    # # get_reference(dataset)
    # model = UNet_conditional(num_classes=(10177//5)).to(device)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)
    # ckpt = torch.load("models/DDPM_conditional/2k_celeb-linear/ckpt.pt")
    # ema_ckpt = torch.load("models/DDPM_conditional/2k_celeb-linear/ema_ckpt.pt")
    # model.load_state_dict(ckpt)
    # ema_model.load_state_dict(ema_ckpt)
    # model.eval()
    # with torch.no_grad():
    #     labels = torch.arange(10)
    #     labels = labels.to(device)
    #     sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
    #     ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
    #     folder_path = os.path.join("results")
    #     if not os.path.exists(folder_path):
    #         os.mkdir(folder_path)
    #     save_images(sampled_images, os.path.join("results", "first10.jpg"))
    #     save_images(ema_sampled_images, os.path.join("results", "first10_ema.jpg"))

    
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

