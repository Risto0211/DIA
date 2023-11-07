# @Author: Zecheng He
# @Date:   2019-09-01

import time
import math
import os
import numpy as np
import pickle

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
from torch.utils.data import Subset, random_split
from torch.cuda.amp import autocast, GradScaler

from net import CIFAR10CNNDecoderconv1, CIFAR10CNNDecoderReLU22, CIFAR10CNNDecoderReLU32, CIFAR10CNN, MyResNet18 \
, CIFAR10Res18DecoderLayer2, CIFAR10Res18DecoderLayer3, CIFAR10Res18DecoderMaxpool, CIFAR10VitDecoder, \
inverseNet_generic, CIFAR10Res18DecoderLayer1

from utils import evalTest, setLearningRate, getImgByClass, deprocess, load_checkpoint, TV

from ginver.Model import Conv_16x16_128, Conv_16x16_128_GRAY
from deit_models import deit_tiny_patch4_32

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio

from TinyImageNet import TinyImageNet

#####################
# Training:
# python inverse_blackbox_decoder_CIFAR.py --layer ReLU22 --iter 50 --training --decodername CIFAR10CNNDecoderReLU22
#
# Testing:
# python inverse_blackbox_decoder_CIFAR.py --testing --decodername CIFAR10CNNDecoderReLU22 --layer ReLU22
#####################


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

def train_target_epoch(device, trainloader, model, optimizer, epoch, scaler, teacher_model=None):
    model.train()
    loss_sum, acc1_num_sum = 0, 0
    num_input_sum = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (images, labels) in enumerate(trainloader):

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

        if batch_idx % 100 == 0:
            avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc1_num_sum / num_input_sum)
            print(f'[Epoch {epoch + 1}][Train][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


def validate_target(device, testloader, model, epoch, time_begin):
    model.eval()
    loss_sum, acc_num_sum = 0, 0
    num_input_sum = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):

            images, labels = images.to(device), labels.to(device)

            with autocast():
                output = model(images)
                loss = criterion(output, labels)

            acc1 = accuracy(output, labels)
            num_input_sum += images.shape[0]
            loss_sum += float(loss.item() * images.shape[0])
            acc_num_sum += float(acc1[0] * images.shape[0])

            if batch_idx % 100 == 0:
                avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
                print(f'[Epoch {epoch + 1}][Eval][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


    avg_loss, avg_acc = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'[Epoch {epoch + 1}] \t \t Top-1 {avg_acc:6.2f} \t \t Time: {total_mins:.2f}')

    return avg_acc


def train_target(target_model, trainloader, testloader):
    device = 'cuda:3'

    # if torch.cuda.device_count() > 1:
    #     target_model = nn.DataParallel(target_model)

    folder_path = './best_model'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    target_model = target_model.to(device)
    optimizer = optim.AdamW(target_model.parameters(), lr=3e-4)
    scaler = GradScaler()
    time_begin = time()
    best_val_acc = 0

    # logging.info(f'Training target model...')
    print(f'Training target model...')
    for epoch in range(50):
        train_target_epoch(device, trainloader, target_model, optimizer, epoch, scaler)

        val_acc=validate_target(device, testloader, target_model, epoch, time_begin)

        if val_acc > best_val_acc:
            print (f"save model")

            best_val_acc = val_acc

            save_model_name=f'./best_model/target_model_cnncifar10_cifar10-10k'

            torch.save(
                {
                    "model": target_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                save_model_name,
            )


def eval_target(target_model, testloader):
    device = 'cuda:3'

    # if torch.cuda.device_count() > 1:
    #     target_model = nn.DataParallel(target_model)


    target_model = target_model.to(device)

    scaler = GradScaler()
    time_begin = time()


    val_acc = validate_target(device, testloader, target_model, 0, time_begin)



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

def trainDecoderDNN(DATASET = 'CIFAR10', network = 'CIFAR10CNNDecoder', NEpochs = 50, imageWidth = 32,
            imageHeight = 32, imageSize = 32*32, NChannels = 3, NClasses = 10, layer = 'ReLU22', BatchSize = 32, learningRate = 1e-3,
            NDecreaseLR = 20, eps = 1e-3, AMSGrad = True, model_dir = "best_model/", model_name = "target_model_cnncifar10_acc74", save_decoder_dir = "best_model/",
            decodername_name = 'CIFAR10CNNDecoderReLU22', gpu = True, validation=False):

    print("DATASET: ", DATASET)

    if DATASET == 'CIFAR10':
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
        tsf = {
            'train': transforms.Compose(
            [
            transforms.Resize(32),
            transforms.ToTensor(),
            Normalize
            ]),
            'test': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ]),
            'gray': transforms.Compose(
            [
                transforms.ToTensor(),
                Normalize,
                transforms.Grayscale()
            ]
            )
        }
        if args.setting == 'same':
            trainset = torchvision.datasets.CIFAR10(root='./data', train = True,
                                            download=True, transform = tsf['train'])
        elif args.setting == 'diff':
            trainset = TinyImageNet(root='data/tiny-imagenet-200', train=True, transform=tsf['train'])
        testset = torchvision.datasets.CIFAR10(root='./data', train = False,
                                       download=True, transform = tsf['test'])
        
        # if args.same_dist:
        #     targetset, trainset, _ = random_split(trainset, [10000, 40000, len(trainset) - 50000])
        #     targetloader = torch.utils.data.DataLoader(targetset, batch_size = BatchSize, shuffle = True)

        # else:
        trainset, _ = random_split(trainset, [40000, len(trainset) - 40000])

        # trainset = torchvision.datasets.MNIST(root='./data', train = True,
        #                                 download=True, transform = None)
        # testset = torchvision.datasets.MNIST(root='./data', train = False,
        #                                download=True, transform = None)

    else:
        print("Dataset unsupported")
        exit(1)

    # with open('data/cifar-10-batches-py/subset_indices_2.pkl', 'rb') as f:
    #         subset_indices = pickle.load(f)
    # trainset = Subset(trainset, subset_indices)

    # print("len(trainset) ", len(trainset))
    # print("len(testset) ", len(testset))
    # x_train, y_train = trainset.data, trainset.targets,
    # x_test, y_test = testset.data, testset.targets,

    # print ("x_train.shape ", x_train.shape)
    # print ("x_test.shape ", x_test.shape)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BatchSize,
                                      shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False)
    trainIter = iter(trainloader)
    testIter = iter(testloader)

    ckpt, _ = load_checkpoint(model_dir + model_name)
    if args.model == 'cnn':
        net = CIFAR10CNN(3)
    elif args.model == 'res18':
        net = MyResNet18(pretrained=False, num_classes=10)
    elif args.model == 'vit':
        net = deit_tiny_patch4_32(num_classes=10)
    net.load_state_dict(ckpt)
    
    if layer == 'ReLU22':
        channels = [128, 64, 32, 3]
        size_change = ['double', 'same', 'same']
    elif layer == 'ReLU32':
        channels = [128, 64, 3]
        # size_change = ['double', 'double', 'same'] 
        size_change = ['double', 'double']  # *
    elif layer == 'layer1':
        channels = [64, 32, 3]
        size_change = ['same', 'same']
    elif layer == 'layer2':
        channels = [128, 64, 32, 3]
        size_change = ['double', 'same', 'same'] # *
    elif layer == 'layer3':
        channels = [256, 128, 64, 3]
        size_change = ['double', 'double', 'same']
    elif layer in ['3', '6', '9']:
        channels = [192, 96, 48, 3]
        size_change = ['double', 'double', 'same']

    for param in net.parameters():
        param.requires_grad = False

    # net.conv11 = nn.Conv2d(in_channels = 1,
    #         out_channels = 64,
    #         kernel_size = 3,
    #         padding = 1)
    # net.layerDict['conv11'] = net.conv11
    # net.features[0] = net.conv11
    net = net.to('cuda')

    # net = torch.load('best_model/target_model_cnncifar10_half-cifar10')
    # print(net.layerDict)

    # ce = nn.CrossEntropyLoss()
    # opt = optim.Adam(net.parameters(), lr=1e-4)
    # for i in range(50):
    #     for batch_id, (images, labels) in enumerate(trainloader):
    #         # print(images.shape, labels.shape)
    #         images = images.to('cuda')
    #         labels = labels.to('cuda')
    #         opt.zero_grad()
    #         pred = net(images)
    #         loss = ce(pred, labels)
    #         loss.backward()
    #         opt.step()
    #     print(f'epoch{i}:')
    #     accTest = evalTest(testloader, net, gpu = gpu)
    
    # torch.save(net, 'best_model/GINVERGRAY.pth')

    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    # print ("Validate the model accuracy...")

    # if validation:
    #     accTest = evalTest(testloader, net, gpu = gpu)

    # Get dims of input/output, and construct the network
    batchX, batchY = next(trainIter)
    if gpu:
        batchX = batchX.cuda()
    if args.model == 'vit':
        originalModelOutput = net.getLayerOutput(batchX, int(layer)).clone()
    else:
        originalModelOutput = net.getLayerOutput(batchX, net.layerDict[layer]).clone()

    decoderNetDict = {
        'CIFAR10CNNDecoder':{
            'conv11': CIFAR10CNNDecoderconv1,
            'ReLU22': CIFAR10CNNDecoderReLU22,
            'ReLU32': CIFAR10CNNDecoderReLU32
        },
        'GINVERDecoder':{
            'ReLU22': Conv_16x16_128
        },
        'GINVERGRAY':{
            'ReLU22': Conv_16x16_128_GRAY
        }
    }
    # decoderNetFunc = decoderNetDict[network][layer]
    # decoderNet = decoderNetFunc(1, originalModelOutput.shape[1], 530)
    # decoderNet = torch.load('best_model/CIFAR10CNNDecoderReLU22_half-cifar10.pth')
    if args.method == 'baseline':
        if args.model == 'cnn':
            if args.layer == 'ReLU22':
                decoderNet = CIFAR10CNNDecoderReLU22(3)
            elif args.layer == 'ReLU32':
                decoderNet = CIFAR10CNNDecoderReLU32(3)
        elif args.model == 'res18':
            if args.layer == 'layer2':
                decoderNet = CIFAR10Res18DecoderLayer2(3)
            elif args.layer == 'layer3':
                decoderNet = CIFAR10Res18DecoderLayer3(3)
            elif args.layer == 'layer1':
                decoderNet = CIFAR10Res18DecoderLayer1(3)
        elif args.model == 'vit':
            decoderNet = CIFAR10VitDecoder(3)
    if args.method == 'eina':
        if args.model == 'vit':
            decoderNet = inverseNet_generic(channels, size_change, vit=True)
        else:
            decoderNet = inverseNet_generic(channels, size_change)
    # print('Pretrained Loaded')
    if gpu:
        decoderNet = decoderNet.cuda()

    print (decoderNet)

    NBatch = int(len(trainset) / BatchSize)
    MSELossLayer = torch.nn.MSELoss()
    if gpu:
        MSELossLayer = MSELossLayer.cuda()

    # Find the optimal config according to the hardware
    cudnn.benchmark = True
    optimizer = optim.Adam(params = decoderNet.parameters(), lr = learningRate, eps = eps, amsgrad = AMSGrad, betas=(0.5, 0.999))


    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        for i in range(NBatch):
            try:
                batchX, batchY = next(trainIter)
            except StopIteration:
                trainIter = iter(trainloader)
                batchX, batchY = next(trainIter)

            if gpu:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            optimizer.zero_grad()
            if args.model == 'vit':
                originalModelOutput = net.getLayerOutput(batchX, int(layer)).clone()
            else:
                originalModelOutput = net.getLayerOutput(batchX, net.layerDict[layer]).clone()

            if args.add_feat_noise:
                originalModelOutput += torch.rand_like(originalModelOutput) * args.feat_noise_coef
                
            decoderNetOutput = decoderNet.forward(originalModelOutput)

            if args.model == 'vit':
                reconstruction_pred = net.getLayerOutput(decoderNetOutput, int(layer)).clone()
            else:
                reconstruction_pred = net.getLayerOutput(decoderNetOutput, net.layerDict[layer]).clone()

            assert batchX.cpu().detach().numpy().shape == decoderNetOutput.cpu().detach().numpy().shape

            # if 'cnn' in decodername_name.lower():
            featureLoss = MSELossLayer(batchX, decoderNetOutput)
                # print('Using Image Loss')
            # elif 'ginver' in decodername_name.lower():
            # featureLoss = MSELossLayer(originalModelOutput, reconstruction_pred)
            tvloss = TV(decoderNetOutput)
            featureLoss += tvloss
                # print('Using Map Loss')

            totalLoss = featureLoss
            totalLoss.backward()
            optimizer.step()

            lossTrain += totalLoss / NBatch

        valData, valLabel = next(iter(testloader))
        if gpu:
            valData = valData.cuda()
            valLabel = valLabel.cuda()
        with torch.no_grad():
            if args.model == 'vit':
                originalModelOutput = net.getLayerOutput(valData, int(layer)).clone()
            else:
                originalModelOutput = net.getLayerOutput(valData, net.layerDict[layer]).clone()
            decoderNetOutput = decoderNet.forward(originalModelOutput)
            valLoss = MSELossLayer(valData, decoderNetOutput)

        print ("Epoch ", epoch, "Train Loss: ", lossTrain.cpu().detach().numpy(), "Test Loss: ", valLoss.cpu().detach().numpy())
        # if (epoch + 1) % NDecreaseLR == 0:
        #     learningRate = learningRate / 2.0
        #     setLearningRate(optimizer, learningRate)

    # if validation:
    #     accTestEnd = evalTest(testloader, net, gpu = gpu)
    #     if accTest != accTestEnd:
    #         print ("Something wrong. Original model has been modified!")
    #         exit(1)

    if not os.path.exists(save_decoder_dir):
        os.makedirs(save_decoder_dir)
    torch.save(decoderNet, save_decoder_dir + decodername_name)
    print ("Model saved")

    newNet = torch.load(save_decoder_dir + decodername_name)
    newNet.eval()
    print ("Model restore done")


def inverse(DATASET = 'CIFAR10', imageWidth = 32, inverseClass = None, imageHeight = 32,
        imageSize = 32*32, NChannels = 3, NClasses = 10, layer = 'conv11',
        model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth", decoder_name = "CIFAR10CNNDecoderconv11.pth",
        save_img_dir = "inverted_blackbox_decoder/CIFAR10/", gpu = True, validation=False):

    print ("DATASET: ", DATASET)
    print ("inverseClass: ", inverseClass)

    assert inverseClass < NClasses

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
    psnr = PeakSignalNoiseRatio(data_range=1.0).to('cuda')

    if DATASET == 'CIFAR10':
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
        tsf = {
            'train': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ]),
            'test': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ]),
            'gray': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize,
            transforms.Grayscale()
            ]),
        }
        trainset = torchvision.datasets.CIFAR10(root='./data', train = True,
                                        download=True, transform = tsf['train'])
        testset = torchvision.datasets.CIFAR10(root='./data', train = False,
                                       download=True, transform = tsf['test'])

    else:
        print ("Dataset unsupported")
        exit(1)


    print ("len(trainset) ", len(trainset))
    print ("len(testset) ", len(testset))
    # x_train, y_train = trainset.train_data, trainset.train_labels,
    # x_test, y_test = testset.test_data, testset.test_labels,

    # print ("x_train.shape ", x_train.shape)
    # print ("x_test.shape ", x_test.shape)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1,
                                      shuffle = False)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False)
    inverseloader = torch.utils.data.DataLoader(testset, batch_size = 1,
                                      shuffle = False)
    trainIter = iter(trainloader)
    testIter = iter(testloader)
    inverseIter = iter(inverseloader)


    ckpt, _ = load_checkpoint(model_dir + model_name)
    net = MyResNet18(pretrained=False, num_classes=10).to('cuda')
    # net = CIFAR10CNN(3).to('cuda')
    net.load_state_dict(ckpt)
    # net = torch.load('best_model/target_model_res18_cifar10')

    if not gpu:
        net = net.cpu()
    net.eval()
    print ("Validate the model accuracy...")

    if validation:
        accTest = evalTest(testloader, net, gpu = gpu)

    decoderNet = torch.load(model_dir + decoder_name)
    if not gpu:
        decoderNet = decoderNet.cpu()
    decoderNet.eval()
    print (decoderNet)
    print ("Validate the alternative model...")
    batchX, batchY = next(iter(testloader))
    if gpu:
        batchX = batchX.cuda()
        batchY = batchY.cuda()

    print ("batchX.shape ", batchX.cpu().detach().numpy().shape)

    MSELossLayer = torch.nn.MSELoss()
    if gpu:
        MSELossLayer = MSELossLayer.cuda()
    originalModelOutput = net.getLayerOutput(batchX, net.layerDict[layer]).clone()
    decoderNetOutput = decoderNet.forward(originalModelOutput)

    assert batchX.cpu().detach().numpy().shape == decoderNetOutput.cpu().detach().numpy().shape

    
    print ("decoderNetOutput.shape ", decoderNetOutput.cpu().detach().numpy().shape)
    print ("MSE ", MSELossLayer(batchX, decoderNetOutput).cpu().detach().numpy())
    

    targetImg, _ = getImgByClass(inverseIter, C = inverseClass)
    print ("targetImg.size()", targetImg.size())

    # deprocessImg = deprocess(targetImg.clone())

    # if not os.path.exists(save_img_dir):
    #     os.makedirs(save_img_dir)
    # torchvision.utils.save_image(deprocessImg, save_img_dir + str(inverseClass) + '-ref.png')

    if gpu:
        targetImg = targetImg.cuda()
    targetLayer = net.layerDict[layer]
    refFeature = net.getLayerOutput(targetImg, targetLayer)

    print ("refFeature.size()", refFeature.size())

    xGen = decoderNet.forward(refFeature)
    avg_ssim = ssim(targetImg, xGen).cpu().detach().numpy()
    avg_psnr = psnr(targetImg, xGen).cpu().detach().numpy()
    print ("MSE ", MSELossLayer(targetImg, xGen).cpu().detach().numpy())
    print('SSIM: ', avg_ssim)
    print('PSNR: ', avg_psnr)

    # save the final result
    # imgGen = xGen.clone()
    # imgGen = deprocess(imgGen)
    # torchvision.utils.save_image(imgGen, save_img_dir + str(inverseClass) + '-inv.png')

    print ("Done")


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'CIFAR10')
        parser.add_argument('--network', type = str, default = 'CIFAR10CNNDecoder')
        parser.add_argument('--model', type = str, default = 'cnn')
        parser.add_argument('--setting', type = str, default = 'same')
        parser.add_argument('--method', type = str, default = 'eina')

        parser.add_argument('--training', dest='training', action='store_true')
        parser.add_argument('--testing', dest='training', action='store_false')
        parser.set_defaults(training=False)

        parser.add_argument('--iters', type = int, default = 500)
        parser.add_argument('--eps', type = float, default = 1e-3)
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--learning_rate', type = float, default = 1e-3)
        parser.add_argument('--decrease_LR', type = int, default = 20)
        parser.add_argument('--layer', type = str, default = 'ReLU32')
        parser.add_argument('--save_iter', type = int, default = 10)
        parser.add_argument('--inverseClass', type = int, default = 0)
        parser.add_argument('--decodername', type = str, default = "CIFAR10CNNDecoderReLU32")

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)

        parser.add_argument('--novalidation', dest='validation', action='store_false')
        parser.add_argument('--add_feat_noise', type=str2bool, default=False)
        parser.add_argument('--feat_noise_coef', default=0.1, type=float)
        parser.set_defaults(validation=True)
        args = parser.parse_args()

        model_dir = "best_model/"
        if args.model == 'cnn':
            model_name = "target_model_cnncifar10_acc74"
        elif args.model == 'res18':
            model_name = "target_model_res18_cifar10"
        elif args.model == 'vit':
            model_name = 'target_model_vit4_nokd_cifar10'
        decoder_name = args.decodername + '.pth'

        save_img_dir = "results/sota/relu22"

        if args.dataset == 'CIFAR10':

            imageWidth = 32
            imageHeight = 32
            imageSize = imageWidth * imageHeight
            NChannels = 3
            NClasses = 10

        else:
            print ("No Dataset Found")
            exit()

        if args.training:
            trainDecoderDNN(DATASET = args.dataset, network = args.network, NEpochs = args.iters, imageWidth = imageWidth,
            imageHeight = imageHeight, imageSize = imageSize, NChannels = NChannels, NClasses = NClasses, layer = args.layer, BatchSize = args.batch_size, learningRate = args.learning_rate,
            NDecreaseLR = args.decrease_LR, eps = args.eps, AMSGrad = True, model_dir = "best_model/", model_name = model_name, save_decoder_dir = "best_model/",
            decodername_name = decoder_name, gpu = args.gpu, validation=args.validation)

        else:
            for c in range(NClasses):
                inverse(DATASET = args.dataset, imageHeight = imageHeight, imageWidth = imageWidth, inverseClass = c,
                imageSize = imageSize, NChannels = NChannels, NClasses = NClasses, layer = args.layer,
                model_dir = model_dir, model_name = model_name, decoder_name = decoder_name,
                save_img_dir = save_img_dir, gpu = args.gpu, validation=args.validation)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
