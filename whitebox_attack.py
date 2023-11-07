import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import utils
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from net import CIFAR10CNN, MyResNet18
from deit_models import deit_tiny_patch4_32
import argparse

def whitebox_attack(net, layer_name, layer, image, Niter, device, lr, eps, amsgrad, lambda_TV, lambda_l2, inv_normalize, gen=False):

    # targetlayer = net.layerDict[layer]
    ref_feature = net.getLayerOutput(image, layer)

    #TODO Change the initialization to X-R!!
    #TODO Try different initialization
    image_hat = torch.rand(image.size(), requires_grad=True, device=device)
    # r = torch.rand(image.size(), requires_grad = True, device=device)
    # image_hat = image - r

    optimizer = optim.Adam(params=[image_hat], lr=lr, eps=eps, amsgrad=amsgrad)
    for i in range(Niter):
        optimizer.zero_grad()
        cur_feature = net.getLayerOutput(image_hat, layer)

        feature_loss = ((cur_feature-ref_feature)**2).mean()
        TVloss = utils.TV(image_hat)
        normLoss = utils.l2loss(image_hat)
        loss = feature_loss +lambda_TV * TVloss + lambda_l2 * normLoss

        loss.backward(retain_graph=True)
        optimizer.step()
        # if i % 500 ==0:
        #     print('Batch update iter: ', i)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    mse = nn.MSELoss()

    # image deprocess: necessary for ssim!!!!
    # de_image_hat = utils.deprocess(image_hat)
    # de_image = utils.deprocess(image)
    de_image_hat = image_hat
    de_image = image

    SSIM = ssim(de_image_hat, de_image).cpu().detach().numpy()
    PSNR = psnr(de_image_hat, de_image).cpu().detach().numpy()
    MSE = mse(de_image_hat, de_image).cpu().detach().numpy()

    # # tmp use for batch>1
    # SSIM = 0
    # PSNR = 0

    # # # show attack result on a single image
    # print('layer: {}, Iter: {}, loss: {}, psnr: {}, ssim:{}'.format(layer_name, i, loss.detach(), PSNR, SSIM))
    if gen:
        image_hat_dep = utils.deprocess(image_hat)
        file = './whitebox.png'
        torchvision.utils.save_image(image_hat_dep, file)
        # torchvision.utils.save_image(image, './whitebox_check/img.png')
    return PSNR, SSIM, MSE


def init_parser(parser):
    parser.add_argument('--name', type=str, default='cnn')
    parser.add_argument('--layer', type=str, default='ReLU22')
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = init_parser(parser)
    args = parser.parse_args()
    if args.name == 'cnn':
        net = CIFAR10CNN(3)
        ckpt, _ = utils.load_checkpoint('best_model/target_model_cnncifar10_acc74')
    elif args.name == 'res18':
        net = MyResNet18(num_classes=10)
        ckpt, _ = utils.load_checkpoint('best_model/target_model_res18_cifar10')
    elif args.name == 'vit':
        net = deit_tiny_patch4_32(num_classes=10)
        ckpt, _ = utils.load_checkpoint('best_model/target_model_vit4_nokd_cifar10')

    net.load_state_dict(ckpt)
    net.eval()
    net.to('cuda:0')
    transform = transforms.Compose([transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    device = 'cuda:0'
    lr = 0.01
    eps = 1e-8
    amsgrad = False
    Niter = 5000
    lambda_l2 = 0.0
    layer_name = args.layer
    if layer_name in ['ReLU22', 'layer1', 'layer2', '3', '6', '9']:
        lambda_TV = 1e1
    elif layer_name in ['ReLU32', 'layer3']:
        lambda_TV = 1e3
    elif layer_name == 'maxpool':
        lambda_TV = 0.0
    
    if args.name == 'vit':
        layer = int(layer_name)
    else:
        layer = net.layerDict[layer_name]
    ssim_sum = 0
    psnr_sum = 0
    mse_sum = 0

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(img_mean, img_std)],
        std=[1 / s for s in img_std]
    )
    gen = False

    for i, data in enumerate(trainloader):
        image, label = data
        image = image.to(device)
        label = label.to(device)
        if i > 99:
            gen = True
        cur_psnr, cur_ssim, cur_mse = whitebox_attack(net, layer_name, layer, image, Niter, device, lr, eps, amsgrad, lambda_TV, lambda_l2, inv_normalize, gen=gen)
        ssim_sum += cur_ssim
        psnr_sum += cur_psnr
        mse_sum += cur_mse
        if i > 99:
            print('layer: {}, avg ssim: {}, avg psnr: {}, avg mse: {}'.format(layer_name, ssim_sum/(i+1), psnr_sum/(i+1), mse_sum/(i+1)))
            break
        