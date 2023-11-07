import os
import torch
import torchvision
import numpy as np
import pickle
import torch.nn as nn
from PIL import Image, ImageDraw
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision.datasets import CIFAR10, CelebA
from torchvision import transforms
from get_celeba import load_data
from pathlib import Path

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    w, h = im.size
    label_width = 30
    new_img = Image.new('RGB', (w + label_width, h))
    new_img.paste(im, (label_width, 0))
    draw = ImageDraw.Draw(new_img)
    labels = ["Label{}".format(i) for i in range(images.size(0))]  # 根据你的需求生成标签
    y_step = h // len(images)
    for i, label in enumerate(labels):
        y_position = i * y_step
        draw.text((5, y_position), label, fill=(255, 255, 255))
    new_img.save(path)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    


def get_data(args, train=False):
    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
    #     torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transform = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transformC = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    if args.dataset_name == 'cifar10':
        length = 10
        dataset = CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform)
        class_datasets = [[] for _ in range(length)]         # CIFAR10有10个类别
    else:
        length = 400
        dataset = CelebASet(transform=transformC)
        class_datasets = [[] for _ in range(length)]

    # 按类别分组数据
    print("Seperating Classes")
    pbar = tqdm(dataset)
    for image, label in pbar:
        if label < 401:
            class_datasets[label - 1].append((image, label))
    # 把0-4类作为训练集，5-9类作为测试集
    print("Concatenating Datasets...")
    train_dataset = ConcatDataset([MyDataset(class_datasets[i]) for i in range(length//2)])
    torch.save(train_dataset, './data/CelebA/train_set200.pth')
    test_dataset = ConcatDataset([MyDataset(class_datasets[i]) for i in range(length//2, length)])
    torch.save(test_dataset, './data/CelebA/test_set200.pth')
    # train_dataset = torch.load('data/CelebA/train_set200.pth')
    # test_dataset = torch.load('data/CelebA/test_set200.pth')
    if train:
        dataset = train_dataset
    else:
        dataset = ConcatDataset([train_dataset, test_dataset])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


class CelebASet(Dataset):
    def __init__(self, list_path='./data/CelebA/data_list.pkl', img_folder='./data/CelebA/celeba/img_align_celeba', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64), antialias=True),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])) -> None:
        super().__init__()
        self.path = list_path
        self.folder = img_folder
        self.list = pickle.load(open("./data/CelebA/data_list.pkl", "rb"))
        self.transform = transform
        # self.length = len(self.list) // 2
        self.length = len(self.list) // 5
        self.data = []
        if train:
            for i in range(self.length // 2):
                for j in range(len(self.list[i])):
                    img_path = self.list[i][j]
                    self.data.append((img_path, i))
        
        else:
            for i in range(self.length):
                for j in range(len(self.list[i])):
                    img_path = self.list[i][j]
                    self.data.append((img_path, i))

        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        (img_path, label) = self.data[index]
        
        img = np.array(self.transform(Image.open(img_path)))
        label = label
        return (img, label)
    

def get_reference(dataset):
    ref = []
    lab = 1018
    for img, label in dataset:
        if label == lab:
            ref.append(img)
            lab += 1
        if lab == 1028:
            break
    
    images = (torch.tensor(np.array(ref)).clamp(-1, 1) + 1) * 255 / 2
    images = images.type(torch.uint8)
    save_images(images=images, path='results/DDPM_conditional/2k_celeb-linear_target-pred/reference.jpg')


def load_checkpoint(checkpoint_pthpath):
    """Given a path to saved checkpoint, load corresponding state dicts
    of model and optimizer from it. This method checks if the current
    commit SHA of codebase matches the commit SHA recorded when this
    checkpoint was saved by checkpoint manager.

    Parameters
    ----------
    checkpoint_pthpath: str or pathlib.Path
        Path to saved checkpoint (as created by ``CheckpointManager``).

    Returns
    -------
    nn.Module, optim.Optimizer
        Model and optimizer state dicts loaded from checkpoint.

    Raises
    ------
    UserWarning
        If commit SHA do not match, or if the directory doesn't have
        the recorded commit SHA.
    """

    if isinstance(checkpoint_pthpath, str):
        checkpoint_pthpath = Path(checkpoint_pthpath)
    checkpoint_dirpath = checkpoint_pthpath.resolve().parent
    checkpoint_commit_sha = list(checkpoint_dirpath.glob(".commit-*"))
    """
    if len(checkpoint_commit_sha) == 0:
        warnings.warn(
            "Commit SHA was not recorded while saving checkpoints."
        )
    else:
        # verify commit sha, raise warning if it doesn't match
        commit_sha_subprocess = Popen(
            ["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE
        )
        commit_sha, _ = commit_sha_subprocess.communicate()
        commit_sha = commit_sha.decode("utf-8").strip().replace("\n", "")

        # remove ".commit-"
        checkpoint_commit_sha = checkpoint_commit_sha[0].name[8:]

        if commit_sha != checkpoint_commit_sha:
            warnings.warn(
                f"Current commit ({commit_sha}) and the commit "
                f"({checkpoint_commit_sha}) at which checkpoint was saved,"
                " are different. This might affect reproducibility."
            )
    """
    # load encoder, decoder, optimizer state_dicts
    components = torch.load(checkpoint_pthpath)
    return components["model"], components["optimizer"]


def accuracy(predictions, labels):

    if not (predictions.shape == labels.shape):
        print ("predictions.shape ", predictions.shape, "labels.shape ", labels.shape)
        raise AssertionError

    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu

def pseudoInverse(W):
    return np.linalg.pinv(W)


def getImgByClass(Itr, C = None):

    if C == None:
        return next(Itr)
    
    imgs = []
    while (True):
        img, label = next(Itr)
        if label == C:
            imgs.append(img.squeeze(0))
            if len(imgs) > 99:
                break

    imgs = torch.stack(imgs)
    print('imgs.shape', imgs.shape)
    return imgs, label


def clip(data):
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0
    return data

def preprocess(data):

    size = data.shape
    NChannels = size[-1]
    assert NChannels == 1 or NChannels == 3
    if NChannels == 1:
        mu = 0.5
        sigma = 0.5
    elif NChannels == 3:
        mu = [0.485, 0.456, 0.406]
        sigma = [0.229, 0.224, 0.225]
    data = (data - mu) / sigma

    assert data.shape == size
    return data


def deprocess(data):

    assert len(data.size()) == 4

    BatchSize = data.size()[0]
    assert BatchSize == 1

    NChannels = data.size()[1]
    if NChannels == 1:
        mu = torch.tensor([0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5], dtype=torch.float32)
    elif NChannels == 3:
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    else:
        print ("Unsupported image in deprocess()")
        exit(1)

    Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
    return clip(Unnormalize(data[0,:,:,:]).unsqueeze(0))


def evalTest(testloader, net, gpu = True):
    testIter = iter(testloader)
    acc = 0.0
    NBatch = 0
    for i, data in enumerate(testIter, 0):
        NBatch += 1
        batchX, batchY = data
        if gpu:
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        logits = net.forward(batchX)

        if gpu:
            pred = np.argmax(logits.cpu().detach().numpy(), axis = 1)
            groundTruth = batchY.cpu().detach().numpy()
        else:
            pred = np.argmax(logits.detach().numpy(), axis = 1)
            groundTruth = batchY.detach().numpy()
        acc += np.mean(pred == groundTruth)
    accTest = acc / NBatch
    print ("Test accuracy: ", accTest) #, "NBatch: ", NBatch, "pred == groundTruth.shape", (pred == groundTruth).shape
    return accTest




def setLearningRate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow(x[:,:,1:,:]-x[:,:,:h_x-1,:], 2).sum()
    w_tv = torch.pow(x[:,:,:,1:]-x[:,:,:,:w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def l2loss(x):
    return (x**2).mean()

def l1loss(x):
    return (torch.abs(x)).mean()

def getL1Stat(net, x):
    for layer in net.layerDict:
        targetLayer = net.layerDict[layer]
        layerOutput = net.getLayerOutput(x, targetLayer)
        print ("Layer " + layer + ' l1 loss:', l1loss(layerOutput).cpu().detach().numpy())


def getModule(net, blob):
    modules = blob.split('.')
#    print "Target layer: ", modules
#    if len(modules) == 1:
#        return net._modules.get(blob)
#    else:

    curr_module = net
    print (curr_module)
    for m in modules:
        curr_module = curr_module._modules.get(m)
    return curr_module

def getLayerOutputHook(module, input, output):
    if not hasattr(module, 'activations'):
        module.activations = []
    module.activations.append(output)

def getHookActs(model, module, input):
    if hasattr(module, 'activations'):
        del module.activations[:]
    _ = model.forward(input)
    assert(len(module.activations) == 1)
    return module.activations[0]

def saveImage(img, filepath):
    torchvision.utils.save_image(img, filepath)


def apply_noise(input, noise_type, noise_level, mean=0.0, gpu=True, args=None):

    if noise_type == 'Gaussian':
        noise = torch.randn(input.size()) * noise_level + mean
        noise = noise.cuda() if gpu else noise
        output = input + noise

    elif noise_type == 'Laplace':
        noise = np.random.laplace(
            loc= mean,
            scale = noise_level,
            size = input.size()
        )
        noise = torch.tensor(noise, dtype = torch.float)
        noise = noise.cuda() if gpu else noise
        output = input + noise

    elif noise_type == 'dropout':
        mask = np.random.choice([0.0, 1.0], size=input.size(), replace=True, p=[noise_level, 1-noise_level])
        mask = torch.tensor(mask, dtype = torch.float)
        mask = mask.cuda() if gpu else mask
        output = input * mask

    elif noise_type == 'dropout-non-zero':
        input_list = input.detach().cpu().numpy().reshape([-1])
        output = input_list.copy()

        for i in range(len(input_list)):
            if input_list[i] > 0:
                if np.random.rand() < noise_level:
                    output[i] = -1.0
            else:
                output[i] = -np.random.rand() * 10.0
        output = torch.tensor(np.array(output).reshape(input.size()), dtype = torch.float)
        output = output.cuda() if gpu else output

    elif noise_type == 'redistribute':
        input_list = input.detach().cpu().numpy().reshape([-1])
        idx = np.argsort(input_list)
        map = np.linspace(start=0.0, stop=1.0, num=len(input_list))

        output = [0]*len(input_list)
        for i in range(len(idx)):
            if input_list[idx[i]] != 0 and np.random.rand() > noise_level:
                output[idx[i]] = 1.0
        output = torch.tensor(np.array(output).reshape(input.size()), dtype = torch.float)
        output = output.cuda() if gpu else output

        #print "input", input
        #print "output", output

    elif noise_type == 'impulse':
        noise = np.random.choice([0.0, 1.0], size=input.size(), replace=True, p=[1-noise_level, noise_level])
        noise = torch.tensor(noise, dtype = torch.float)
        noise = noise.cuda() if gpu else noise
        output = input + noise

    elif noise_type == 'noise_gen' or 'noise_gen_opt':
        noise_dir = 'noise' + ('_opt' if noise_type == 'noise_gen_opt' else "") + '/' + args.dataset + '/'
        noise_file_name = args.noise_sourceLayer + '-' + args.noise_targetLayer + '-' + str(round(noise_level, 2))

        noise = np.load(noise_dir + noise_file_name + '.npy')
        noise = torch.tensor(noise, dtype = torch.float)

        batch_size = input.size()[0]
        noise = torch.cat(batch_size * [noise])
        noise = noise.cuda() if gpu else noise
        output = input + noise

    else:
        print ("Unsupported Noise Type: ", noise_type)
        exit(1)

    return output

def evalTestSplitModel(testloader, netEdge, netCloud, layer, gpu, noise_type = None, noise_level = 0.0, args=None):
    testIter = iter(testloader)
    acc = 0.0
    NBatch = 0
    for i, data in enumerate(testIter, 0):
        batchX, batchY = data
        if gpu:
            batchX = batchX.cuda()
            batchY = batchY.cuda()

        if hasattr(args, 'add_noise_to_input') and args.add_noise_to_input:
            batchX = apply_noise(batchX, noise_type, noise_level, gpu=gpu, args=args)

        try:
            edgeOutput = netEdge.getLayerOutput(batchX, netEdge.layerDict[layer]).clone()
        except Exception as e:
            #print "Except in evalTestSplitModel getLayerOutput, this is a Edge-only model"
            #print str(e)
            edgeOutput = netEdge.forward(batchX).clone()

        if noise_type != None and not (hasattr(args, 'add_noise_to_input') and args.add_noise_to_input):
            edgeOutput = apply_noise(edgeOutput, noise_type, noise_level, gpu=gpu, args=args)

        #cloudOuput = net.forward(batchX)
        logits = netCloud.forward_from(edgeOutput, layer)

        #softmax = nn.Softmax().cuda()
        #prob = softmax(logits)
        #print prob[:100,:].max(dim=1)

        if gpu:
            pred = np.argmax(logits.cpu().detach().numpy(), axis = 1)
            groundTruth = batchY.cpu().detach().numpy()
        else:
            pred = np.argmax(logits.detach().numpy(), axis = 1)
            groundTruth = batchY.detach().numpy()
        acc += np.mean(pred == groundTruth)
        NBatch += 1

    accTest = acc / NBatch
    #print "Test accuracy: ", accTest #, "NBatch: ", NBatch, "pred == groundTruth.shape", (pred == groundTruth).shape
    return accTest

def get_PSNR(refimg, invimg, peak = 1.0):
    psnr = 10*np.log10(peak**2 / np.mean((refimg - invimg)**2))
    return psnr

def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])

    h_tv = torch.pow(x[:, :, 1:, :]-x[:, :, :h_x-1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:]-x[:, :, :, :w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


if __name__ == '__main__':
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CelebASet(train=False)
    # dataset = torch.load('data/CelebA/test_set200.pth')
    get_reference(dataset)
    # print(len(dataset))

