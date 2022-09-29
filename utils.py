import os
import os.path
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image

import shutil
from torch.utils.tensorboard import SummaryWriter

import train
import model_utils
import models

dirname = os.path.dirname(__file__)
MODELPATH = os.path.join(dirname, 'models')


def parse_images_to_input(img1_path, img2_path):
    image_size = 64
    imageA = Image.open(img1_path)
    imageB = Image.open(img2_path)
    imageA = imageA.resize((image_size, image_size))
    imageB = imageB.resize((image_size, image_size))
    imageA = np.asarray(imageA)
    imageB = np.asarray(imageB)
    imageA_mean = 0
    imageB_mean = 0
    imageA_std = 256
    imageB_std = 256
    imageA = (imageA - imageA_mean) / imageA_std
    imageB = (imageB - imageB_mean) / imageB_std


    comp = torch.randn(2, 3, 64, 64, 2).to(DEVICE)
    # print(input.size())
    comp[0, :, :, :, 0] = torch.Tensor(imageA).permute(2, 0, 1)
    comp[0, :, :, :, 1] = torch.Tensor(imageB).permute(2, 0, 1)

    comp[1, :, :, :, 0] = torch.Tensor(imageA).permute(2, 0, 1)
    comp[1, :, :, :, 1] = torch.Tensor(imageB).permute(2, 0, 1)

    # img1 = comp[0, :, :, :, 0].permute(1, 2, 0).cpu().detach().numpy()
    # img2 = comp[0, :, :, :, 1].permute(1, 2, 0).cpu().detach().numpy()
    #
    # plt.imshow(img1)
    # plt.show()
    # #
    # plt.imshow(img2)
    # plt.show()

    return comp


def set_matplotlib_fontsize(size):
    SMALL_SIZE = size
    MEDIUM_SIZE = size + 3
    BIGGER_SIZE = size + 8

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def img_tensor_to_numpy(img):
    """torch.Tensor to numpy.ndarray conversion
    Input shape: (C, H, W)
    Ouput shape: (H, W, C) 
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu()
    return np.moveaxis(np.asarray(img), 0, -1)


def load_file(path):
    with open(path, "rb") as a_file:
        loaded = pickle.load(a_file)
    return loaded


def save_file(path, obj):
    path = pathlib.Path(path)
    if not os.path.exists(path.parent):
      os.makedirs(path.parent)
    # if os.path.exists(path):
    #     path = f'{path}_1'
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def set_device(printing=True):
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if printing:
        print(f"Device is : {DEVICE}")
    if not os.path.exists(MODELPATH):
        os.makedirs(MODELPATH)
    return DEVICE


def set_new_t_board(name):
    LOGS = os.path.join(os.getcwd(), "tboard_logs", name)
    if not os.path.exists(LOGS):
        os.makedirs(LOGS)

    shutil.rmtree(LOGS)
    return SummaryWriter(LOGS)


def save_image(array, name):
    im = Image.fromarray(array)
    im.save(f"{name}.jpg")


def save_txt(filename, text, text2):
    with open(f'{filename}.txt', 'w') as f:
        for i in range(len(text2)):
            f.write(f"Loss: {text[i+1]}  Output:{text2[i].detach().numpy()}\n")


def make_tensor(int_list):
    return torch.tensor(np.array(int_list, dtype='f'))


def load_model(path):

    path = path
    model = models.EN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.9), eps=1e-08, weight_decay=0)

    model, _, _, _ = model_utils.load_model_simple(model, optimizer, path)

    return model


def equal(tensor1, tensor2):
    return torch.all(torch.eq(tensor1, tensor2))