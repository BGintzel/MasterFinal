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
from mlhelp import (
    model_utils,
    models
)

PATH = '/home/gintzel/PycharmProjects/pytorchENOpen/'


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
    if not os.path.exists(os.path.join(PATH, 'models')):
        os.makedirs(os.path.join(PATH, 'models'))
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

model_path = '/home/gintzel/PycharmProjects/pytorchENOpen/models/best__checkpoint.pth'


def load_model(type='euler', model=None, optimizer=None):
    if type == 'euler':
        path = '/home/gintzel/PycharmProjects/pytorchENOpen/models/best__checkpoint.pth'
        model = models.Combine()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.9), eps=1e-08, weight_decay=0)
    elif type == 'euler_19':
        path = '/home/gintzel/PycharmProjects/ENCircularTraining/models/old/bestcircular_18__checkpoint.pth'
        model = models.Combine()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.9), eps=1e-08, weight_decay=0)
    elif type == 'generator':
        path = '/home/gintzel/PycharmProjects/pytorchENOpen/models/generator_model_generator1__checkpoint.pth'
        model = models.GeneratorDCGAN(input_size=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.5, 0.9))
    elif type == 'generatorCircle':
        path = '/home/gintzel/PycharmProjects/pytorchENOpen/models/generator_model_generator3__checkpoint.pth'
        model = models.GeneratorSimpleCircles(input_size=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.5, 0.9))
    else:
        path = type
        model = models.Combine()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.9), eps=1e-08, weight_decay=0)
    global DEVICE
    DEVICE = set_device()
    model, _, _, _ = model_utils.load_model_simple(model, optimizer, path)
    return model