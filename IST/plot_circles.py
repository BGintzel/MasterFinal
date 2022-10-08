import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


def get_max(out, i):
    if i == 1:
        x = out[0]
        y = out[1]
    elif i == 2:
        x = out[2]
        y = out[3]
    max_x = min(x, 1 - x)
    max_y = min(y, 1 - y)
    return min(max_x, max_y)


def make_random(len=6):
    out = []
    for i in range(len):
        if i < 4:
            out.append(np.random.uniform(0.2, 0.8))
        else:
            out.append(np.random.uniform(0.1, get_max(out, i - 3)))
    return np.asarray(out)


def make_image(middle, other, image):
    out = make_random()

    if image == 1:
        A_color = 'r'
    else:
        A_color = 'b'

    Middle_color = 'g'

    A_center = (out[0], out[1])
    Middle_center = (out[2], out[3])
    A_radius = out[4]
    Middle_radius = out[5]

    circle1 = plt.Circle(A_center, A_radius, color=A_color, fill=False, linewidth=8)
    circle_middle = plt.Circle(Middle_center, Middle_radius, color=Middle_color, fill=False, linewidth=8)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    if other:
        ax.add_artist(circle1)
    # middle
    if middle:
        ax.add_artist(circle_middle)

    # if only other or only middle for label [0,0,0,0]
    if not other or not middle:
        out[0] = -100

    ax.axis('off')
    fig.savefig(f'temp.jpg')


    plt.close(fig)
    image_return = Image.open(f'temp.jpg')
    image_return = image_return.resize((64, 64))
    image_return = np.asarray(image_return)

    image_return = np.transpose(image_return, (2, 0, 1))

    return image_return, out


def plot_circle_for_random_training(middle, other):
    images = np.zeros((1, 3, 64, 64, 2), dtype='f')

    img1, out1 = make_image(middle=middle, other=other, image=1)
    img2, out2 = make_image(middle=middle, other=other, image=2)

    images[0, :, :, :, 0] = img1 / 256
    images[0, :, :, :, 1] = img2 / 256

    return torch.from_numpy(images), [out1, out2]
