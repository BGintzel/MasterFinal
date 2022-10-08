import math
import time

import matplotlib.pyplot as plt
import matplotlib.patches as pplt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
from PIL import Image
import numpy as np
import os
import torch


# import cv2


def gen_B_in_A(r_lb, r_ub):
    A_radius = np.random.uniform(r_lb, r_ub)
    B_radius = np.random.uniform(r_lb / 2, A_radius * 0.8)
    A_center = (0.5, 0.5)
    B_center = (0.5 + np.random.uniform((B_radius - A_radius) * 0.5, (A_radius - B_radius) * 0.5),
                0.5 + np.random.uniform((B_radius - A_radius) * 0.5, (A_radius - B_radius) * 0.5))
    return A_center, B_center, A_radius, B_radius


def gen_A_in_B(r_lb, r_ub):
    B_radius = np.random.uniform(r_lb, r_ub)
    A_radius = np.random.uniform(r_lb / 2, B_radius * 0.8)
    B_center = (0.5, 0.5)
    A_center = (0.5 + np.random.uniform((A_radius - B_radius) * 0.5, (B_radius - A_radius) * 0.5),
                0.5 + np.random.uniform((A_radius - B_radius) * 0.5, (B_radius - A_radius) * 0.5))
    return A_center, B_center, A_radius, B_radius


def gen_join(r_lb, r_ub):
    B_radius = np.random.uniform(r_lb, r_ub)
    A_radius = np.random.uniform(r_lb, r_ub)
    B_center = (0.5 - np.random.uniform(B_radius * 0.2, B_radius * 0.8), 0.5)
    A_center = (0.5 + np.random.uniform(A_radius * 0.2, A_radius * 0.8), 0.5)

    return A_center, B_center, A_radius, B_radius


def gen_disjoin(r_lb, r_ub):
    B_radius = np.random.uniform(r_lb, r_ub)
    A_radius = np.random.uniform(r_lb, r_ub)
    B_center = (0.5 - np.random.uniform(B_radius * 1.1, B_radius * 1.3), 0.5)
    A_center = (0.5 + np.random.uniform(A_radius * 1.1, A_radius * 1.3), 0.5)

    return A_center, B_center, A_radius, B_radius


def gen_circle(op, filename, color_1, color_2):
    op_num = 0
    if op == '>':
        A_center, B_center, A_radius, B_radius = gen_B_in_A(0.2, 0.3)
        op_num = 1
    elif op == '<':
        A_center, B_center, A_radius, B_radius = gen_A_in_B(0.2, 0.3)
        op_num = 2
    elif op == '&':
        A_center, B_center, A_radius, B_radius = gen_join(0.2, 0.26)
        op_num = 3
    elif op == '!':
        A_center, B_center, A_radius, B_radius = gen_disjoin(0.15, 0.2)
        op_num = 4

    plot_circle(A_center, B_center, A_radius, B_radius, color_1, color_2, filename)
    return op_num
    # return A_center,B_center,A_radius,B_radius


def plot_circle(A_center, B_center, A_radius, B_radius, A_color, B_color, filename):
    circle1 = plt.Circle(A_center, A_radius, color=A_color, fill=False, linewidth=8)
    circle2 = plt.Circle(B_center, B_radius, color=B_color, fill=False, linewidth=8)

    fig, ax = plt.subplots(figsize=(8, 8))  # note we must use plt.subplots, not plt.subplot

    ax.add_artist(circle1)
    ax.add_artist(circle2)

    ax.axis('off')

    isExist = os.path.exists(os.path.dirname(os.path.abspath(filename)))

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)))
        print("The new directory is created!")
    fig.savefig(filename)
    plt.close()


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


def check_img(out):
    dist_centers = math.dist([out[0], out[1]], [out[2], out[3]])
    radiusA = out[4]
    radiusB = out[5]
    if dist_centers > radiusA + radiusB:
        op = "!"
    elif radiusA < radiusB + dist_centers and radiusB < radiusA + dist_centers and radiusA + radiusB > dist_centers:
        op = "&"
    elif radiusA > dist_centers + radiusB:
        op = ">"
    elif radiusB > dist_centers + radiusA:
        op = "<"
    return op


def make_random(tune_for_label, image, len=6):
    out = []
    myorder = [2, 3, 0, 1, 5, 4]

    for i in range(len):
        if i < 4:
            out.append(np.random.uniform(0.2, 0.8))
        else:
            out.append(np.random.uniform(0.2, get_max(out, i - 3)))

    if image == 2:
        out_check = [out[i] for i in myorder]
    else:
        out_check = out

    while check_img(out_check) == '<' and tune_for_label:
        out = []
        for i in range(len):
            if i < 4:
                out.append(np.random.uniform(0.2, 0.8))
            else:
                out.append(np.random.uniform(0.2, get_max(out, i - 3)))
        if image==2:
            out_check = [out[i] for i in myorder]
        else:
            out_check = out
    return np.asarray(out)


def make_normal(image):
    out = []
    if image == 1:
        A_center, B_center, A_radius, B_radius = gen_A_in_B(0.2, 0.3)
    else:
        A_center, B_center, A_radius, B_radius = gen_B_in_A(0.2, 0.3)

    out.append(A_center[0])
    out.append(A_center[1])
    out.append(B_center[0])
    out.append(B_center[1])
    out.append(A_radius)
    out.append(B_radius)

    return np.asarray(out)


def make_image(experiments, image, value=False, values=None, arc=None, start=-100, angle=180, colors=False, color=None,
               tune_for_label=True):

    if experiments[3]:
        out = make_random(tune_for_label, image)

    if image == 1:
        A_color = 'r'

    else:
        A_color = 'b'

    if colors:
        Middle_color = color
    else:
        Middle_color = 'g'

    if experiments[2]:
        Middle_color = (192/255, 128/255, 192/255)

    A_center = (out[0], out[1])
    Middle_center = (out[2], out[3])
    A_radius = out[4]
    Middle_radius = out[5]
    if arc:
        if angle < 360:
            if start > -100:
                start_angle1 = start
                #A_radius = A_radius*2
            else:
                start_angle1 = np.random.uniform(0, 360)
                start_angle2 = np.random.uniform(0, 360)
                # angle = 180
                out = np.append(out, start_angle1)

            circle1 = pplt.Arc(xy=A_center, width=A_radius*2, height=A_radius*2, theta1=start_angle1, theta2=start_angle1 + angle,
                               color=A_color, fill=False, linewidth=8)
            # circle_middle = pplt.Arc(xy=Middle_center, width=Middle_radius, height=Middle_radius, theta1=start_angle2,
            #                         theta2=start_angle2 + 180, color=Middle_color, fill=False, linewidth=8)
            circle_middle = plt.Circle(Middle_center, Middle_radius, color=Middle_color, fill=False, linewidth=8)
    else:
        circle1 = plt.Circle(A_center, A_radius, color=A_color, fill=False, linewidth=8)
        circle_middle = plt.Circle(Middle_center, Middle_radius, color=Middle_color, fill=False, linewidth=8)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)  # note we must use plt.subplots, not plt.subplot

    if experiments[1]:
        ax.add_artist(circle1)
    # middle
    if experiments[0]:
        ax.add_artist(circle_middle)

    # if only other or only middle
    if not experiments[0] or not experiments[1]:
        out[0] = -1000

    ax.axis('off')
    fig.savefig(f'temp.jpg')
    dirname = os.path.dirname(__file__)
    if value:
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, f'img{image}.jpg')
        fig.savefig(f'{path}', dpi=30)
    plt.close(fig)
    image_return = Image.open(f'temp.jpg')
    image_return = image_return.resize((64, 64))
    image_return = np.asarray(image_return)

    image_return = np.transpose(image_return, (2, 0, 1))

    return image_return, out


def plot_circle_simple_for_random(experiment, arc=False, colors=False, color=None):
    images = np.zeros((1, 3, 64, 64, 2), dtype='f')

    img1, out1 = make_image(experiments=experiment, image=1, arc=arc, colors=colors, color=color, tune_for_label=True)
    img2, out2 = make_image(experiments=experiment, image=2, arc=arc, colors=colors, color=color, tune_for_label=True)

    out = [out1, out2]

    images[0, :, :, :, 0] = img1 / 256
    images[0, :, :, :, 1] = img2 / 256

    # plt.imshow(np.transpose(images[0, :, :, :, 0], (1, 2, 0)))
    # plt.show()
    # plt.imshow(np.transpose(images[0, :, :, :, 1], (1, 2, 0)))
    #
    # plt.show()
    return torch.from_numpy(images), out


def plot_circle_simple_for_GUI(values, experiment=[True, True, False, True, False]):
    images = np.zeros((1, 3, 64, 64, 2), dtype='f')

    img1, out1 = make_image(experiments=experiment, image=1, values=values[:6])
    img2, out2 = make_image(experiments=experiment, image=2, values=values[6:])

    images[0, :, :, :, 0] = img1 / 256
    images[0, :, :, :, 1] = img2 / 256

    # plt.imshow(np.transpose(images[0, :, :, :, 0], (1, 2, 0)))
    # plt.imshow(np.transpose(images[0, :, :, :, 1], (1, 2, 0)))

    return torch.from_numpy(images), [out1, out2]


def plot_circle_simple_for_random_training(experiment, label):
    images = np.zeros((1, 3, 64, 64, 2), dtype='f')

    img1, out1 = make_image(experiments=experiment, image=1)
    img2, out2 = make_image(experiments=experiment, image=2)

    images[0, :, :, :, 0] = img1 / 256
    images[0, :, :, :, 1] = img2 / 256

    # plt.imshow(np.transpose(images[0, :, :, :, 0], (1, 2, 0)))
    # plt.imshow(np.transpose(images[0, :, :, :, 1], (1, 2, 0)))

    return torch.from_numpy(images), [out1, out2]

