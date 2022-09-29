
import numpy as np
import torch
import torch.nn as nn
import math
import cv2
import matplotlib.pyplot as plt

from euler_circle_generator.euler_circle_gen_duo import inference
from euler_circle_generator.euler_circle_lib import make_image
from euler_circle_generator.euler_circle_lib import plot_circle_simple_for_random_training
from euler_circle_generator.euler_circle_lib import plot_circle_simple_for_random

import utils


def train_random_circles(model_path, experiment, iterations=10000, label=np.array([0.0, 1.0, 0., 0.], dtype='f'), arc=False, colors=False, color=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label = torch.tensor(label)
    BCE = nn.BCELoss()

    # load EN
    en = utils.load_model(model_path)
    en.eval()

    best_images = []
    best_loss = [100000000000]
    best_output = []
    best_out = []
    print(iterations)
    while len(best_out)==0:
        for i in range(iterations):
            if i % 100 == 0:
                print(f"{i * 100 / iterations} %  {best_loss[-1]}")
            if best_loss[-1] < 0.0001:
                print(f'Break with loss:{best_loss[-1]}    At Iteration: {i}')
                break

            # create circlepair
            images, out = plot_circle_simple_for_random(experiment, arc=arc, colors=colors, color=color)

            output_label = create_label(out.copy())

            # print(label)
            # print(output_label)
            # print(torch.all(torch.eq(label, output_label)))
            # print()

            images.to(device)

            output = en(images)[0].cpu()

            loss = BCE(output, label)
            if loss < best_loss[-1]:
                print(len(best_out))
                best_out.append(out)
                best_images.append(images)
                best_loss.append(loss)
                best_output.append(output)

    return best_images, best_loss, best_output, best_out


'''
Relational statement format:
Letter:A-Z for different entity
Relationship for Venn-2: 
1. A contains B ['A','>','B']
2. A contained in B ['A','<','B']
3. A intersects B ['A','&','B']
4. A does not intersect B ['A','!','B']
'''


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


def create_label(out):
    if out[0][0] == -100:
        label = [0, 0, 0, 0]
    else:
        op1 = check_img(out[0])
        myorder = [2, 3, 0, 1, 5, 4]
        out[1] = [out[1][i] for i in myorder]
        op2 = check_img(out[1])
        op3, label = inference(op1, op2)
    return utils.make_tensor(label)


def get_random_circles_for_training(en, experiment_input=[True, True, False, True, False], number_of_samples=1000,
                                    show=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0],
              [1, 0, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]]

    BCE = nn.BCELoss()
    imgs = []
    lbls = []

    en.eval()

    for i, label in enumerate(labels):
        label = utils.make_tensor(label)
        threshold = 2
        a = 0
        while len(imgs) < int(number_of_samples / len(labels)) * (i + 1):
            a += 1
            if i == 0 and len(imgs) < int(number_of_samples / (len(labels) * 2)):
                experiment = [True, False, False, True, False]
            elif i == 0 and len(imgs) >= int(number_of_samples / (len(labels) * 2)):
                experiment = [True, False, False, True, False]
            else:
                experiment = experiment_input
            # create circlepair
            images, out = plot_circle_simple_for_random_training(experiment, label)
            output_label = create_label(out)
            images.to(device)
            output = en(images)[0].cpu()

            loss = BCE(output, output_label)
            if a % 100 == 0:
                print(loss.item())
                threshold = threshold / 2
                print(threshold)
            if threshold < 0.01:
                imgs.append(images)
                lbls.append(output_label)

            if loss > threshold:
                threshold = 2
                imgs.append(images)
                lbls.append(output_label)
                if len(imgs) % 10 == 0:
                    print(len(imgs))
                if show:
                    img1 = images[0, :, :, :, 0].permute(1, 2, 0).cpu().detach().numpy()
                    img2 = images[0, :, :, :, 1].permute(1, 2, 0).cpu().detach().numpy()

                    img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_32F).astype(
                        np.uint8)
                    img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_32F).astype(
                        np.uint8)
                    plt.imshow(img1)
                    plt.show()

                    plt.imshow(img2)
                    plt.show()
                    print(output)
                    print(output_label)
                    print(loss)
                    break
        print(f'Label {i} dataset completed! Progress:{(i + 1) / (len(labels)) * 100}%')
    return imgs, lbls


def get_random_circles_for_testing(angle, number_of_samples=1000):

    labels = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0],
              [1, 0, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]]

    imgs = []
    lbls = []

    for i, label in enumerate(labels):

        while len(imgs) < int(number_of_samples / len(labels)) * (i + 1):

            # create circle pair
            img1, out1 = make_image(experiments=[True, True, False, True, False], image=1, arc=True, angle=angle)
            img2, out2 = make_image(experiments=[True, True, False, True, False], image=2, arc=True, angle=angle)

            out = [out1, out2]

            images = np.zeros((1, 3, 64, 64, 2), dtype='f')

            images[0, :, :, :, 0] = img1 / 256
            images[0, :, :, :, 1] = img2 / 256

            output_label = create_label(out)

            imgs.append(images)
            lbls.append(output_label)
            if len(imgs) % 10 == 0:
                print(len(imgs))

        print(f'Label {i} dataset completed! Progress:{(i + 1) / (len(labels)) * 100}%')
    return imgs, lbls

def get_random_circles_for_testing_color(color, number_of_samples=1000):

    labels = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0],
              [1, 0, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]]

    imgs = []
    lbls = []

    for i, label in enumerate(labels):

        while len(imgs) < int(number_of_samples / len(labels)) * (i + 1):

            # create circle pair
            img1, out1 = make_image(experiments=[True, True, False, True, False], image=1, colors=True, color=color)
            img2, out2 = make_image(experiments=[True, True, False, True, False], image=2, colors=True, color=color)

            out = [out1, out2]

            images = np.zeros((1, 3, 64, 64, 2), dtype='f')

            images[0, :, :, :, 0] = img1 / 256
            images[0, :, :, :, 1] = img2 / 256

            output_label = create_label(out)

            imgs.append(images)
            lbls.append(output_label)
            if len(imgs) % 10 == 0:
                print(len(imgs))

        print(f'Label {i} dataset completed! Progress:{(i + 1) / (len(labels)) * 100}%')
    return imgs, lbls

# def get_random_circles_for_testing(angle, number_of_samples=1000):
#
#     labels = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0],
#               [1, 0, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]]
#
#
#     imgs = []
#     lbls = []
#
#     for i, label in enumerate(labels):
#         label = utils.make_tensor(label)
#
#         while len(imgs) < int(number_of_samples / len(labels)) * (i + 1):
#             middle = True
#             other = True
#
#             if i == 0 and len(imgs) < int(number_of_samples / (len(labels) * 2)):
#                 middle = False
#             elif i == 0 and len(imgs) >= int(number_of_samples / (len(labels) * 2)):
#                 other = False
#
#             # create circle pair
#             img1, out1 = make_image(experiments=[True, True, False, True, False], image=1, arc=True, angle=angle)
#             img2, out2 = make_image(experiments=[True, True, False, True, False], image=2, arc=True, angle=angle)
#
#             out = [out1, out2]
#
#             images = np.zeros((1, 3, 64, 64, 2), dtype='f')
#
#             images[0, :, :, :, 0] = img1 / 256
#             images[0, :, :, :, 1] = img2 / 256
#
#             if angle == 360:
#                 output_label = utils.make_tensor([0, 0, 0, 0])
#             else:
#                 output_label = create_label(out)
#
#             imgs.append(images)
#             lbls.append(output_label)
#             if len(imgs) % 10 == 0:
#                 print(len(imgs))
#
#         print(f'Label {i} dataset completed! Progress:{(i + 1) / (len(labels)) * 100}%')
#     return imgs, lbls