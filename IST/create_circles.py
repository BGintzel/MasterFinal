import numpy as np
import torch.nn as nn
import math

import utils
from IST import plot_circles

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


def get_random_circles_for_training(en, number_of_samples=1000):
    device = utils.set_device()

    labels = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0],
              [1, 0, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]]

    BCE = nn.BCELoss()
    imgs = []
    lbls = []

    en.eval()

    for i, label in enumerate(labels):
        label = utils.make_tensor(label)
        threshold = 2
        counter_to_adjust_threshold = 0
        while len(imgs) < int(number_of_samples / len(labels)) * (i + 1):
            counter_to_adjust_threshold += 1
            middle = True
            other = True

            if i == 0 and len(imgs) < int(number_of_samples / (len(labels) * 2)):
                middle = False
            elif i == 0 and len(imgs) >= int(number_of_samples / (len(labels) * 2)):
                other = False

            # create circle pair
            images, out = plot_circles.plot_circle_for_random_training(middle, other)
            output_label = create_label(out)
            images.to(device)
            output = en(images)[0].cpu()

            loss = BCE(output, output_label)
            if counter_to_adjust_threshold % 100 == 0:
                threshold = threshold / 2
                print(f'Adjusted Threshold to {threshold} at Loss: {loss.item()}')
            if threshold < 0.01:
                imgs.append(images)
                lbls.append(output_label)

            if loss > threshold and (not utils.equal(label, output_label) or
                                     utils.equal(utils.make_tensor([0, 0, 0, 0]), output_label)):
                # not resetting threshold for label [0, 0, 0, 0] because the net learns it very fast
                if i > 0:
                    threshold = 2
                    counter_to_adjust_threshold = 0
                imgs.append(images)
                lbls.append(output_label)
                if len(imgs) % (int(0.05 * number_of_samples)+1) == 0:
                    print(f'{len(imgs)*100 / number_of_samples}% done')

        print(f'Label {i} dataset completed! Progress:{(i + 1) / (len(labels)) * 100}%')
    return imgs, lbls


def inference(op1, op2):
    relational_operator = ['>', '<', '&', '!']

    op3 = []
    if op1 == '>':
        if op2 == '>':
            op3 = ['>']
        if op2 == '<':
            op3 = ['<', '>', '&']
        if op2 == '&':
            op3 = ['&', '>']
        if op2 == '!':
            op3 = ['>', '!', '&']
    elif op1 == '<':
        if op2 == '>':
            op3 = ['>', '<', '&', "!"]
        if op2 == '<':
            op3 = ['<']
        if op2 == '&':
            op3 = ['&', '<', '!']
        if op2 == '!':
            op3 = ['!']
    elif op1 == '&':
        if op2 == '>':
            op3 = ['>', '&', '!']
        if op2 == '<':
            op3 = ['<', '&']
        if op2 == '&':
            op3 = ['&', '>', '<', '!']
        if op2 == '!':
            op3 = ['>', '!', '&']
    elif op1 == '!':
        if op2 == '>':
            op3 = ['!']
        if op2 == '<':
            op3 = ['<', '!', '&']
        if op2 == '&':
            op3 = ['&', '!', '<']
        if op2 == '!':
            op3 = ['>', '!', '&', '<']
    label = np.zeros(4)
    for op in op3:
        label[relational_operator.index(op)] = 1
    return op3, label
