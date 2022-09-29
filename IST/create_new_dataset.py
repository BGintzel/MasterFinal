import data
import utils
import numpy as np
import random

from IST import create_circles


def get_original_images():
    images = utils.load_file('datasets/images')
    labels = utils.load_file('../datasets/labels')
    random_list = random.sample(range(0, len(images)), 1000)
    images = [images[i] for i in random_list]
    labels = [labels[i] for i in random_list]
    return np.asarray(images), np.asarray(labels)


def get_unexpected_images(model):
    imgs, labels = create_circles.get_random_circles_for_training(model, number_of_samples=1000)

    imgs = [img.cpu().detach().numpy() for img in imgs]
    labels = [label.cpu().detach().numpy() for label in labels]

    imgs = np.asarray(imgs)
    labels = np.asarray(labels)

    imgs = np.reshape(imgs, (imgs.shape[0], 3, 64, 64, 2))
    imgs = np.transpose(imgs, (0, 2, 3, 1, 4))
    return imgs, labels


def create(model, iteration):
    print('Started creating new Dataset')
    # imgs_new, labels_new = get_unexpected_images(model)
    # imgs_old, labels_old = get_original_images()
    #
    # imgs, labels = utils.combine_both(imgs_new, imgs_old, labels_new, labels_old)

    imgs, labels = get_unexpected_images(model)
    utils.save_file(f'datasets/images_iteration_{iteration}', imgs)
    utils.save_file(f'datasets/labels_iteration_{iteration}', labels)

    train_dataloader, valid_dataloader = data.get_new_dataloaders(imgs, labels)
    print('Finished creating new dataset!')
    return train_dataloader, valid_dataloader


def create_dataset_from_images_and_labels(imgs, labels):
    train_dataloader, valid_dataloader = data.get_new_dataloaders(imgs, labels)
    print('Finished creating new dataset!')
    return train_dataloader, valid_dataloader
