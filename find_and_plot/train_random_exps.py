import numpy as np
from find_and_plot import train_circles

import utils
import trainrunner

def get_random_images(angle, number=1):
    imgs, labels = train_circles.get_random_circles_for_testing(angle, number_of_samples=number)

    # imgs = [img.cpu().detach().numpy() for img in imgs]
    labels = [label.cpu().detach().numpy() for label in labels]

    imgs = np.asarray(imgs)
    labels = np.asarray(labels)

    imgs = np.reshape(imgs, (imgs.shape[0], 3, 64, 64, 2))
    imgs = np.transpose(imgs, (0, 2, 3, 1, 4))
    return imgs, labels


def create_new_dataset_random(start=0, number=100):
    for i in range(start, 361, 60):
        print(f'Started creating new Dataset for angle {i}')
        imgs, labels = get_random_images(i, number=number)
        utils.save_file(f'datasets/images_random_arc_{i}', imgs)
        utils.save_file(f'datasets/labels_random_arc_{i}', labels)

        print(f'Finished creating new dataset for angle {i}!')
    return


def get_images_color(color, number=1):
    imgs, labels = train_circles.get_random_circles_for_testing_color(color, number_of_samples=number)

    # imgs = [img.cpu().detach().numpy() for img in imgs]
    labels = [label.cpu().detach().numpy() for label in labels]

    imgs = np.asarray(imgs)
    labels = np.asarray(labels)

    imgs = np.reshape(imgs, (imgs.shape[0], 3, 64, 64, 2))
    imgs = np.transpose(imgs, (0, 2, 3, 1, 4))
    return imgs, labels


def create_new_dataset_color(r, g, b, number=100):
    print(f'Started creating new Dataset for color {r} {g} {b}')
    imgs, labels = get_images_color((r / 255, g / 255, b / 255), number=number)
    utils.save_file(f'datasets/images_random_color_{r}_{g}_{b}', imgs)
    utils.save_file(f'datasets/labels_random_color_{r}_{g}_{b}', labels)

    print(f'Finished creating new dataset for color {r} {g} {b}')


def create_color_datasets():
    for r in range(64, 255, 64):
        for g in range(64, 255, 64):
            for b in range(64, 255, 64):
                create_new_dataset_color(r, g, b)


#create_color_datasets()


def run_training():
    run = '99'
    net_name = 'combine' + run
    writer = utils.set_new_t_board(name=net_name)
    utils.save_file('stats/stats_' + run, trainrunner.trainer_combine(run, writer))


def run_training_generator(net, true_class, run, steps, epochs):
    run = str(run)
    net_name = 'generator' + run
    writer = utils.set_new_t_board(name=net_name)
    utils.save_file('stats/stats_' + run, trainrunner.trainer_generator(net, run, writer, true_class, steps, epochs))


def run_training_circles(net, true_class, run, steps, epochs):
    run = str(run)
    net_name = 'generator' + run
    writer = utils.set_new_t_board(name=net_name)
    utils.save_file('stats/stats_' + run, trainrunner.trainer_generator(net, run, writer, true_class, steps, epochs))


def train_random_circles(run, experiment, model_path, iterations=1000, label=np.array([0.0, 1.0, 0., 0.], dtype='f'), arc=False,
                         colors=False, color=None):
    best_images, best_loss, best_output, best_out = train_circles.train_random_circles(model_path, experiment, iterations, label,
                                                                                        arc=arc, colors=colors,
                                                                                       color=color)
    while len(best_images) == 0:
        print("Nothing found. Trying again")
        best_images, best_loss, best_output, best_out = train_circles.train_random_circles(experiment, iterations,
                                                                                           label, arc=arc)
    imgs = best_images[-1]

    img1 = (imgs[0, :, :, :, 0].numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    img2 = (imgs[0, :, :, :, 1].numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

    utils.save_file('experiments/' + run + '_out.pkl', best_out)
    utils.save_file('experiments/' + run + '_output.pkl', best_output)
    utils.save_file('experiments/' + run + '_imgs.pkl', best_images)
    utils.save_file('experiments/' + run + '_loss.pkl', best_loss)

    utils.save_image(img1, 'experiments/' + run + '1')
    utils.save_image(img2, 'experiments/' + run + '2')
    utils.save_txt('experiments/',run, best_loss, best_output)
