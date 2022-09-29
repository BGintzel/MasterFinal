import trainrunner
import utils
import numpy as np

from IST.start_self_training import circular_training
from find_and_plot.experiments import make_new_experiment_arcs, make_new_experiment_colors, make_experiments



def run_training(run):
    net_name = 'EN' + run
    writer = utils.set_new_t_board(name=net_name)
    utils.save_file('stats/stats_' + run, trainrunner.trainer_EN(run, writer))


def use_EN(path, img1_path, img2_path):
    device = utils.set_device()
    model = utils.load_model(path)
    model.to(device)
    model.eval()

    input = utils.parse_images_to_input(img1_path, img2_path)

    np.set_printoptions(precision=5, suppress=True)
    print(f'Output: {model(input).cpu().detach().numpy()[0]}')


def get_accuracy(path):
    device = utils.set_device()
    model = utils.load_model(path)
    model.to(device)
    model.eval()

    accuracy = trainrunner.test_EN(model)
    print(f"The accuracy over the Test data set is: {accuracy}%")


def run_ist(iterations, model_path):
    circular_training(iterations, model_path)


def find_and_plot_1(model_path):
    make_experiments(model_path)


def find_and_plot_2(model_path):
    make_new_experiment_arcs(model_path)


def find_and_plot_3(model_path):
    make_new_experiment_colors(model_path)
