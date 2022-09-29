import trainrunner
import utils
import numpy as np

from IST.start_self_training import circular_training


def run_training(run):
    net_name = 'EN' + run
    writer = utils.set_new_t_board(name=net_name)
    utils.save_file('stats/stats_' + run, trainrunner.trainer_EN(run, writer))


def use_EN(path, img1_path, img2_path):
    device = utils.set_device()

    print(path)
    model = utils.load_model(path)
    print(model)
    model.to(device)
    model.eval()

    input = utils.parse_images_to_input(img1_path, img2_path)

    np.set_printoptions(precision=5, suppress=True)
    print(f'Output: {model(input).cpu().detach().numpy()[0]}')

def do_experiment_one(path):
    device = utils.set_device()
    print(path)
    model = utils.load_model(path)
    model.to(device)
    model.eval()

    accuracy = trainrunner.test_EN(model)
    print(accuracy)


def run_ist(iterations, model_path):
    circular_training(iterations, model_path)