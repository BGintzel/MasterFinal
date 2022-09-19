import trainrunner
import utils
import os
import numpy as np


def run_training(args):
    run = args[2]
    net_name = 'EN' + run
    writer = utils.set_new_t_board(name=net_name)
    utils.save_file('stats/stats_' + run, trainrunner.trainer_EN(run, writer))


def use_EN(args):
    device = utils.set_device()
    path = args[2]
    print(path)
    model = utils.load_model(path)

    model.to(device)
    model.eval()
    dirname = os.path.dirname(__file__)

    img1_path = os.path.join(dirname, args[3])
    img2_path = os.path.join(dirname, args[4])

    input = utils.parse_images_to_input(img1_path, img2_path)

    np.set_printoptions(precision=5, suppress=True)
    print(f'Output: {model(input).cpu().detach().numpy()[0]}')




