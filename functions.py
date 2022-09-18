import trainrunner
import utils


def run_training(args):
    run = args[2]
    net_name = 'EN' + run
    writer = utils.set_new_t_board(name=net_name)
    utils.save_file('stats/stats_' + run, trainrunner.trainer_EN(run, writer))


def use_EN(args):
    device = utils.set_device()
    path = args[2]

    model = utils.load_model(path)

    model.to(device)

    img1_path = args[3]
    img2_path = args[4]

    input = parse_images_to_input(img1_path, img2_path)





