import utils


def run_training(args):
    run = args[1]
    net_name = 'EN' + run
    writer = utils.set_new_t_board(name=net_name)
    utils.save_file('stats/stats_' + run, trainrunner.trainer_combine(run, writer))
