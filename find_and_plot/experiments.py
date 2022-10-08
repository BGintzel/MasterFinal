import numpy as np

from find_and_plot.overviews import get_overview_colors

from find_and_plot.overviews import get_overview_arcs, get_overview_incomplete
from find_and_plot.train_random_exps import train_random_circles


def make_experiments(model_path):
    # [Middel_circle, other_circle, different_middle_color, random_position_and_radius, shifted]
    experiments = [[False, True, False, True, False],  # normal_random_position
                   [True, False, False, True, False]]  # only_middle_random_position

    iterations = 1000
    labels = [[0, 1, 0, 0]]

    for j, label in enumerate(labels):
        label = np.array(label, dtype='f')
        print(f'Starting Label {j}')
        for i, experiment in enumerate(experiments):
            train_random_circles(model_path=model_path, run=f'incomplete_{i}_{label}', experiment=experiment,
                                 iterations=iterations, label=label)
    get_overview_incomplete(model_path, label, 'incomplete')



def make_new_experiment_arcs(model_path):
    label1 = np.array([0, 1, 0, 0], dtype='f')

    iterations = 1000
    run = f'arc_{label1}'
    train_random_circles(model_path=model_path, experiment=[True, True, False, True, False], run=run,
                         iterations=iterations, label=label1, arc=True)

    get_overview_arcs(model_path, label1, run)


def make_new_experiment_colors(model_path, color=(192/255, 128/255, 192/255)):
    label = np.array([0, 1, 0, 0], dtype='f')

    iterations = 1000
    run = f'color_{label}'
    train_random_circles(model_path=model_path,experiment=[True, True, False, True, False], run=f'color_{label}',
                         iterations=iterations, label=label, colors=True, color=color)
    #
    # train_random_circles(experiment=[True, True, False, True, False], run=f'run_color__{label2}_full_green_circles',
    #                      iterations=iterations, label=label2, colors=True, color=color)

    get_overview_colors(model_path, label, run)
