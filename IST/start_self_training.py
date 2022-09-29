import utils
import trainrunner

from IST import create_new_dataset

ITERATION = 0
PATH_TO_EN_INIT = 'models/EN.pth'

PATH_TO_EN = 'models/current_EN.pth'


def run_training_one_step():
    run = f'iteration_{ITERATION}'
    net_name = 'Iteration' + run
    if ITERATION == 0:
        path_en = PATH_TO_EN_INIT
    else:
        path_en = PATH_TO_EN

    current_en = utils.load_model(path_en)

    train_dataloader, valid_dataloader = create_new_dataset.create(current_en, ITERATION)

    utils.save_file('stats/stats_' + run,
                    trainrunner.trainer(run, nb_epoch=10, model=current_en,
                                      train_loader=train_dataloader,
                                      valid_loader=valid_dataloader))
    print(f'Finished Training iteration {ITERATION}!')


def circular_training(iterations=10, model_path='models/EN.pth'):
    global ITERATION
    global PATH_TO_EN_INIT
    PATH_TO_EN_INIT = model_path
    for iter_ in range(iterations):
        run_training_one_step()
        print(f'Finished iteration {iter_}!')
        ITERATION += 1

