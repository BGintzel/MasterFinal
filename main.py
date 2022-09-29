import sys
import os

from functions import run_training, use_EN, get_accuracy, run_ist, find_and_plot_1, find_and_plot_2, find_and_plot_3



if __name__ == '__main__':
    dirname = os.path.dirname(__file__)

    # make_new_experiment_arcs()
    if sys.argv[1] == 'train':
        run = sys.argv[2]
        run_training(run)

    elif sys.argv[1] == 'use':
        model_path = sys.argv[2]
        img1_path = os.path.join(dirname, sys.argv[3])
        img2_path = os.path.join(dirname, sys.argv[4])

        use_EN(model_path, img1_path, img2_path)

    elif sys.argv[1] == 'check_accuracy':
        model_path = sys.argv[2]
        get_accuracy(model_path)

    elif sys.argv[1] == 'find':
        model_path = sys.argv[2]
        input_type = int(sys.argv[3])
        if input_type==1:
            find_and_plot_1(model_path)
        elif input_type==2:
            find_and_plot_2(model_path)
        elif input_type==3:
            find_and_plot_3(model_path)

    elif sys.argv[1] == 'IST':
        iterations = int(sys.argv[3])
        model_path = sys.argv[2]
        run_ist(iterations, model_path)
