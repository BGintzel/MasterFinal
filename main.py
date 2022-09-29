import sys
import os

from functions import run_training, use_EN, get_accuracy, run_ist, find_and_plot_1, find_and_plot_2, find_and_plot_3



if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    print(sys.argv)
    i=0
    if sys.argv[1] == 'main.py':
        i=1
    # make_new_experiment_arcs()
    if sys.argv[1+i] == 'train':
        run = sys.argv[2+i]
        run_training(run)

    elif sys.argv[1+i] == 'use':
        model_path = sys.argv[2+i]
        img1_path = os.path.join(dirname, sys.argv[3+i])
        img2_path = os.path.join(dirname, sys.argv[4+i])

        use_EN(model_path, img1_path, img2_path)

    elif sys.argv[1+i] == 'check':
        model_path = sys.argv[2+i]
        get_accuracy(model_path)

    elif sys.argv[1+i] == 'find':
        model_path = sys.argv[2+i]
        input_type = int(sys.argv[3+i])
        if input_type==1:
            find_and_plot_1(model_path)
        elif input_type==2:
            find_and_plot_2(model_path)
        elif input_type==3:
            find_and_plot_3(model_path)

    elif sys.argv[1+i] == 'IST':
        iterations = int(sys.argv[3+i])
        model_path = sys.argv[2+i]
        run_ist(iterations, model_path)
