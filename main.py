import sys
import os

from functions import run_training, use_EN, do_experiment_one, run_ist

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
        do_experiment_one(model_path)

    elif sys.argv[1] == 'IST':
        iterations = int(sys.argv[3])
        model_path = sys.argv[2]
        run_ist(iterations, model_path)
