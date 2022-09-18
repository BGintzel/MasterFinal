import sys

from functions import run_training, use_EN

if __name__ == '__main__':
    # make_new_experiment_arcs()
    if sys.argv[0]=='train':
        run_training(sys.argv)
    elif sys.argv[0]=='use':
        use_EN(sys.argv)