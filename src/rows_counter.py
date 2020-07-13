from multiprocessing import Pool
import os
from argparse import ArgumentParser


def func(filename):
    with open(filename) as file:
        suma = sum(1 for line in file) - 1
    return suma


if __name__ == '__main__':
    # Setup parameters
    parser = ArgumentParser()

    parser.add_argument(
        '--train_datapath',
        type=str,
        default='./data/train-final-complete'
    )

    hparams = parser.parse_args()
    train_path = hparams.train_datapath

    p = Pool(None)
    with p:
        results = p.map(func, [os.path.join(train_path, i) for
                               i in os.listdir(train_path) if i.endswith('csv')])

    lines = sum(results)
    counter_path = os.path.join(train_path, 'counter.txt')
    with open(counter_path, 'w') as f:
        f.write('%d' % lines)
