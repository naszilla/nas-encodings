import argparse
import time
import logging
import sys
import os
import pickle
import numpy as np
import copy

sys.path.append(os.path.expanduser('~/naszilla'))
from data import Data


"""
local_search_runner is used in run_experiments.sh
"""

def compute_next_arches(search_space, data, 
                        query=0,
                        filepath='trained_spec',
                        loss='val_loss_avg',
                        k=1):
    new_dicts = []
    best = -1
    best_val = 100
    last_chosen = -1
    for i, arch_dict in enumerate(data):
        if 'chosen' in arch_dict:
            last_chosen = i
        if arch_dict[loss] < best_val:
            best = i
            best_val = arch_dict[loss]

    new_chosen = -1
    if last_chosen == -1:
        # if we just finished the random initialization
        new_chosen = best

    if data[-1][loss] < data[last_chosen][loss]:
        # if the last architecture did better than its parent
        new_chosen = len(data) - 1

    print('last chosen', last_chosen, 'new chosen', new_chosen)
    if new_chosen >= 0:
        # get its neighbors and return them
        print('new chosen arch:', new_chosen, data[new_chosen][loss])
        neighbors = search_space.get_nbhd(data[new_chosen]['spec'])
        neighbors = [nbr['spec'] for nbr in neighbors]
        dict_with_nbrs = copy.deepcopy(data[new_chosen])
        dict_with_nbrs['neighbors'] = neighbors
        dict_with_nbrs['chosen'] = 1
        if 'parent' not in dict_with_nbrs:
            dict_with_nbrs['parent'] = last_chosen

        filename = '{}_{}.pkl'.format(filepath, dict_with_nbrs['index'])
        dict_with_nbrs['filepath'] = filename
        with open(filename, 'wb') as f:
            pickle.dump(dict_with_nbrs, f)

        for i in range(k):
            new_dicts.append({'spec':neighbors[i], 'parent':new_chosen})
        return new_dicts

    # try more neighbors from the last chosen architecture
    neighbors = data[last_chosen]['neighbors']
    if len(neighbors) <= len(data) - (last_chosen + 1):
        print('reached local minimum:', last_chosen, data[last_chosen][loss])
    else:
        nbr_index = len(data) - (last_chosen + 1)
        for i in range(nbr_index, min(len(neighbors) - 1, nbr_index + k)):
            new_dicts.append({'spec':neighbors[i], 'parent':last_chosen})
        return new_dicts

def run_local_search(args):

    save_dir = '{}/'.format(args.experiment_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    query = args.query
    num_init = args.num_init
    k = args.k
    trained_prefix = args.trained_filename
    untrained_prefix = args.untrained_filename

    search_space = Data('darts')

    # if it's the first iteration, choose k arches at random to train
    if query == 0:
        print('about to generate {} random'.format(num_init))
        data = search_space.generate_random_dataset(num=num_init, train=False)
        next_arches = [{'spec':d['spec']} for d in data]

    elif query < num_init:
        # if we're still training the initial arches, continue
        return

    else:
        # get the data from prior iterations from pickle files
        data = []
        for i in range(query):

            filepath = '{}{}_{}.pkl'.format(save_dir, trained_prefix, i)
            with open(filepath, 'rb') as f:
                arch = pickle.load(f)
            data.append(arch)

        print('Iteration {}'.format(query))
        print('Arches so far')
        for d in data:
            print(d['spec'])
            print('val_loss', d['val_loss_avg'])
            if 'chosen' in d and 'parent' in d:
                print('chosen', 'parent', d['parent'])
            elif 'chosen' in d:
                print('chosen')

        # run the meta neural net to output the next arches
        filepath = save_dir + trained_prefix
        next_arches = compute_next_arches(search_space, data, 
                                          query=query,
                                          filepath=filepath,
                                          k=k)

    print('next arch(es)')
    print([arch['spec'] for arch in next_arches])

    # output the new arches to pickle files
    num_to_train = k if query != 0 else num_init
    for i in range(num_to_train):
        index = query + i
        filepath = '{}{}_{}.pkl'.format(save_dir, untrained_prefix, index)
        next_arches[i]['index'] = index
        next_arches[i]['filepath'] = filepath
        with open(filepath, 'wb') as f:
            pickle.dump(next_arches[i], f)


def main(args):

    #set up save dir
    save_dir = './'

    #set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run_local_search(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for meta neural net')
    parser.add_argument('--experiment_name', type=str, default='ls_darts_test', help='Folder for input/output files')
    parser.add_argument('--trained_filename', type=str, default='trained_spec', help='name of input files')
    parser.add_argument('--untrained_filename', type=str, default='untrained_spec', help='name of output files')
    parser.add_argument('--query', type=int, default=0, help='What query is the algorithm on')
    parser.add_argument('--num_init', type=int, default=20, help='Number of initial random architectures')
    parser.add_argument('--k', type=int, default=1, help='Number of architectures per iteration')

    args = parser.parse_args()
    main(args)