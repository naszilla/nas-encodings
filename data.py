import numpy as np
import pickle
import sys
import os

"""
currently we need to set an environment variable in 'search_space',
and then import the correct Cell class based on the search space
TODO: handle this better by making Cell subclasses for each search space
"""
if 'search_space' not in os.environ or os.environ['search_space'] == 'nasbench':
    from nasbench import api
    from nas_bench.cell import Cell
elif os.environ['search_space'] == 'darts':
    from darts.arch import Arch
elif os.environ['search_space'][:12] == 'nasbench_201':
    from nas_201_api import NASBench201API as API
    from nas_bench_201.cell import Cell
else:
    print('Invalid search space environ in data.py')
    sys.exit()


class Data:

    def __init__(self, 
                 search_space,
                 dataset='cifar10', 
                 nasbench_folder='./', 
                 loaded_nasbench=None):

        self.search_space = search_space
        self.dataset = dataset
        self.index_hash = pickle.load(open(os.path.expanduser('~/nas_encodings/index_hash.pkl'), 'rb'))

        if loaded_nasbench:
            self.nasbench = loaded_nasbench
        elif search_space == 'nasbench':
            self.nasbench = api.NASBench(nasbench_folder + 'nasbench_only108.tfrecord')
        elif search_space == 'nasbench_201':
            self.nasbench = API(os.path.expanduser('~/nas-bench-201/NAS-Bench-201-v1_0-e61699.pth'))
        elif search_space != 'darts':
            print(search_space, 'is not a valid search space')
            sys.exit()

    def get_type(self):
        return self.search_space

    def convert_to_cells(self,
                         arches,
                         predictor_encoding='path',
                         cutoff=0,
                         train=True):
        cells = []
        for arch in arches:
            spec = Cell.convert_to_cell(arch)
            cell = self.query_arch(spec,
                                   predictor_encoding=predictor_encoding,
                                   cutoff=cutoff,
                                   train=train)
            cells.append(cell)
        return cells

    def query_arch(self, 
                   arch=None, 
                   train=True, 
                   predictor_encoding=None, 
                   cutoff=0,
                   random_encoding='standard',
                   deterministic=True, 
                   epochs=0,
                   random_hash=False,
                   max_edges=None,
                   max_nodes=None):

        arch_dict = {}
        arch_dict['epochs'] = epochs
        if self.search_space in ['nasbench', 'nasbench_201']:
            if arch is None:

                arch = Cell.random_cell(self.nasbench,
                                        random_encoding=random_encoding, 
                                        max_edges=max_edges, 
                                        max_nodes=max_nodes,
                                        cutoff=cutoff,
                                        index_hash=self.index_hash)
            arch_dict['spec'] = arch
    
            if predictor_encoding:
                arch_dict['encoding'] = Cell(**arch).encode(predictor_encoding=predictor_encoding,
                                                            cutoff=cutoff)

            # special keys for local search and outside_ss experiments
            if self.search_space == 'nasbench_201' and random_hash:
                arch_dict['random_hash'] = Cell(**arch).get_random_hash()
            if self.search_space == 'nasbench':
                arch_dict['adj'] = Cell(**arch).encode(predictor_encoding='adj')
                arch_dict['path'] = Cell(**arch).encode(predictor_encoding='path')

            if train:
                arch_dict['val_loss'] = Cell(**arch).get_val_loss(self.nasbench, 
                                                                  deterministic=deterministic,
                                                                  dataset=self.dataset)
                arch_dict['test_loss'] = Cell(**arch).get_test_loss(self.nasbench,
                                                                    dataset=self.dataset)
                arch_dict['num_params'] = Cell(**arch).get_num_params(self.nasbench)
                arch_dict['val_per_param'] = (arch_dict['val_loss'] - 4.8) * (arch_dict['num_params'] ** 0.5) / 100

                if self.search_space == 'nasbench':
                    arch_dict['dist_to_min'] = arch_dict['val_loss'] - 4.94457682
                elif self.dataset == 'cifar10':
                    arch_dict['dist_to_min'] = arch_dict['val_loss'] - 8.3933
                elif self.dataset == 'cifar100':
                    arch_dict['dist_to_min'] = arch_dict['val_loss'] - 26.5067
                else:
                    arch_dict['dist_to_min'] = arch_dict['val_loss'] - 53.2333

        else:
            # if the search space is DARTS
            if arch is None:
                arch = Arch.random_arch()

            arch_dict['spec'] = arch

            if predictor_encoding == 'path':
                encoding = Arch(arch).encode_paths()
            elif predictor_encoding == 'trunc_path':
                encoding = Arch(arch).encode_freq_paths()
            else:
                encoding = arch

            arch_dict['encoding'] = encoding

            if train:
                if epochs == 0:
                    epochs = 50
                arch_dict['val_loss'], arch_dict['test_loss'] = Arch(arch).query(epochs=epochs)
        
        return arch_dict           

    def mutate_arch(self, 
                    arch, 
                    mutation_rate=1.0, 
                    mutate_encoding='adj',
                    cutoff=0):

        if self.search_space in ['nasbench', 'nasbench_201']:
            return Cell(**arch).mutate(self.nasbench,
                                       mutation_rate=mutation_rate,
                                       mutate_encoding=mutate_encoding,
                                       index_hash=self.index_hash,
                                       cutoff=cutoff)
        else:
            return Arch(arch).mutate(int(mutation_rate))

    def get_nbhd(self, arch, mutate_encoding='adj'):
        if self.search_space == 'nasbench':
            return Cell(**arch).get_neighborhood(self.nasbench, 
                                                 mutate_encoding=mutate_encoding,
                                                 index_hash=self.index_hash)
        elif self.search_space == 'nasbench_201':
            return Cell(**arch).get_neighborhood(self.nasbench, 
                                                 mutate_encoding=mutate_encoding)
        else:
            return Arch(arch).get_neighborhood()

    def get_hash(self, arch):
        # return a unique hash of the architecture+fidelity
        # we use path indices + epochs
        if self.search_space == 'nasbench':
            return Cell(**arch).get_path_indices()
        elif self.search_space == 'darts':
            return Arch(arch).get_path_indices()[0]
        else:
            return Cell(**arch).get_string()

    def generate_random_dataset(self,
                                num=10, 
                                train=True,
                                predictor_encoding=None, 
                                random_encoding='adj',
                                deterministic_loss=True,
                                patience_factor=5,
                                allow_isomorphisms=False,
                                cutoff=0,
                                max_edges=None,
                                max_nodes=None):
        """
        create a dataset of randomly sampled architectues
        test for isomorphisms using a hash map of path indices
        use patience_factor to avoid infinite loops
        """
        data = []
        dic = {}
        tries_left = num * patience_factor
        while len(data) < num:
            tries_left -= 1
            if tries_left <= 0:
                break

            arch_dict = self.query_arch(train=train,
                                        predictor_encoding=predictor_encoding,
                                        random_encoding=random_encoding,
                                        deterministic=deterministic_loss,
                                        cutoff=cutoff,
                                        max_edges=max_edges,
                                        max_nodes=max_nodes)

            h = self.get_hash(arch_dict['spec'])

            if allow_isomorphisms or h not in dic:
                dic[h] = 1
                data.append(arch_dict)
        return data


    def get_candidates(self, 
                       data, 
                       num=100,
                       acq_opt_type='mutation',
                       predictor_encoding=None,
                       mutate_encoding='adj',
                       loss='val_loss',
                       allow_isomorphisms=False, 
                       patience_factor=5, 
                       deterministic_loss=True,
                       num_arches_to_mutate=1,
                       max_mutation_rate=1,
                       cutoff=0):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """

        candidates = []
        # set up hash map
        dic = {}
        for d in data:
            arch = d['spec']
            h = self.get_hash(arch)
            dic[h] = 1

        if acq_opt_type in ['mutation', 'mutation_random']:
            # mutate architectures with the lowest loss
            best_arches = [arch['spec'] for arch in sorted(data, key=lambda i:i[loss])[:num_arches_to_mutate * patience_factor]]

            # stop when candidates is size num
            # use patience_factor instead of a while loop to avoid long or infinite runtime
            for arch in best_arches:
                if len(candidates) >= num:
                    break
                for i in range(num // num_arches_to_mutate // max_mutation_rate):
                    for rate in range(1, max_mutation_rate + 1):
                        mutated = self.mutate_arch(arch, 
                                                   mutation_rate=rate, 
                                                   mutate_encoding=mutate_encoding)
                        arch_dict = self.query_arch(mutated, 
                                                    train=False,
                                                    predictor_encoding=predictor_encoding,
                                                    cutoff=cutoff)
                        h = self.get_hash(mutated)

                        if allow_isomorphisms or h not in dic:
                            dic[h] = 1    
                            candidates.append(arch_dict)

        if acq_opt_type in ['random', 'mutation_random']:
            # add randomly sampled architectures to the set of candidates
            for _ in range(num * patience_factor):
                if len(candidates) >= 2 * num:
                    break

                arch_dict = self.query_arch(train=False, 
                                            predictor_encoding=predictor_encoding,
                                            cutoff=cutoff)
                h = self.get_hash(arch_dict['spec'])

                if allow_isomorphisms or h not in dic:
                    dic[h] = 1
                    candidates.append(arch_dict)

        return candidates

    def remove_duplicates(self, candidates, data):
        # input: two sets of architectues: candidates and data
        # output: candidates with arches from data removed

        dic = {}
        for d in data:
            dic[self.get_hash(d['spec'])] = 1
        unduplicated = []
        for candidate in candidates:
            if self.get_hash(candidate['spec']) not in dic:
                dic[self.get_hash(candidate['spec'])] = 1
                unduplicated.append(candidate)
        return unduplicated

    def encode_data(self, dicts):
        # input: list of arch dictionary objects
        # output: xtrain (in binary path encoding), ytrain (val loss)

        data = []
        for dic in dicts:
            arch = dic['spec']
            encoding = Arch(arch).encode_paths()
            data.append((arch, encoding, dic['val_loss_avg'], None))
        return data

    def get_arch_list(self,
                      aux_file_path,
                      distance=None,
                      iteridx=0,
                      num_top_arches=5,
                      max_edits=20,
                      num_repeats=5,
                      random_encoding='adj',
                      verbose=1):
        # Method used for gp_bayesopt

        if self.search_space == 'darts':
            print('get_arch_list only supported for nasbench and nasbench_201')
            sys.exit()

        # load the list of architectures chosen by bayesopt so far
        base_arch_list = pickle.load(open(aux_file_path, 'rb'))
        top_arches = [archtuple[0] for archtuple in base_arch_list[:num_top_arches]]
        if verbose:
            top_5_loss = [archtuple[1][0] for archtuple in base_arch_list[:min(5, len(base_arch_list))]]
            print('top 5 val losses {}'.format(top_5_loss))

        # perturb the best k architectures    
        dic = {}
        for archtuple in base_arch_list:
            path_indices = Cell(**archtuple[0]).get_path_indices()
            dic[path_indices] = 1

        new_arch_list = []
        for arch in top_arches:
            for edits in range(1, max_edits):
                for _ in range(num_repeats):
                    #perturbation = Cell(**arch).perturb(self.nasbench, edits)
                    perturbation = Cell(**arch).mutate(self.nasbench, edits)
                    path_indices = Cell(**perturbation).get_path_indices()
                    if path_indices not in dic:
                        dic[path_indices] = 1
                        new_arch_list.append(perturbation)

        # make sure new_arch_list is not empty
        while len(new_arch_list) == 0:
            for _ in range(100):
                arch = Cell.random_cell(self.nasbench, random_encoding=random_encoding)
                path_indices = Cell(**arch).get_path_indices()
                if path_indices not in dic:
                    dic[path_indices] = 1
                    new_arch_list.append(arch)

        return new_arch_list

    # Method used for gp_bayesopt for nasbench
    @classmethod
    def generate_distance_matrix(cls, arches_1, arches_2, distance):
        matrix = np.zeros([len(arches_1), len(arches_2)])
        for i, arch_1 in enumerate(arches_1):
            for j, arch_2 in enumerate(arches_2):
                matrix[i][j] = Cell(**arch_1).distance(Cell(**arch_2), dist_type=distance)
        return matrix
