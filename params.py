import sys


def algo_params(param_str, queries=150):
    """
      Return params list based on param_str.
      These are the parameters used to produce the figures in the paper
      For AlphaX and Reinforcement Learning, we used the corresponding github repos:
      https://github.com/linnanwang/AlphaX-NASBench101
      https://github.com/automl/nas_benchmarks
    """
    params = []

    if param_str == 'main_experiments':
        params.append({'algo_name':'bananas', 'total_queries':queries})   
        params.append({'algo_name':'random', 'total_queries':queries})
        params.append({'algo_name':'evolution', 'total_queries':queries})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries})       
        params.append({'algo_name':'dngo', 'total_queries':queries})
        params.append({'algo_name':'local_search', 'total_queries':queries})

    elif param_str == 'ablation':
        params.append({'algo_name':'bananas', 'total_queries':queries})   
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'adj'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'path_distance'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'adj_distance'})
        params.append({'algo_name':'bananas', 'total_queries':queries, 'acq_opt_type':'random'})

    elif param_str == 'test_simple':
        queries = 30
        params.append({'algo_name':'random', 'total_queries':queries})

    elif param_str == 'test_algos':
        queries = 30
        params.append({'algo_name':'bananas', 'total_queries':queries})   
        params.append({'algo_name':'random', 'total_queries':queries})
        params.append({'algo_name':'evolution', 'total_queries':queries})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries})
        params.append({'algo_name':'dngo', 'total_queries':queries})
        params.append({'algo_name':'local_search', 'total_queries':queries})

    elif param_str == 'bananas':
        params.append({'algo_name':'bananas', 'total_queries':queries, 'acq_opt_type':'mutation', 'explore_type':'its'})

    elif param_str == 'bo_encodings':
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'cont_path'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'trunc_cont_path'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'trunc_path'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'cont_adj'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'path'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'adj'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'nasbot'})

    elif param_str == 'random_encodings':
        params.append({'algo_name':'random', 'total_queries':queries, 'random_encoding':'uniform'})
        params.append({'algo_name':'random', 'total_queries':queries, 'random_encoding':'adj'})
        params.append({'algo_name':'random', 'total_queries':queries, 'random_encoding':'cont_adj'})
        params.append({'algo_name':'random', 'total_queries':queries, 'random_encoding':'path'})
        params.append({'algo_name':'random', 'total_queries':queries, 'random_encoding':'trunc_path'})
        params.append({'algo_name':'random', 'total_queries':queries, 'random_encoding':'cont_path'})
        params.append({'algo_name':'random', 'total_queries':queries, 'random_encoding':'trunc_cont_path'})
        params.append({'algo_name':'random', 'total_queries':queries, 'random_encoding':'wtd_path'})
        params.append({'algo_name':'random', 'total_queries':queries, 'random_encoding':'wtd_trunc_path'})
        params.append({'algo_name':'random', 'total_queries':queries, 'random_encoding':'wtd_cont_path'})
        params.append({'algo_name':'random', 'total_queries':queries, 'random_encoding':'wtd_trunc_cont_path'})

    elif param_str == 'ls_encodings':
        params.append({'algo_name':'local_search', 'total_queries':queries, 'mutate_encoding':'path'})
        params.append({'algo_name':'local_search', 'total_queries':queries, 'mutate_encoding':'trunc_path'})
        params.append({'algo_name':'local_search', 'total_queries':queries, 'mutate_encoding':'adj'})

    elif param_str == 'bananas_encodings':
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'cont_adj'})
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'path'})
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'trunc_path'})
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'cat_path'})
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'trunc_cat_path'})
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'adj'})
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'cat_adj'})

    elif param_str == 'evo_encodings':
        params.append({'algo_name':'evolution', 'total_queries':queries, 'mutate_encoding':'adj'})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'mutate_encoding':'cont_adj'})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'mutate_encoding':'path'})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'mutate_encoding':'trunc_path'})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'mutate_encoding':'cont_path'})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'mutate_encoding':'trunc_cont_path'})

    elif param_str == 'bananas_all':
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'trunc_path', 'mutate_encoding':'cat_adj', 'random_encoding':'adj'})
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'trunc_path', 'mutate_encoding':'trunc_path', 'random_encoding':'trunc_path'})
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'adj', 'mutate_encoding':'adj', 'random_encoding':'adj'})

    elif param_str == 'bo_encodings_201':
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'trunc_path_distance'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'path_distance'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'adj_distance'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'nasbot_distance'})

    elif param_str == 'bananas_201':
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'path'})
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'trunc_path'})
        params.append({'algo_name':'bananas', 'total_queries':queries, 'predictor_encoding':'adj'})

    elif param_str == 'evo_201':
        params.append({'algo_name':'evolution', 'total_queries':queries, 'mutate_encoding':'adj'})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'mutate_encoding':'path'})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'mutate_encoding':'trunc_path'})

    elif param_str == 'evo_trunc':
        queries = 300
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':1})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':2})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':4})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':7})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':10})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':13})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':16})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':19})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':22})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':25})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':28})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_path', 'cutoff':31})

        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':1})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':2})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':4})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':7})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':10})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':13})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':16})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':19})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':22})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':25})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':28})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'num_init':1,  'mutate_encoding':'trunc_adj', 'cutoff':31})

    else:
        print('invalid algorithm params: {}'.format(param_str))
        sys.exit()

    print('\n* Running experiment: ' + param_str)
    return params


def meta_neuralnet_params(param_str):

    if param_str == 'nasbench':
        params = {'search_space':'nasbench', 'dataset':'cifar10', 'mf':False, 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}

    elif param_str == 'darts':
        params = {'search_space':'darts', 'dataset':'cifar10', 'mf':False, 'loss':'mape', 'num_layers':10, 'layer_width':20, \
            'epochs':10000, 'batch_size':32, 'lr':.00001, 'regularization':0, 'verbose':0}

    elif param_str == 'nasbench_outside':
        params = {'search_space':'nasbench', 'dataset':'cifar10', 'mf':False, 'loss':'mse', 'num_layers':10, 'layer_width':20, \
            'epochs':500, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}

    elif param_str == 'nasbench_201_cifar10':
        params = {'search_space':'nasbench_201', 'dataset':'cifar10', 'mf':False, 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}

    elif param_str == 'nasbench_201_cifar100':
        params = {'search_space':'nasbench_201', 'dataset':'cifar100', 'mf':False, 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}

    elif param_str == 'nasbench_201_imagenet':
        params = {'search_space':'nasbench_201', 'dataset':'ImageNet16-120', 'mf':False, 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}

    else:
        print('invalid meta neural net params: {}'.format(param_str))
        sys.exit()

    return params
