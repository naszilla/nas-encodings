# A Study on Encodings for Neural Architecture Search

**Note: this repository has been combined with other naszilla projects into [naszilla/naszilla](https://github.com/naszilla/naszilla). This repo is deprecated and not maintained. Please use [naszilla/naszilla](https://github.com/naszilla/naszilla), which has more functionality.**

[A Study on Encodings for Neural Architecture Search](https://arxiv.org/abs/2007.04965)\
Colin White, Willie Neiswanger, Sam Nolen, and Yash Savani.\
_arxiv:2007.04965_.

Many algorithms for neural architecture search (NAS) represent each neural architecture in the search space as a directed acyclic graph (DAG), and then search over all DAGs by encoding the adjacency matrix and list of operations as a set of hyperparameters. Recent work has demonstrated that even small changes to the way each architecture is encoded can have a significant effect on the performance of NAS algorithms. We present the first formal study on the effect of architecture encodings for NAS.

## Requirements
- jupyter
- tensorflow == 1.14.0 (used for all experiments)
- nasbench (follow the installation instructions [here](https://github.com/google-research/nasbench))
- nas-bench-201 (follow the installation instructions [here](https://github.com/D-X-Y/NAS-Bench-201))
- pytorch == 1.2.0, torchvision == 0.4.0 (used for experiments on the DARTS search space)
- pybnn (used only for the DNGO baselien algorithm. Installation instructions [here](https://github.com/automl/pybnn))

If you run experiments on the DARTS search space, you will need our fork of the DARTS repo:
- Download our fork of the DARTS repo: https://github.com/naszilla/darts
- If you don't put the repo in your home directory, i.e., ~/darts, then update line 7 of nas-encodings/darts/arch.py and line 8 of nas-encodings/train_arch_runner.py with the correct path.


#### Download nasbench-101
- Download the nasbench_only108 tfrecord file (size 499MB) [here](https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord)
- Place `nasbench_only108.tfrecord` in the top level folder of this repo

#### Download index-hash
Some of the path-based encoding methods require a hash map from path indices to cell architectures. We have created a pickle file which contains this hash map (size 57MB), located [here](https://drive.google.com/file/d/1yMRFxT6u3ZyfiWUPhtQ_B9FbuGN3X-Nf/view?usp=sharing). Place it in the top level folder of this repo.

## Get started quickly: open jupyter notebook
- The easiest way to get started is to run one of our jupyter notebooks
- Open and run `meta_neuralnet.ipynb` to train a neural predictor with different encodings
- Open and run `notebooks/test_nas.ipynb` to test out each algorithm + encoding combination

## Run experiments on nasbench-101
```bash
python run_experiments_sequential.py --algo_params evo_encodings
```
This command will run evolutionary search with six different encodings. To run other experiments, open up `params.py`.

## Run experiments on nasbench-201
To run experiments with NAS-Bench-201, download `NAS-Bench-201-v1_0-e61699.pth` from [here](https://github.com/D-X-Y/NAS-Bench-201) and place it in the top level folder of this repo. Choose between cifar10, cifar100, and imagenet. For example,

```bash
python run_experiments_sequential.py --algo_params evo_encodings --search_space nasbench_201_cifar10
```

## Citation
Please cite [our paper](https://arxiv.org/abs/2007.04965) if you use code from this repo:

```bibtex
@inproceedings{white2020study,
  title={A Study on Encodings for Neural Architecture Search},
  author={White, Colin and Neiswanger, Willie and Nolen, Sam and Savani, Yash},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
