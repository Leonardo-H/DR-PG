# Doubly-Robust Policy Gradient 

Code for Doubly-Robust Policy Gradient (DR-PG) Algorithm in <https://arxiv.org/abs/1910.09066>. If you find DR-PG helpful, please cite as follow:

```
@article{huang2019importance,
  title={From Importance Sampling to Doubly Robust Policy Gradient},
  author={Huang, Jiawei and Jiang, Nan},
  journal={arXiv preprint arXiv:1910.09066},
  year={2019}
}
```

Our code is based on and reuses the code from the following paper: 

> Trajectory-wise Control Variates for Variance Reduction in Policy Gradient Methods.
>
> Ching-An Cheng\*, Xinyan Yan\*, Byron Boots. CoRL 2019. (*: equal contribution).

Their original code can be found in https://github.com/gtrll/rlfamily_cv.



## Installation

Please follow the instruction in `Installation.md` to install the environments.



## Running Experiments

### Exp 1: Variance_Reduction

```shell
# Train Models
python var_exp.py cp cp --type train -r dr

# Generate Rollouts
python var_exp.py cp cp --type gen-ro --show-freq 5

# Calculate Gradients for Each Rollouts, 
# The choices of 'your_cv_type' include:
#		(1) nocv:	Standard PG
#		(2) st	:	State-Dependent Baseline
#		(3) sa	:	State-Action-Dependent Baseline
#		(4) traj:	Trajectory-wise Baseline
#		(5) dr	:	Doubly-Robust PG
python var_exp.py cp cp --type cal-var --show-freq 5 -r your_cv_type

# Calculate Estimated Mean via State-Dependent Baseline
python var_exp.py cp cp --type est-mean --show-freq 5
```

Plot results

```shell
python scripts/var_exp_plot.py
```

By default, we omit the standard PG while ploting. If you want to compare togther with it, use the following command.

```shell
python scripts/var_exp_plot.py --show-nocv
```



### Exp 2: Optimization

Set your own seeds in `./Optimization/scripts/ranges_cv.py`, Line 9. The default setting is:

```
['general', 'seed'], [x * 10000 + 5000 for x in range(10)]
```

Run the experiments

```sh
# standard policy gradient
python opt_exp.py cp cp -r nocv

# state-dependent baseline
python opt_exp.py cp cp -r st

# state-action-dependent baseline
python opt_exp.py cp cp -r sa

# trajectory-wise control variate
python opt_exp.py cp cp -r traj

# doubly-robust policy gradient
python opt_exp.py cp cp -r dr

```

Plot Curves with Error Bar

```shell
# Median over trials with different seeds
python scripts/opt_exp_plot.py --logdir_parent ./log --value MeanSumOfRewards --curve median

# Mean over trials with different seeds
python scripts/opt_exp_plot.py --logdir_parent ./log --value MeanSumOfRewards --curve mean

```

## Contact

If you have any questions about the code or paper, please feel free to [contact us](mailto:jiaweileonardo@outlook.com).