# -*- coding: utf-8 -*-

import argparse


def get_common_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_id', type=str, default='Pinball')
    parser.add_argument('--log_dir', type=str, default='./log', help='where to save the log files')
    parser.add_argument('--model_dir', type=str, default='./ckpt', help='where to save the ckpt files')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use GPU')
    parser.add_argument('--gpu', type=str, default=None, help='which gpu to use')
    parser.add_argument('--render', type=bool, default=False, help='whether to render the env')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode or not')

    args = parser.parse_args()
    return args


def get_laprepr_args(args):

    args.d = 20 # the smallest d eigenvectors
    args.n_samples = 100000 # the total number of samples for training
    args.w_neg = 1.0
    args.c_neg = 1.0
    args.reg_neg = 0.0
    args.generalized = False # generalized spectral drawing or not

    args.use_position_only = True # important hyperparameters
    args.lap_n_layers = 3
    args.lap_n_units = 256

    args.lap_batch_size = 128
    args.lap_discount = 0.9 # important hyperparameters
    args.lap_replay_buffer_size = 10000 # in fact 10000 * epi_length; original parameter: 100000,
    args.lap_opt_args_name = 'Adam'
    args.lap_opt_args_lr = 0.001
    args.lap_sub_dir = 'laprepr'
    args.lap_train_steps = 30000
    args.lap_print_freq = 1000
    args.lap_save_freq = 10000

    return args

def get_clustering_args(args):
    args.sc_n_samples = -1  # number of samples for spectral clustering; -1 for all the samples in the replay buffer
    args.inerval = 10000 # we cannot feed all the data points into the NN directly
    args.sc_normlized = True # whether to normalize the embeddings (rows)
    # estimate eigenvalues
    args.ev_n_samples = 100000
    args.gap_threshold = 0.1 # all the eigenvalues for the normalized Laplacian matrix should be within [0,1], important hyperparameter
    # soft k-means (fuzzy c-means)
    args.km_range = 1 # the range of the cluster numbers for try
    args.km_m = 2
    args.km_error = 5e-4
    args.km_max_iters = 10000
    # cluster center and boundary
    args.sub_goal_threshold = 0.1 # threshold for the difference to the max likelihood, important hyperparameter
    args.sub_goal_num_limit = 5
    return args

def get_rl_args(args):
    # network
    args.rl_n_layers = 3
    args.rl_n_units = 256
    # training
    args.rl_batch_size = 128
    args.rl_opt_args_name = 'Adam'
    args.rl_opt_args_lr = 0.001
    args.rl_discount = 0.98
    args.rl_update_freq = 50
    args.rl_update_rate = 0.05

    args.epsilon_greedy = 0.2
    args.rl_train_steps = 30000
    args.rl_print_freq = 1000
    args.rl_save_freq = 10000
    args.rl_test_freq = 1000
    args.rl_test_episodes = 50
    # replay buffer
    args.rl_replay_buffer_size = 10000 # number of episodes, where the length of the episodes vary
    args.rl_replay_buffer_init = 1000 # number of episodes, collected with random policy
    args.rl_replay_update_freq = 10 # how often do we collect transitions, important
    args.rl_replay_update_num = 10 # how many episodes to sample for once, important

    return args

def get_option_args(args):
    args.opt_sub_dir = 'opt_learner'
    args.opt_reward_mode = 'lap' # 'l2', 'lap', 'raw', 'l2_mix': 'l2' + 'raw', 'lap_mix': 'lap' + 'raw'
    args.opt_use_fiedler_only = False
    args.opt_dist_reward_coeff = 1.0
    args.opt_sparse_reward = False # whether to change the original env reward to a sparse version: 0.0 if reaches; -1.0 otherwise;
    # for offine scenarios, we don't need to reuse the option dqn network, since we have enough time, and we prefer the dqn agent to overfit with its specific task
    # args.opt_reuse_network = False # if true, then we can reduce the number of training steps for rl

    return args

def get_planner_args(args):
    args.planner_sub_dir = 'planner'
    args.planner_reward_mode = 'lap' # 'lap', 'raw', 'lap_mix': 'lap' + 'raw'
    args.planner_use_fiedler_only = False
    args.planner_dist_reward_coeff = 1.0
    args.planner_sparse_reward = False
    args.planner_test_episodes = 50

    return args

if __name__ == '__main__':
    import os
    args = get_common_args()
    args = get_laprepr_args(args)
    saver_dir = os.path.join(args.log_dir, args.lap_sub_dir)
    if not os.path.exists(saver_dir):
        os.makedirs(saver_dir)
