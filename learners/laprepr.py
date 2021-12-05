import os
import logging
import collections
import random
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from configs import get_laprepr_args
from utils import episodic_replay_buffer, torch_tools, timer_tools, summary_tools
from learners.networks import ReprNetMLP
# from envs.simple_rl.mdp.MDPClass import MDP


def l2_dist(x1, x2, generalized):
    if not generalized:
        return (x1 - x2).pow(2).sum(-1)
    d = x1.shape[1]
    weight = np.arange(d, 0, -1).astype(np.float32)
    weight = torch_tools.to_tensor(weight, x1.device)
    return weight @ ((x1 - x2).pow(2))

def pos_loss(x1, x2, generalized=False):
    return l2_dist(x1, x2, generalized).mean()

# used in the original code
# def _rep_loss(inprods, n, k, c, reg):
#
#     norms = inprods[torch.arange(n), torch.arange(n)]
#     part1 = inprods.pow(2).sum() - norms.pow(2).sum()
#     part1 = part1 / ((n - 1) * n)
#     part2 = - 2 * c * norms.mean() / k
#     part3 = c * c / k
#     # regularization
#     # if reg > 0.0:
#     #     reg_part1 = norms.pow(2).mean()
#     #     reg_part2 = - 2 * c * norms.mean()
#     #     reg_part3 = c * c
#     #     reg_part = (reg_part1 + reg_part2 + reg_part3) / n
#     # else:
#     #     reg_part = 0.0
#     # return part1 + part2 + part3 + reg * reg_part
#     return part1 + part2 + part3

def _rep_loss(inprods, n, k, c, reg):

    norms = inprods[torch.arange(n), torch.arange(n)]
    part1 = (inprods.pow(2).sum() - norms.pow(2).sum()) / ((n - 1) * n)
    part2 = - 2 * c * norms.mean()
    part3 = c * c * k

    return part1 + part2 + part3


def neg_loss(x, c=1.0, reg=0.0, generalized=False): # derivation and modification
    """
    x: n * d.
    The formula shown in the paper
    """
    n = x.shape[0]
    d = x.shape[1]
    if not generalized:
        inprods = x @ x.T
        return _rep_loss(inprods, n, d, c, reg)

    tot_loss = 0.0
    # tot_loss = torch.tensor(0.0, device=x.device, requires_grad=True) # danger
    for k in range(1, d+1):
        inprods = x[:, :k] @ x[:, :k].T
        tot_loss += _rep_loss(inprods, n, k, c, reg)
    return tot_loss



class LapReprLearner:

    def __init__(self, common_args, env, replay_buffer: episodic_replay_buffer.EpisodicReplayBuffer):
        self.args = get_laprepr_args(common_args)
        self.env = env
        # NN
        if self.args.use_position_only:
            self._repr_fn = ReprNetMLP(self.args.obs_pos_dim, n_layers=self.args.lap_n_layers, n_units=self.args.lap_n_units, d=self.args.d)
        else:
            self._repr_fn = ReprNetMLP(self.args.obs_dim, n_layers=self.args.lap_n_layers, n_units=self.args.lap_n_units, d=self.args.d)
        self._repr_fn.to(device=self.args.device)
        # optimizer
        opt = getattr(optim, self.args.lap_opt_args_name)
        self._optimizer = opt(self._repr_fn.parameters(), lr=self.args.lap_opt_args_lr)
        # replay_buffer
        self._replay_buffer = replay_buffer

        self._global_step = 0
        self._train_info = collections.OrderedDict()

        # create ckpt save dir and log dir
        self.saver_dir = os.path.join(self.args.model_dir, self.args.lap_sub_dir)
        if not os.path.exists(self.saver_dir):
            os.makedirs(self.saver_dir)
        self.log_dir = os.path.join(self.args.log_dir, self.args.lap_sub_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

    def _collect_samples(self):
        # start actors, collect trajectories from random actions
        possible_actions = self.env.get_actions()
        print('Possible actions of the agent: ', possible_actions)
        action_idxs = np.arange(len(possible_actions))

        print('Start collecting samples.')
        timer = timer_tools.Timer()
        # collect initial transitions
        total_n_steps = 0
        collect_batch = 10000
        while total_n_steps < self.args.n_samples:
            self.env.reset(random_init=True) # random start points for the offline setting
            cur_state = self.env.get_curr_state()
            assert not cur_state.is_terminal
            epi_len = 0
            episode = []
            while (not cur_state.is_terminal) and (not self.env.over_episode_length(epi_len)):
                action = random.choice(action_idxs)
                rwd, next_state = self.env.execute_agent_action(action)
                # redundant info
                # transition = {'s': cur_state.features, 'a': action, 'r': rwd, 'next_s': next_state.features, 'done': next_state.is_terminal} # may be slow when save and get
                transition = {'s': cur_state, 'a': action, 'r': rwd, 'next_s': next_state, 'done': next_state.is_terminal}
                cur_state = next_state
                epi_len += 1
                episode.append(transition)
                # log
                total_n_steps += 1
                if (total_n_steps + 1) % collect_batch == 0:
                    print('({}/{}) steps collected.'.format(total_n_steps + 1, self.args.n_samples))
            final_transition = {'s': cur_state, 'a': random.choice(action_idxs), 'r': 0.0, 'next_s': cur_state, 'done': True} # not used to train the DQN
            episode.append(final_transition) # to make sure the last state in the episodes can be sampled in the future process
            self._replay_buffer.add_steps(episode)
        time_cost = timer.time_cost()
        print('Data collection finished, time cost: {}s'.format(time_cost))

    def train(self):
        self._collect_samples()
        # learning begins
        timer = timer_tools.Timer()
        timer.set_step(0)
        for step in range(self.args.lap_train_steps):
            assert step == self._global_step
            self._train_step()
            # save
            if (step + 1) % self.args.lap_save_freq == 0:
                saver_path = os.path.join(self.saver_dir, 'model_{}.ckpt'.format(step+1))
                torch.save(self._repr_fn.state_dict(), saver_path)
            # print info
            if step == 0 or (step + 1) % self.args.lap_print_freq == 0:
                steps_per_sec = timer.steps_per_sec(step)
                print('Training steps per second: {:.4g}.'.format(steps_per_sec))
                summary_str = summary_tools.get_summary_str(step=self._global_step, info=self._train_info)
                print(summary_str)
        # save the final laprepr model
        saver_path = os.path.join(self.saver_dir, 'final_model.ckpt')
        torch.save(self._repr_fn.state_dict(), saver_path)
        # log the time cost
        time_cost = timer.time_cost()
        print('Training finished, time cost {:.4g}s.'.format(time_cost))

    def _train_step(self):
        train_batch = self._get_train_batch()
        loss = self._build_loss(train_batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._global_step += 1

    def _get_train_batch(self): # how will the discount influence the performance?
        s1, s2 = self._replay_buffer.sample_steps(self.args.lap_batch_size, mode='pair', discount=self.args.lap_discount)
        s_neg, _ = self._replay_buffer.sample_steps(self.args.lap_batch_size, mode='single')
        s1, s2, s_neg = map(self._get_obs_batch, [s1, s2, s_neg])
        batch = {}
        batch['s1'] = self._tensor(s1)
        batch['s2'] = self._tensor(s2)
        batch['s_neg'] = self._tensor(s_neg)
        return batch

    def _build_loss(self, batch): # modification
        s1 = batch['s1']
        s2 = batch['s2']
        s_neg = batch['s_neg']
        s1_repr = self._repr_fn(s1)
        s2_repr = self._repr_fn(s2)
        s_neg_repr = self._repr_fn(s_neg)
        loss_positive = pos_loss(s1_repr, s2_repr)
        loss_negative = neg_loss(s_neg_repr, c=self.args.c_neg, reg=self.args.reg_neg, generalized=self.args.generalized)
        assert loss_positive.requires_grad and loss_negative.requires_grad # danger
        loss = loss_positive + self.args.w_neg * loss_negative
        info = self._train_info
        info['loss_pos'] = loss_positive.item()
        info['loss_neg'] = loss_negative.item()
        info['loss_total'] = loss.item()
        summary_tools.write_summary(self.writer, info=info, step=self._global_step)
        return loss

    def _get_obs_batch(self, steps): # which way is better for spectral clustering?
        if self.args.use_position_only:
            obs_batch = [s.get_position() for s in steps]
        else:
            obs_batch = [s.features() for s in steps]
        return np.stack(obs_batch, axis=0)

    def _tensor(self, x):
        return torch_tools.to_tensor(x, self.args.device)

    def get_all_embeddings(self, data_input, interval):
        obs_input = self._get_obs_batch(data_input)
        obs_input = self._tensor(obs_input) # maybe too much for the gpu?
        data_size = int(obs_input.shape[0])

        embeddings = []
        with torch.no_grad(): # danger
            cur_idx = 0
            while cur_idx < data_size:
                next_idx = min(cur_idx + interval, data_size)
                data_segment = obs_input[cur_idx:next_idx, :]
                embedding_segment = self._repr_fn(data_segment)
                embeddings = embeddings + embedding_segment.cpu().tolist()
                cur_idx = next_idx
        embeddings = np.array(embeddings)
        assert embeddings.shape[0] == data_size

        return embeddings

    def get_pair_embeddings(self, sample_num, interval):
        pair_embeddings = [[], []]
        with torch.no_grad(): # danger
            cur_idx = 0
            while cur_idx < sample_num:
                next_idx = min(cur_idx + interval, sample_num)
                s1, s2 = self._replay_buffer.sample_steps(next_idx-cur_idx, mode='pair', discount=self.args.lap_discount)
                s1, s2 = map(self._get_obs_batch, [s1, s2])
                s1, s2 = map(self._tensor, [s1, s2]) # danger
                s1_repr = self._repr_fn(s1)
                s2_repr = self._repr_fn(s2)
                pair_embeddings[0] += s1_repr.cpu().tolist()
                pair_embeddings[1] += s2_repr.cpu().tolist()
                cur_idx = next_idx
        pair_embeddings = np.array(pair_embeddings)
        assert pair_embeddings.shape[1] == sample_num

        return pair_embeddings







