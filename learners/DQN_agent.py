import os
import collections
import numpy as np
import random
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import get_rl_args
from learners.networks import DiscreteQNetMLP
from utils import episodic_replay_buffer, summary_tools, timer_tools, torch_tools


class DQNAgent(object):
    def __init__(self, env, args, sub_dir, noise_init=False):
        self.env = env
        self.args = get_rl_args(args)
        # networks
        self._q_fn_learning = DiscreteQNetMLP(input_shape=self.args.obs_dim, n_actions=self.args.act_dim, n_layers=self.args.rl_n_layers, n_units=self.args.rl_n_units)
        self._q_fn_target = DiscreteQNetMLP(input_shape=self.args.obs_dim, n_actions=self.args.act_dim, n_layers=self.args.rl_n_layers, n_units=self.args.rl_n_units)
        self._q_fn_learning.to(self.args.device)
        self._q_fn_target.to(self.args.device)
        self._q_fn_target.load_state_dict(self._q_fn_learning.state_dict())
        self._vars_learning = self._q_fn_learning.state_dict()
        self._vars_target = self._q_fn_target.state_dict()
        # optimizer
        opt = getattr(optim, self.args.rl_opt_args_name)
        self._optimizer = opt(self._q_fn_learning.parameters(), lr=self.args.rl_opt_args_lr)
        # replay buffer
        self._replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(max_size=self.args.rl_replay_buffer_size)
        # create ckpt save dir and log dir
        self.saver_dir = os.path.join(self.args.model_dir, sub_dir)
        if not os.path.exists(self.saver_dir):
            os.makedirs(self.saver_dir)
        self.log_dir = os.path.join(self.args.log_dir, sub_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        # others
        self.noise_init = noise_init # True, if we are building the options
        self._global_step = 0
        self._train_info = collections.OrderedDict()


    def _collect_samples(self, episode_num, reward_fn, is_random=False, is_evaluate=False):

        cumulated_rwds, cumulated_env_rwds, episode_lengths = [], [], []
        for _ in tqdm(range(episode_num)):
            self.env.reset(random_init=False, noise_init=self.noise_init) # if the dqn_agent is for option construction, then the start and end points for this option is specified, defined in the option constructer
            cur_state = self.env.get_curr_state()
            assert not cur_state.is_terminal
            epi_len = 0
            episode = []
            cumulated_rwd, env_rwd = 0.0, 0.0
            while (not cur_state.is_terminal) and (not self.env.over_episode_length(epi_len)):
                if is_random: # danger
                    action = random.choice(self.action_idxs)
                else:
                    if (not is_evaluate) and (np.random.uniform() <= self.args.epsilon_greedy):
                        action = random.choice(self.action_idxs)
                    else:
                        action = self._policy_fn(cur_state)

                rwd, next_state = self.env.execute_agent_action(action)
                env_rwd += rwd

                # if self.reward_mode != 'raw':
                rwd = reward_fn(rwd, next_state) # provided by the option constructer
                cumulated_rwd += rwd
                transition = {'s': cur_state, 'a': action, 'r': rwd, 'next_s': next_state, 'done': next_state.is_terminal}
                # cur_state = next_state
                cur_state = self.env.get_curr_state()
                assert cur_state == next_state # danger
                epi_len += 1
                episode.append(transition)

            cumulated_rwds.append(cumulated_rwd)
            cumulated_env_rwds.append(env_rwd)
            episode_lengths.append(epi_len)
            self._replay_buffer.add_steps(episode)

        return np.array(cumulated_rwds).mean(), np.array(cumulated_env_rwds).mean(), np.array(episode_lengths).mean()

    def train(self, reward_fn=None):
        # reward_fn: a function that takes the state as input and outputs the reward as a scalar, providing the intrinsic reward

        # start actors, collect trajectories from random actions
        print('Start collecting initial transitions.')
        timer = timer_tools.Timer()
        # start actors, collect trajectories from random actions
        possible_actions = self.env.get_actions()
        print('Possible actions of the agent: ', possible_actions)
        self.action_idxs = np.arange(len(possible_actions))
        avg_rwd, env_rwd, epi_len = self._collect_samples(self.args.rl_replay_buffer_init, reward_fn, is_random=True)
        time_cost = timer.time_cost()
        print('Initial data collection finished, average reward: {}, average environment reward: {}, episode length: {}, time cost: {}s'.format(avg_rwd, env_rwd, epi_len, time_cost))

        # learning begins
        timer.reset()
        timer.set_step(0)
        for step in range(self.args.rl_train_steps):
            assert step == self._global_step
            self._train_step()
            # interact with the environment
            if (step+1) % self.args.rl_replay_update_freq == 0:
                avg_rwd, env_rwd, epi_len = self._collect_samples(episode_num=self.args.rl_replay_update_num, reward_fn=reward_fn, is_random=False, is_evaluate=False)
                summary_tools.write_summary(self.writer, info={'episodic_reward': avg_rwd, 'episodic_reward_env': env_rwd, 'episode_length': epi_len}, step=step)
            # save
            if (step + 1) % self.args.rl_save_freq == 0:
                torch.save(self._q_fn_learning.state_dict(), os.path.join(self.saver_dir, 'q_model_{}.ckpt'.format(step + 1)))
                torch.save(self._q_fn_target.state_dict(), os.path.join(self.saver_dir, 'target_model_{}.ckpt'.format(step + 1)))
            # print info
            if step == 0 or (step + 1) % self.args.rl_print_freq == 0:
                steps_per_sec = timer.steps_per_sec(step)
                print('Training steps per second: {:.4g}.'.format(steps_per_sec))
                summary_str = summary_tools.get_summary_str(step=self._global_step, info=self._train_info)
                print(summary_str)
            # test agent
            if step == 0 or (step + 1) % self.args.rl_test_freq == 0:
                avg_rwd, env_rwd, epi_len = self._collect_samples(self.args.rl_test_episodes, reward_fn, is_random=False, is_evaluate=True)
                info = {'test_episodic_reward': avg_rwd, 'test_episodic_reward_env': env_rwd, 'test_episode_length': epi_len}
                summary_tools.write_summary(self.writer, info=info, step=step)
                summary_str = summary_tools.get_summary_str(step=self._global_step, info=info, prefix="Evaluation: ")
                print(summary_str)
        # save the final DQN model
        torch.save(self._q_fn_learning.state_dict(), os.path.join(self.saver_dir, 'q_final_model_{}.ckpt'))
        torch.save(self._q_fn_target.state_dict(), os.path.join(self.saver_dir, 'target_final_model_{}.ckpt'))
        # final evaluation
        avg_rwd, env_rwd, epi_len = self._collect_samples(self.args.rl_test_episodes, reward_fn, is_random=False, is_evaluate=True)
        final_performance = info = {'episodic_reward': avg_rwd, 'episodic_reward_env': env_rwd, 'episode_length': epi_len}
        # log the time cost
        time_cost = timer.time_cost()
        print('Training finished, time cost {:.4g}s.'.format(time_cost))

        return final_performance

    def _train_step(self):
        train_batch = self._get_train_batch()
        loss = self._build_loss(train_batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._global_step += 1
        if self._global_step % self.args.rl_update_freq == 0: # update the target network
            self._update_target_vars()

    def _get_train_batch(self):
        s1, a, s2, r, done = self._replay_buffer.sample_transitions(batch_size=self.args.rl_batch_size)
        s1, s2 = map(self._get_obs_batch, [s1, s2])
        done = np.array(done).astype(np.float32)
        dsc = (1.0 - done) * self.args.rl_discount # size: (batch_size, )

        batch = {}
        batch['s1'] = self._tensor(s1)
        batch['s2'] = self._tensor(s2)
        batch['a'] = self._tensor(a)
        batch['r'] = self._tensor(r)
        batch['dsc'] = self._tensor(dsc)
        return batch

    def _build_loss(self, batch):
        # modules and tensors
        s1, s2, a, r, dsc = batch['s1'], batch['s2'], batch['a'], batch['r'], batch['dsc']
        # networks
        q_vals_learning = self._q_fn_learning(s1)
        q_val_learning = q_vals_learning[torch.arange(self.args.rl_batch_size), a] # danger
        q_vals_target = self._q_fn_target(s2)
        val_target = q_vals_target.max(-1)[0] # size: (batch_size, )
        q_val_target = (r + dsc * val_target).detach()
        loss = (q_val_learning - q_val_target).pow(2).mean()
        # build print info
        info = self._train_info
        info['q_loss'] = loss.item()
        info['mean_q'] = q_val_target.mean().item()
        summary_tools.write_summary(self.writer, info=info, step=self._global_step)

        return loss

    def _tensor(self, x):
        return torch_tools.to_tensor(x, self.args.device)

    def _get_obs_batch(self, steps):
        obs_batch = [s.features() for s in steps]  # for dqn_agent, the input is in line with the info that the simulator provides
        return np.stack(obs_batch, axis=0)

    def _policy_fn(self, state):
        s = self._get_obs_batch([state])
        s = self._tensor(s)
        with torch.no_grad(): # danger
            q_vals = self._q_fn_learning(s).cpu().numpy()

        return np.argmax(q_vals[0])

    def _update_target_vars(self):
        # requires self._vars_learning and self._vars_target as state_dict`s
        for var_name, var_t in self._vars_target.items():
            updated_val = (self.args.rl_update_rate * self._vars_learning[var_name].data + (1.0 - self.args.rl_update_rate) * var_t.data)
            var_t.data.copy_(updated_val)

    def _load_model(self, sub_dir):
        model_dir = os.path.join(self.args.model_dir, sub_dir, 'q_final_model_{}.ckpt')
        self._q_fn_learning.load_state_dict(torch.load(model_dir))

    def _test_rollout(self, epi_len, env_rwd, sub_goal):
        cur_state = self.env.get_curr_state()
        while (not self.env.over_episode_length(epi_len)) and (not self.env.reach_subgoal(sub_goal)):
            action = self._policy_fn(cur_state)
            rwd, next_state = self.env.execute_agent_action(action)
            env_rwd += rwd
            cur_state = self.env.get_curr_state()
            assert cur_state == next_state  # danger
            epi_len += 1
        return epi_len, env_rwd, self.env.reach_subgoal(sub_goal)

    def evaluate(self, task_idx, start_point, shortest_path, goal_point, oc_agent):
        self.env.reset(random_init=False, noise_init=self.noise_init)
        cur_state = self.env.get_curr_state()
        assert cur_state == start_point # no noise
        assert not cur_state.is_terminal
        epi_len = 0
        env_rwd = 0.0
        # from start_point to the start_center
        start_center = oc_agent.idx_to_state(shortest_path[0])
        self._load_model(sub_dir=os.path.join(self.args.planner_sub_dir, 'task_{}'.format(task_idx), 'start_to_center'))
        epi_len, env_rwd, done = self._test_rollout(epi_len, env_rwd, sub_goal=start_center) # danger
        if not done:
            return env_rwd, epi_len, False
        # navigate along the shortest option path
        for i in range(len(shortest_path)-1):
            temp_goal_state = oc_agent.idx_to_state(shortest_path[i+1])
            self._load_model(sub_dir=os.path.join(self.args.opt_sub_dir, '{}_to_{}'.format(shortest_path[i], shortest_path[i+1])))
            epi_len, env_rwd, done = self._test_rollout(epi_len, env_rwd, sub_goal=temp_goal_state)
            if not done:
                return env_rwd, epi_len, False
        # navigate from the goal_center to the goal_point
        self._load_model(sub_dir=os.path.join(self.args.planner_sub_dir, 'task_{}'.format(task_idx), 'center_to_goal'))
        epi_len, env_rwd, done = self._test_rollout(epi_len, env_rwd, sub_goal=goal_point)
        if done:
            assert self.env.get_curr_state().is_terminal()
        return epi_len, env_rwd, done
