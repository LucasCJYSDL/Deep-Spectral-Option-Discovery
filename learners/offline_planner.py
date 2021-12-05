import os
import numpy as np
import networkx as nx

from learners import option_map_constructor, spectral_cluster, DQN_agent, laprepr
from configs import get_planner_args

class OfflinePlanner(object):

    def __init__(self, args, env, oc_agent: option_map_constructor.OptionConstructor, sc_agent: spectral_cluster.SpectralCluster, lr_agent: laprepr.LapReprLearner):
        self.args = get_planner_args(args)
        self._reward_mode = self.args.planner_reward_mode
        self.env = env
        self.oc_agent = oc_agent
        self.sc_agent = sc_agent
        self.laprepr = lr_agent

    def _reward_fn(self, raw_rwd, next_state): # 'lap', 'raw', 'lap_mix': 'lap' + 'raw'

        if self.args.planner_sparse_reward:
            if next_state.is_terminal():
                raw_rwd = 0.0
            else:
                raw_rwd = -1.0
        # 'raw'
        if self._reward_mode == 'raw':
            return raw_rwd
        # 'lap', 'lap_mix'
        # compute embedding distance
        embeddings = self.laprepr.get_all_embeddings(data_input=[next_state, self._goal_point], interval=2)
        if not self.args.planner_use_fiedler_only:
            dist = np.linalg.norm(embeddings[0] - embeddings[1])
        else:
            # only use the second eigenvector as the embedding
            dist = np.absolute(embeddings[0][1] - embeddings[1][1])

        if 'mix' in self._reward_mode:
            # mix the embedding distance with the original reward
            r = - dist * self.args.planner_dist_reward_coeff * 0.5 + raw_rwd * 0.5
        else:
            r = - dist * self.args.planner_dist_reward_coeff

        return r

    def planning(self, task_list):
        cluster_membership_list = self.sc_agent.get_cluster_memberships(task_list)
        assert len(task_list) == len(cluster_membership_list)
        task_num = len(task_list)
        for task_idx in range(task_num):
            start_state, goal_state = task_list[task_idx]
            start_cluster, goal_cluster = cluster_membership_list[task_idx]
            if start_cluster == goal_cluster:
                # since they are in the same cluster, we can directly train the policy to navigate from the start to the goal using DQN
                sub_dir = os.path.join(self.args.planner_sub_dir, 'task_{}'.format(task_idx), 'start_to_goal')
                self.env.set_start_and_goal(s=start_state, g=goal_state)
                self._goal_point = goal_state
                rl_agent = DQN_agent.DQNAgent(self.env, self.args, sub_dir, noise_init=False) # the start point will not change, so we don't need to add noise when training
                metrics = rl_agent.train(reward_fn=self._reward_fn)
                print("For task #{}, the agent can directly go from the start point to the end point, with performance metrics: {}.".format(task_idx, metrics))
            else:
                start_center = self.oc_agent.centers[start_cluster]
                goal_center = self.oc_agent.centers[goal_cluster]
                # the rl_agent from start_point to the start_center
                s_sub_dir = os.path.join(self.args.planner_sub_dir, 'task_{}'.format(task_idx), 'start_to_center')
                self.env.set_start_and_goal(s=start_state, g=start_center)
                self._goal_point = start_center
                s_rl_agent = DQN_agent.DQNAgent(self.env, self.args, s_sub_dir, noise_init=False)
                s_metrics = s_rl_agent.train(reward_fn=self._reward_fn)
                print("For task #{}, the training performance from the start point to the start center: {}.".format(task_idx, s_metrics))
                # the rl_agent from the goal_center to the goal_state
                g_sub_dir = os.path.join(self.args.planner_sub_dir, 'task_{}'.format(task_idx), 'center_to_goal')
                self.env.set_start_and_goal(s=goal_center, g=goal_state)
                self._goal_point = goal_state
                g_rl_agent = DQN_agent.DQNAgent(self.env, self.args, g_sub_dir, noise_init=True) # danger
                g_metrics = g_rl_agent.train(reward_fn=self._reward_fn)
                print("The training performance from the goal center to the goal state: {}.".format(g_metrics))
                # get the shortest option path between the start center and goal center based on topological map
                shortest_path = nx.dijkstra_path(self.oc_agent.G, source=start_cluster, target=goal_cluster)
                metrics = self._get_overall_performance(task_idx, start_state, shortest_path, goal_state)
                print("The test performance: {}.".format(metrics))

    def _get_overall_performance(self, task_idx, start_point, shortest_path, goal_point):
        env_rwds, lengths, dones = [], [], []
        self.env.set_start_and_goal(s=start_point, g=goal_point)
        test_agent = DQN_agent.DQNAgent(self.env, self.args, sub_dir=os.path.join(self.args.planner_sub_dir, 'task_{}'.format(task_idx), 'test'), noise_init=False)
        for epi in self.args.planner_test_episodes:
            env_rwd, length, done = test_agent.evaluate(task_idx, start_point, shortest_path, goal_point, self.oc_agent)
            env_rwds.append(env_rwd)
            lengths.append(length)
            if done:
                dones.append(1.0)
            else:
                dones.append(0.0)
            print("The test episode #{} of task #{} is {}, the episode length: {} and the reward: {}.".format(epi, task_idx, done, length, env_rwd))

        return {"Environment Reward": np.array(env_rwds).mean(), "Episode Length": np.array(lengths).mean(), "Success Rate": np.array(dones).mean()}