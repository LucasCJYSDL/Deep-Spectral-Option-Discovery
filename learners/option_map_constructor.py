import os
import numpy as np
import random
import networkx as nx

from utils.timer_tools import Timer
from learners.DQN_agent import DQNAgent
from learners.laprepr import LapReprLearner
from configs import get_option_args


class OptionConstructor(object):

    def __init__(self, env, args, centers, subgoals, laprepr: LapReprLearner):
        self.env = env
        self.args = args
        self.opt_args = get_option_args(args)
        self._reward_mode = self.opt_args.opt_reward_mode

        self.centers = centers
        self.subgoals = subgoals
        self.laprepr = laprepr

    def build_topo_map(self):
        '''
        How to use this topo map: when getting a new (start, goal) pair, we need to first identify the corresponding cluster center pair;
                                  then we need to find the shortest path between the two centers based on this topo map; (so the weights of the edges should be the avarage steps required)
                                  when navigating through the shortest path, the agent should move between the center and the subgoal according to the options
        '''
        ## build the graph represented as adjacency matrix
        center_num = len(self.centers)
        self.idx_to_node = {} # key: node index in the graph, value: center or sub-goal (pair of centers)
        self.node_to_idx = {} # opposite to the above one
        # add centers to the graph
        for ctr in range(center_num):
            self.idx_to_node[ctr] = ctr
            self.node_to_idx[ctr] = ctr
        # choose one possible sub-goal randomly, if exists; then add them to the graph
        self._subgoals = {}
        cur_idx = center_num
        for i in range(center_num - 1):
            for j in range(i+1, center_num):
                if len(self.subgoals[(i, j)]) > 0:
                    self._subgoals[(i, j)] = random.choice(self.subgoals[(i, j)])
                    self.idx_to_node[cur_idx] = (i,j)
                    self.node_to_idx[(i,j)] = cur_idx
                    cur_idx += 1
        assert (cur_idx - center_num) == len(self._subgoals)
        print("There are in total {} centers, and {} sub-goals!".format(center_num, len(self._subgoals)))

        ## connecting the landmarks with options and build the adjacency matrix accordingly
        self.topo_map = np.zeros((cur_idx, cur_idx))

        for ctr_pair in self._subgoals.keys():
            # get the State based on the index
            i, j = ctr_pair
            k = self.node_to_idx[(i, j)]
            center_i = self.centers[i]
            center_j = self.centers[j]
            sub_goal = self._subgoals[ctr_pair]
            # center_i -> sub_goal
            print("Start building the option between the center {} and the sub_goal {}.".format(i, ctr_pair))
            metrics = self.construct_options(center_i, sub_goal, sub_dir='{}_to_{}'.format(i, k))
            self.topo_map[i][k] = metrics['episode_length']
            # sub_goal -> center_i
            print("Start building the option between the sub_goal {} and the center {}.".format(ctr_pair, i))
            metrics = self.construct_options(sub_goal, center_i, sub_dir='{}_to_{}'.format(k, i))
            self.topo_map[k][i] = metrics['episode_length']
            # center_j -> sub_goal
            print("Start building the option between the center {} and the sub_goal {}.".format(j, ctr_pair))
            metrics = self.construct_options(center_j, sub_goal, sub_dir='{}_to_{}'.format(j, k))
            self.topo_map[j][k] = metrics['episode_length']
            # sub_goal -> center_j
            print("Start building the option between the sub_goal {} and the center {}.".format(ctr_pair, j))
            metrics = self.construct_options(sub_goal, center_j, sub_dir='{}_to_{}'.format(k, j))
            self.topo_map[k][j] = metrics['episode_length']

        ## check the connectivity of the topo_map; if not, make it connected
        print("Checking the connectivity of the initial topological map.")
        self._check_connectivity()
        # build the networkx topological map based on self.topo_map
        self._build_networkx_topo_map()

    def construct_options(self, start_point, goal_point, sub_dir):
        self.env.set_start_and_goal(s=start_point, g=goal_point) # danger: goal_point.is_terminal
        self._goal_point = goal_point
        sub_dir = os.path.join(self.opt_args.opt_sub_dir, sub_dir)
        rl_agent = DQNAgent(env=self.env, args=self.args, sub_dir=sub_dir, noise_init=True)
        agent_performance = rl_agent.train(reward_fn=self._reward_fn) # danger

        return agent_performance

    def _reward_fn(self, raw_rwd, next_state): # 'l2', 'lap', 'raw', 'l2_mix': 'l2' + 'raw', 'lap_mix': 'lap' + 'raw'

        if self.opt_args.opt_sparse_reward:
            if next_state.is_terminal():
                raw_rwd = 0.0
            else:
                raw_rwd = -1.0
        # 'raw'
        if self._reward_mode == 'raw':
            return raw_rwd
        # 'l2', 'lap', 'l2_mix', 'lap_mix'
        dist = 0.0
        if 'l2' in self._reward_mode:
            # compute l2 distance between s2 and sg
            dist = np.linalg.norm(next_state.get_position() - self._goal_point.get_position())
        elif 'lap' in self._reward_mode:
            # compute embedding distance
            embeddings = self.laprepr.get_all_embeddings(data_input=[next_state, self._goal_point], interval=2)
            if not self.opt_args.opt_use_fiedler_only:
                dist = np.linalg.norm(embeddings[0] - embeddings[1])
            else:
                # only use the second eigenvector as the embedding
                dist = np.absolute(embeddings[0][1] - embeddings[1][1])

        if 'mix' in self._reward_mode:
            # mix the embedding distance with the original reward
            r = - dist * self.opt_args.opt_dist_reward_coeff * 0.5 + raw_rwd * 0.5
        else:
            r = - dist * self.opt_args.opt_dist_reward_coeff

        return r

    def _check_connectivity(self):
        # build the topo map with networkx
        G = nx.Graph() # undirected graph
        for n_id in self.idx_to_node.keys():
            G.add_node(n_id)
        node_num = len(G.nodes())
        assert node_num == self.topo_map.shape[0] == self.topo_map.shape[1]
        for i in range(node_num):
            for j in range(node_num):
                if self.topo_map[i][j] > 0:
                    assert i != j
                    G.add_edge(i, j)
        # collect the connected components
        connected_components = []
        for c in nx.connected_components(G):
            sub_node_list = G.subgraph(c).nodes()
            connected_components.append(sub_node_list)
        component_num = len(connected_components)
        if component_num == 1:
            print("Perfect! This is already a connected topological map!")
        else:
            print("The components that need to be connected are listed as: {}.".format(connected_components))
            timer = Timer()
            # connect the components with options
            # abstract the connected components as nodes, build a fully connected graph between these abstracted nodes and then search for the minimum spanning tree
            # the weight for the edge is the minimum distance between the two connected components
            cluster_G = nx.Graph()
            for c in range(component_num):
                cluster_G.add_node(c)
            cluster_edge_dict = {}
            for i in range(component_num-1):
                for j in range(i+1, component_num):
                    node_pair, node_dist = self._get_closest_pair(connected_components[i], connected_components[j])
                    cluster_edge_dict[(i,j)] = node_pair
                    cluster_G.add_edge(i, j, weight=node_dist)
            mst = nx.algorithms.tree.minimum_spanning_edges(cluster_G, data=False)
            edge_list = list(mst)
            # based on the minimum spanning tree, add edges to the topo_map
            for edge in edge_list:
                pair = tuple(sorted(edge))
                node_i, node_j = cluster_edge_dict[pair]
                print("Connecting Components {} with an option between Nodes {} and {}.".format(pair, node_i, node_j))
                state_i, state_j = map(self.idx_to_state, [node_i, node_j])
                metrics = self.construct_options(state_i, state_j, sub_dir='{}_to_{}'.format(node_i, node_j))
                assert self.topo_map[node_i][node_j] == 0.0
                self.topo_map[node_i][node_j] = metrics['episode_length']
                metrics = self.construct_options(state_j, state_i, sub_dir='{}_to_{}'.format(node_j, node_i))
                assert self.topo_map[node_j][node_i] == 0.0
                self.topo_map[node_j][node_i] = metrics['episode_length']
                # update the whole networkx graph
                G.add_edge(node_i, node_j)
            assert nx.is_connected(G)
            print("You have got a connected topological map, with extra time cost {}.".format(timer.time_cost()))


    def _get_closest_pair(self, node_list_i, node_list_j):
        min_dist = float('inf')
        min_pair = None
        for node_i in range(node_list_i):
            for node_j in range(node_list_j):
                state_i = self.idx_to_state(node_i)
                state_j = self.idx_to_state(node_j)
                embeddings = self.laprepr.get_all_embeddings(data_input=[state_i, state_j], interval=2)
                if not self.opt_args.opt_use_fiedler_only:
                    dist = np.linalg.norm(embeddings[0] - embeddings[1])
                else:
                    dist = np.absolute(embeddings[0][1] - embeddings[1][1])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (node_i, node_j)
        return min_pair, min_dist


    def idx_to_state(self, node_idx):
        node = self.idx_to_node[node_idx]
        if isinstance(node, tuple):
            return self._subgoals[node]
        else:
            return self.centers[node]


    def _build_networkx_topo_map(self):
        self.G = nx.DiGraph() # directed graph
        for n_id in self.idx_to_node.keys():
            self.G.add_node(n_id)
        node_num = len(self.G.nodes())
        assert node_num == self.topo_map.shape[0] == self.topo_map.shape[1]
        for i in range(node_num):
            for j in range(node_num):
                if self.topo_map[i][j] > 0:
                    assert i != j
                    self.G.add_edge(i, j, weight=self.topo_map[i][j])


