import numpy as np
import skfuzzy as fuzz

from utils import timer_tools, episodic_replay_buffer
from learners.laprepr import LapReprLearner
from configs import get_clustering_args

class SpectralCluster(object):

    def __init__(self, args, laprepr: LapReprLearner, replay_buffer: episodic_replay_buffer.EpisodicReplayBuffer):

        self.args = get_clustering_args(args)
        self.laprepr = laprepr
        self.replay_buffer = replay_buffer

    def _get_embeddings(self):
        print('Start collecting embeddings.')
        timer = timer_tools.Timer()
        self._raw_states = self.replay_buffer.get_all_steps(max_num=self.args.sc_n_samples) # the index of the raw_states and _embeddings should be the same
        self._embeddings = self.laprepr.get_all_embeddings(self._raw_states, interval=self.args.inerval) # np.ndarray
        self._sample_num = self._embeddings.shape[0]
        time_cost = timer.time_cost()
        print('Embeddings collection finished, time cost: {}s'.format(time_cost))

    def _get_eigenvalues(self): # important and dangerous; time complexity: d|S| # check!
        print('Start estimating eigenvalues.')
        timer = timer_tools.Timer()

        self.eigenvalue_list = []
        d_max = self.laprepr.args.d
        pair_embeddings = self.laprepr.get_pair_embeddings(sample_num=self.args.ev_n_samples, interval=self.args.inerval) # np.ndarray: [2, |S|, d]
        assert pair_embeddings.shape[2] == d_max
        # This is wrong since the eigenvectors are not in ascending order
        # if not self.laprepr.args.generalized: # no generalized weights
        #     last_value = 0.0 # sum of the first k - 1 eigenvalues
        #     for k in range(d_max):
        #         # danger
        #         cur_value = 0.5 * np.square(pair_embeddings[0][:, :(k+1)] - pair_embeddings[1][:, :(k+1)]).sum(-1).mean() # sum of the first k eigenvalues
        #         k_value = cur_value - last_value
        #         last_value = cur_value
        #         self.eigenvalue_list.append(k_value)
        # else: # with generalized weights
        for k in range(d_max):
            # danger
            k_value = 0.5 * np.square(pair_embeddings[0][:, k] - pair_embeddings[1][:, k]).mean()
            self.eigenvalue_list.append(k_value)

        time_cost = timer.time_cost()
        print('Eigenvalues estimating finished, time cost: {}s, generalized: {}.'.format(time_cost, self.laprepr.args.generalized))

    def clustering(self):
        ## determine the number of clusters according to the eigenvalue gap
        ## number of clusters should be larger than 1
        self._get_eigenvalues()
        d_max = self.laprepr.args.d
        assert d_max > 1
        ev_gaps = []
        for i in range(1, d_max):
            ev_gaps.append(self.eigenvalue_list[i] - self.eigenvalue_list[i-1])

        # for real eigenvalues, the elements in ev_gaps should be non-negative!
        best_k = 0 # cluster numbers
        for i in range(d_max-1):
            if ev_gaps[i] > self.args.gap_threshold:
                best_k = i + 1 # danger
                break
        if best_k == 0:
            best_k = d_max
        print("Eigenvalues: {}, possible cluster number: {}.".format(self.eigenvalue_list, best_k))

        ## cluster the embeddings
        self._get_embeddings()
        print("Start fuzzy c-means.")
        timer = timer_tools.Timer()
        centers = None
        likelihoods = None
        best_fpc = -1
        real_k = None
        for k in range(best_k-self.args.km_range, best_k+self.args.km_range+1):
            if k <= 1 or k > d_max:
                continue
            timer.reset()

            first_K_embeddings = self._embeddings[:, :(k+1)]
            if self.args.sc_normlized:
                first_K_embeddings = first_K_embeddings / np.linalg.norm(first_K_embeddings, axis=1, keepdims=True)

            temp_centers, temp_likelihoods, _, _, _, tot_iters, fpc = fuzz.cmeans(data=first_K_embeddings.transpose(), c=k, m=self.args.km_m,
                                                                                  error=self.args.km_error, maxiter=self.args.km_max_iters)
            if fpc > best_fpc:
                centers = temp_centers # size: (cluster_num, feature_num), should they be equal?
                likelihoods = temp_likelihoods # size: (cluster_num, sample_num)
                best_fpc = fpc
                real_k = k

            time_cost = timer.time_cost()
            print("Fuzzy c-means finished with k: {}, fpc: {}, time cost: {}s, total iterations: {}.".format(k, fpc, time_cost, tot_iters))

        assert best_fpc > 0
        print("Fuzzy c-means end, with the cluster number: {}, best partition score: {}.".format(real_k, best_fpc))
        # hard_membership = np.argmax(likelihoods, axis=0) # size: (sample_num)
        self.real_k = real_k
        self.centers = centers # save for later prediction use
        assert self.real_k == self.centers.shape[0] == self.centers.shape[1], self.centers.shape

        # find the centers of the clusters, by comparing the embeddings, any better ideas?
        print("Start finding centers.")
        timer.reset()
        state_centers = []
        embedding_list = self._embeddings[:, :(real_k+1)]
        for ctr in range(self.real_k):
            center_vec = np.array(centers[ctr])
            dist_vec = np.square(embedding_list-center_vec).sum(-1)
            min_idx = np.argmin(dist_vec)
            state_centers.append(self._raw_states[min_idx])
        print("Found the center states, with time cost: {}.".format(timer.time_cost()))

        # find the sub-goals
        print("Start finding sub-goals.")
        timer.reset()
        likelihoods = np.array(likelihoods).transpose() # size: (sample_num, cluster_num), the rows are already normalized
        max_likelihoods = np.max(likelihoods, axis=1) # size: (sample_num, )
        sub_goal_dict = {}
        for idx in range(self._sample_num):
            temp_list = []
            for ctr in range(self.real_k):
                diff = max_likelihoods[idx] - likelihoods[idx][ctr]
                if diff < self.args.sub_goal_threshold:
                    temp_list.append(ctr)
            if len(temp_list) > 1: # possible sub-goals
                try:
                    sub_goal_dict[self._raw_states[idx]] = temp_list
                except:
                    print("The same sub-goal has been added!")
        print("Total number of possible sub-goals: {}.".format(len(sub_goal_dict)))

        sub_goal_cluster_dict = {}
        for i in range(self.real_k - 1):
            for j in range(i+1, self.real_k):
                sub_goal_cluster_dict[(i, j)] = []
                # list_dict = list(sub_goal_dict.items())
                # np.random.shuffle(list_dict)
                # sub_goal_dict = dict(list_dict)
                for key, value in sub_goal_dict.items():
                    if (i in value) and (j in value):
                        sub_goal_cluster_dict[(i, j)].append(key)
                        if len(sub_goal_cluster_dict[(i, j)]) > self.args.sub_goal_num_limit:
                            break
        print("Found the sub-goals, with time cost: {}.".format(timer.time_cost()))

        return state_centers, sub_goal_cluster_dict

    def get_cluster_memberships(self, task_list):

        state_list = []
        for task in task_list:
            state_list.append(task[0])
            state_list.append(task[1])
        embeddings = self.laprepr.get_all_embeddings(data_input=state_list, interval=self.args.inerval)[:, :(self.real_k+1)]
        u, u0, d, jm, p, fpc = fuzz.cmeans_predict(embeddings.transpose(), cntr_trained=self.centers, m=self.args.km_m, error=self.args.km_error, maxiter=self.args.km_max_iters)
        cluster_membership = np.argmax(u, axis=0)
        cluster_membership_list = []
        assert len(cluster_membership)%2 == 0
        for idx in range(0, len(cluster_membership), 2):
            cluster_membership_list.append((cluster_membership[idx], cluster_membership[idx+1]))

        return cluster_membership_list




