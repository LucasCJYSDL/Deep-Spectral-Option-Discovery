from configs import get_common_args
from learners import laprepr, spectral_cluster, option_map_constructor, offline_planner
from utils import timer_tools
from envs.simple_rl.tasks import PinballMDP
from utils import episodic_replay_buffer
import os
import torch
import time

def main():
    timer = timer_tools.Timer()
    args = get_common_args()
    temp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    args.model_dir = args.model_dir + '/' + temp_time
    args.log_dir = args.log_dir + '/' + temp_time

    if torch.cuda.is_available() and args.cuda:
        args.device = torch.device('cuda')
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        args.device = torch.device('cpu')
    print('device: {}.'.format(args.device))

    if args.env_id == 'Pinball':
        cases = {'medium':{'cfg': "pinball_medium.cfg", 'start': [(0.2, 0.9), (0.2, 0.5), (0.9, 0.9), (0.9, 0.5), (0.3, 0.3), (0.9, 0.1), (0.5, 0.1),
                                                                  (0.05, 0.05), (0.2, 0.05), (0.35, 0.65), (0.5, 0.3), (0.5, 0.45), (0.45, 0.6)],
                           'task_list': [[(0.2, 0.5), (0.9, 0.1)], [(0.9, 0.5), (0.45, 0.6)], [(0.9, 0.5), (0.35, 0.65)], [(0.2, 0.9), (0.5, 0.1)]]}}
        env = PinballMDP(render=args.render, episode_length=1000, reward_scale=1000., cfg=cases['medium']['cfg'], start_points=cases['medium']['start'],
                         task_list=cases['medium']['task_list'])
    else:
        raise NotImplementedError

    env_info = env.get_env_info()
    args.obs_dim = env_info['obs_dim']
    args.obs_pos_dim = env_info['obs_pos_dim']
    args.act_dim = env_info['act_dim']

    # learn the eigenfunctions
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(max_size=args.lap_replay_buffer_size) # shared
    learner = laprepr.LapReprLearner(args, env, replay_buffer)
    learner.train()
    # spectral clustering with the learned estimations
    sc_agent = spectral_cluster.SpectralCluster(args, learner, replay_buffer)
    centers, sub_goals = sc_agent.clustering() # centers: list of States, sub_goals: dict
    # option construction with the newly found landmarks
    oc_agent = option_map_constructor.OptionConstructor(env, args, centers, sub_goals, learner)
    oc_agent.build_topo_map()
    # offline planning based on the topological map
    op_agent = offline_planner.OfflinePlanner(args, env, oc_agent, sc_agent, learner)
    op_agent.planning(task_list=env.get_task_list())
    print('Total time cost: {:.4g}s.'.format(timer.time_cost()))


if __name__ == '__main__':
    main()
