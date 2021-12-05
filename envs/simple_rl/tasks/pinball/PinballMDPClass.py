# Python Imports.
from __future__ import print_function
import copy
import numpy as np
import os
import time
# Other imports.
from rlpy.Domains.Pinball import Pinball
from envs.simple_rl.mdp.MDPClass import MDP
from envs.simple_rl.tasks.pinball.PinballStateClass import PinballState
import time
from rlpy.Tools import __rlpy_location__

class PinballMDP(MDP):
    """ Class for pinball domain. """

    def __init__(self, noise=0., episode_length=1000, reward_scale=1000., cfg="pinball_medium.cfg", start_points=None, task_list=None, render=False):
        default_config_dir = os.path.dirname(__file__)
        self.cfg = cfg
        self.start_points = start_points
        self.task_list = task_list
        self.domain = Pinball(noise=noise, episodeCap=episode_length, configuration=os.path.join(default_config_dir, "PinballConfigs", self.cfg))
        self.render = render
        self.reward_scale = reward_scale # will there be any influence for the rl agent?
        self.episode_length = episode_length
        self.region_rad = self.domain.environment.target_rad # used for both the start point and goal point

        # Each observation from domain.step(action) is a tuple of the form reward, next_state, is_term, possible_actions
        # s0 returns initial state, is_terminal, possible_actions: all the actions
        init_observation = self.domain.s0()
        init_state = tuple(init_observation[0])
        actions = self.domain.actions # danger
        self.act_dim = len(actions)

        MDP.__init__(self, actions, self._transition_func, self._reward_func, init_state=PinballState(*init_state))

    def reset(self, random_init=False, noise_init=False):
        if not random_init:
            init_observation = self.domain.s0()
            init_state = init_observation[0]
            if noise_init:
                init_state[0] = random.uniform(init_state[0]-self.region_rad, init_state[0]+self.region_rad) # a square rather than a circle
                init_state[1] = random.uniform(init_state[1]-self.region_rad, init_state[1]+self.region_rad)
            init_state = tuple(init_state)

        else:
            assert self.start_points is not None
            start_point = random.choice(self.start_points)
            init_state = (start_point[0], start_point[1], 0.0, 0.0) # the initial speed is set as 0

        self.init_state = PinballState(*init_state)
        cur_state = copy.deepcopy(self.init_state)
        self._set_current_state(cur_state)

    def _set_current_state(self, new_state):
        self.cur_state = new_state
        self.domain.state = new_state.features()

    def set_start_and_goal(self, s: PinballState, g: PinballState): # danger
        self.domain.environment.start_pos = s.get_position()
        self.domain.environment.target_pos = g.get_position()

    def get_env_info(self):
        env_info = {}
        env_info['obs_dim'] = len(self.init_state.features())
        env_info['obs_pos_dim'] = len(self.init_state.get_position())
        env_info['act_dim'] = self.act_dim
        return env_info

    def get_task_list(self):
        task_list = []
        for task in self.task_list:
            start_state = PinballState(x=task[0][0], y=task[0][1], xdot=0.0, ydot=0.0, is_terminal=False)
            goal_state = PinballState(x=task[1][0], y=task[1][1], xdot=0.0, ydot=0.0, is_terminal=True)
            task_list.append((start_state, goal_state))
        return task_list

    def _reward_func(self, state, action):
        """
        Slanting the plate costs -4 reward in addition to -1 reward for each timestep.
        When the ball reaches the hole, the agent receives 10000 units of reward.
        Args:
            state (PinballState)
            action (str): number between 0 and 4 inclusive: [ACC_X, DEC_Y, DEC_X, ACC_Y, ACC_NONE]
        Returns:
            scaled reward
        """
        # assert self.is_primitive_action(action), "Can only implement primitive actions to the MDP"
        reward, obs, done, possible_actions = self.domain.step(action)

        if self.render: # visualization
            self.domain.showDomain(action)

        self.next_state = PinballState(*tuple(obs), is_terminal=done)

        assert done == self.is_goal_state(self.next_state), "done = {}, s' = {} should match".format(done, self.next_state)

        # negatively_clamped_reward = -1. if reward < 0 else reward
        return reward / self.reward_scale

    def _transition_func(self, state, action): # must be called after self._reward_func(state, action)
        return self.next_state

    def execute_agent_action(self, action_idx):
        """
        Args:
            action_idx (int)
        Returns:
            (tuple: <float, State>): reward, State
        """
        action = self.actions[action_idx]
        reward = self.reward_func(self.cur_state, action)
        next_state = self.transition_func(self.cur_state, action)
        self.cur_state = next_state

        return reward, next_state

    def reach_subgoal(self, subgoal):
        return (np.linalg.norm(self.cur_state.get_position() - subgoal.get_position()) < self.region_rad)

    def over_episode_length(self, cur_length):
        return cur_length >= self.episode_length

    def is_goal_state(self, state):
        """
        We will pass a reference to the PinballModel function that indicates
        when the ball hits its target.
        Returns:
            is_goal (bool)
        """
        target_pos = np.array(self.domain.environment.target_pos)
        target_rad = self.domain.environment.target_rad
        return np.linalg.norm(state.get_position() - target_pos) < target_rad

    def bounds(self):
        # Low and then high
        low_bound = np.asarray([0.0, 0.0, -2.0, -2.0])
        up_bound = np.asarray([1.0, 1.0, 2.0, 2.0])
        return low_bound, up_bound

    # This visualization is effective or not?
    # def draw_state(self, s, filename=None):
    #     draw_trajectory([s], filename=filename)
    # 
    # def draw_trajectory(self, traj, filename=None):
    #     assert(isinstance(traj, list))
    #     # 1. set up canvas of size 500x500.
    #     width = 1000
    #     height = 1000
    #     screen = pygame.Surface((width, height))
    #     screen.fill((255, 255, 255))
    # 
    #     # 2. Render the obstacle positions.
    #     for obs in self.domain.environment.obstacles:
    #         point_list = obs.points
    #         plist = []
    #         for p in point_list:
    #             pair = (int(p[0] * width), int(p[1] * height))
    #             plist.append(pair)
    #         # print('point_list=', point_list)
    #         # TODO: Normalize to the width/height
    #         pygame.draw.polygon(screen, (0, 0, 0), plist, 0)    
    #     
    #     # 3. Render the trajectories
    #     # print('traj=', traj)
    #     lines = []
    #     for i in range(len(traj)):
    #         lines.append((int(width * traj[i].x), int(height * traj[i].y)))
    #     pygame.draw.lines(screen, (46, 224, 49), False, lines, 15)
    #     # print('lines=', lines)
    #     # ball_pos = (int(s[0] * width), int(s[1] * height))
    #     pygame.draw.circle(screen, (46, 224, 224), lines[0], 15)
    #     pygame.draw.circle(screen, (224, 145, 157), lines[-1], 15)
    # 
    #     # 4. Render the goal position
    #     # target_pos = self.domain.environment.target_pos
    #     # tpos = (int(target_pos[0] * width), int(target_pos[1] * height))
    #     # TODO: Normalize to the width/height
    #     # pygame.draw.circle(screen, (224, 0, 0), tpos, 15)
    # 
    #     flipped_scr = pygame.transform.flip(screen, False, True)
    #     # 5. Print to file.
    #     if filename is None:
    #         pygame.image.save(flipped_scr, './pinball_state.png')
    #     else:
    #         pygame.image.save(flipped_scr, filename)

    @staticmethod
    def is_primitive_action(action):
        assert action >= 0, "Action cannot be negative {}".format(action)
        return action < 5



if __name__ == '__main__':
    import random

    cases = {'medium': {'cfg': "pinball_medium.cfg", 'start': [(0.2, 0.9), (0.2, 0.5), (0.9, 0.9), (0.9, 0.5), (0.3, 0.3), (0.9, 0.1), (0.5, 0.1),
                                                               (0.05, 0.05), (0.2, 0.05), (0.35, 0.65), (0.5, 0.3), (0.5, 0.45), (0.45, 0.6)]}}
    test_env = PinballMDP(render=True, cfg=cases['medium']['cfg'], start_points=cases['medium']['start'])
    # print(test_env.get_actions())
    test_env.set_start_and_goal(s=PinballState(x=0.5, y=0.9, xdot=0.0, ydot=0.0, is_terminal=False), g=PinballState(x=0.9, y=0.1, xdot=0.0, ydot=0.0, is_terminal=True))
    for _ in range(10):
        # test_env.reset(random_init=True)
        # test_env.reset(random_init=False, noise_init=False)
        test_env.reset(random_init=False, noise_init=True)
        print("initial state: ", test_env.get_curr_state())
        print("init env state: ", test_env.domain.state)
        for i in range(100):
            time.sleep(0.02)
            action = random.choice([0, 1, 2, 3, 4])
            # action = 0 # acc_x -> right
            # action = 1 # dec_y -> up
            # action = 2 # dex_x -> left
            # action = 3 # acc_y -> down
            # action = 4 # none
            print("action: ", action)
            rwd, state = test_env.execute_agent_action(action)
            print("reward: ", rwd)
            print("state: ", state)
            print("env state: ", test_env.domain.state)
            if state.is_terminal():
                break