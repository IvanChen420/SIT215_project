# -*- coding: utf-8 -*-

import random
import gym
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 12

""" Document 
    Part I:  Reinforcement Learning Agent
    Part II: Tool Functions
    Part III:Runner
"""


""" Part I: Reinforcement Learning Agent
    1. Agent interface
    2. Random policy control: RandomPolicyAgent
    3. Q-learning control (off-policy): QLearningAgent
    4. S-A-R-S-A control (on-policy): SARSAgent
"""

discrete_factor = 10


class Agent:

    def __init__(self, env_id):

        self.env_id = env_id
        self.env = gym.make(env_id)  # "Taxi-v3" "CartPole-v1" "Pendulum-v0"

        """ Three different ways to initialize the q-value table. """
        if env_id == "Taxi-v3":  # Both observation space and action space are discrete.
            self.q_table = np.zeros([self.env.observation_space.n,  # Num of State
                                     self.env.action_space.n])  # Num of Action
        elif env_id == "CartPole-v1":  # Observation space is continuous, action space are discrete.
            discrete_factor = 10
            self.q_table = np.zeros([discrete_factor ** self.env.observation_space.shape[0],  # Num of State
                                     self.env.action_space.n])  # Num of Action
        elif env_id == "Pendulum-v0":  # Both observation space and action space are discrete.
            discrete_factor = 10
            self.q_table = np.zeros([discrete_factor ** self.env.observation_space.shape[0],  # Num of State
                                     discrete_factor ** self.env.action_space.shape[0]])  # Num of Action

        # Hyper-parameters
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1

    def _epsilon_greedy_select(self, _state):
        if self.env_id in ["CartPole-v1", "Pendulum-v0"]:
            _state = discrete_state_helper(_state, self.env_id)

        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(self.q_table[_state])  # Exploit learned values

        if self.env_id == "Pendulum-v0":  # The Pendulum action needs a list type
            return [action]  # Convert int type to list type
        else:
            return action

    def _choose_action(self, *args):
        pass

    def run(self, *args):
        pass


class RandomPolicyAgent(Agent):

    def __init__(self, env_id):
        super(RandomPolicyAgent, self).__init__(env_id)

    def _choose_action(self):
        return self.env.action_space.sample()  # Explore action space

    def run(self, num_episode: int, *args):

        all_epochs, all_reward, all_penalty = [], [], []

        """ Loop for each episode. """
        for _ in tqdm(range(1, num_episode)):

            """ Initialize S. """
            self.env.reset()

            epochs, sum_reward, num_penalty, done = 0, 0, 0, False

            """ Loop for each step of episode until S is terminal. """
            while not done:
                action = self._choose_action()
                next_state, reward, done, info = self.env.step(action)

                # For taxi case: executing "pickup" and "drop-off" actions illegally
                if done:
                    reward = -10
                if reward == -10:
                    num_penalty += 1

                epochs += 1
                sum_reward += reward

            all_epochs.append(epochs)
            all_reward.append(sum_reward)
            all_penalty.append(num_penalty)

        summary(all_epochs, all_reward, all_penalty)
        return all_epochs, all_reward, all_penalty


class QLearningAgent(Agent):
    """ Q-Learning: On-policy TD control. """

    def __init__(self, env_id: str):
        super(QLearningAgent, self).__init__(env_id)
        print(self.q_table.shape)

    def _choose_action(self, _state):
        r""" Choose A from S using policy derived from Q (e.g. \epsilon-greedy). """
        return self._epsilon_greedy_select(_state)

    def _update_q_table(self, _state, _action, _new_state, _reward):
        r""" Q-learning Update Rule
            Q(S,A) <- Q(S,A) + \alpha [R + \gamma max_a Q(S',a) - Q(S,A)]
        """

        if self.env_id in ["CartPole-v1", "Pendulum-v0"]:
            _state = discrete_state_helper(_state, self.env_id)
            _new_state = discrete_state_helper(_new_state, self.env_id)

        if self.env_id == "Pendulum-v0":  # Need to convert the continuous action sampled
            _action = discrete_action_helper(_action, self.env_id)

        old_q_value = self.q_table[_state, _action]  # Q(S,A)
        max_q_value = np.max(self.q_table[_new_state])  # max_a Q(S',a)

        # Update Rule
        new_q_value = old_q_value + self.alpha * (_reward + self.gamma * max_q_value - old_q_value)
        self.q_table[_state, _action] = new_q_value

    def run(self, num_episode: int, mode="train"):

        all_epochs, all_reward, all_penalty = [], [], []

        """ Loop for each episode. """
        for _ in tqdm(range(1, num_episode)):

            """ Initialize S. """
            state = self.env.reset()
            epochs, reward_sum, num_penalty, done = 0, 0, 0, False

            """ Loop for each step of episode until S is terminal. """
            while not done:

                r""" Choose A from S using policy derived from Q (e.g. \epsilon greedy). """
                action = self._choose_action(state)

                r""" Take action A observe R, S'. """
                next_state, reward, done, info = self.env.step(action)

                r""" Update Q(S,A) with off-policy rule. """
                if mode == "train":
                    self._update_q_table(state, action, next_state, reward)

                """ S <- S' """
                state = next_state

                # For taxi case: executing "pickup" and "drop-off" actions illegally
                if done:
                    reward = -10
                if reward == -10:
                    num_penalty += 1

                reward_sum += reward
                epochs += 1

            all_epochs.append(epochs)
            all_reward.append(reward_sum)
            all_penalty.append(num_penalty)

        summary(all_epochs, all_reward, all_penalty)
        return all_epochs, all_reward, all_penalty


class SARSAgent(Agent):

    def __init__(self, env_id: str):
        super(SARSAgent, self).__init__(env_id)
        print(self.q_table.shape)

    def _choose_action(self, _state):
        r""" Choose A from S using policy derived from Q (e.g. \epsilon-greedy). """
        return self._epsilon_greedy_select(_state)

    def _update_q_table(self, _state, _action, _reward, _new_state, _new_action):
        r""" S-A-R-S-A Updating
        State - Action - Reward - New State - New Action => S-A-R-S-A
        Updating Rule: Q(S,A) <- Q(S,A) + \alpha [R + \gamma Q(S',A') - Q(S,A)]
        """

        # Convert continuous state to discrete state
        if self.env_id in ["CartPole-v1", "Pendulum-v0"]:
            _state = discrete_state_helper(_state, self.env_id)
            _new_state = discrete_state_helper(_new_state, self.env_id)

        # Convert continuous action to discrete action
        if self.env_id == "Pendulum-v0":  # Need to convert the continuous action sampled
            _action = discrete_action_helper(_action, self.env_id)
            _new_action = discrete_action_helper(_new_action, self.env_id)

        old_q_value = self.q_table[_state, _action]  # Q(S,A)
        # print(self.q_table.shape)
        # print(_new_state, _new_state)
        next_q_value = self.q_table[_new_state, _new_action]

        # Update Rule
        new_q_value = old_q_value + self.alpha * (_reward + self.gamma * next_q_value - old_q_value)
        self.q_table[_state, _action] = new_q_value

    def run(self, num_episode: int, mode="train"):

        all_epochs, all_reward, all_penalty = [], [], []

        """ Loop for each episode. """
        for _ in tqdm(range(1, num_episode)):

            """ Initialize S. """
            state = self.env.reset()
            epochs, reward_sum, num_penalty, done = 0, 0, 0, False

            r""" Choose A from S using policy derived from Q (e.g. \epsilon greedy). """
            action = self._choose_action(state)

            """ Loop for each step of episode until S is terminal. """
            while not done:

                r""" Take action A observe R, S'. """
                next_state, reward, done, info = self.env.step(action)

                r""" Choose A' from S' using policy derived from Q (e.g. \epsilon greedy). """
                next_action = self._choose_action(next_state)

                r""" Update Q(S,A) with on-policy rule. """
                if mode == "train":
                    self._update_q_table(state, action, reward, next_state, next_action)

                """ S <- S', A <- A' """
                state, action = next_state, next_action

                # For taxi case: executing "pickup" and "drop-off" actions illegally
                if done:
                    reward = -10
                if reward == -10:
                    num_penalty += 1

                reward_sum += reward
                epochs += 1

            all_epochs.append(epochs)
            all_reward.append(reward_sum)
            all_penalty.append(num_penalty)

        summary(all_epochs, all_reward, all_penalty)
        return all_epochs, all_reward, all_penalty


""" Part II: Tool functions 
    1. bins(): Convert continuous variable to discrete variables.
    2. discrete_action_helper(_action, _env_id): Convert continuous action value to discrete 
        (for Pendulum case)
    3. discrete_state_helper(_state, _env_id): Convert continuous state value to discrete 
        (for Cart-Pole and Pendulum case)
    4. summary(): Print the average rewards / time-step of different algorithms and cases
    5. visualize(): Plot 
"""


def bins(min_clip, max_clip, number):
    """ Convert continuous variable to discrete variable. """
    return np.linspace(min_clip, max_clip, number + 1)[1: -1]  # Num of bins needs num+1 variable.


def discrete_action_helper(_action, _env_id):
    """ Convert discrete action value to discrete action value. (For Pendulum-v0 case) """
    if _env_id == "Pendulum-v0":
        discrete_action = np.digitize(_action, bins=bins(-2.0, 2.0, discrete_factor)),
        discrete_state_id = sum([x * (discrete_factor ** i) for i, x in enumerate(discrete_action)])
        return discrete_state_id
    else:
        return 0


def discrete_state_helper(_state, _env_id):
    """ Convert the continuous variable of CartPole to 1296 discrete states.
        :return discrete_state_id: from 0 to 1295 (for CartPole Case)
    """

    if _env_id == "CartPole-v1":
        _cart_pos, _cart_v, _pole_angle, _pole_v = _state
        discrete_state = [
            np.digitize(_cart_pos, bins=bins(-2.4, 2.4, discrete_factor)),
            np.digitize(_cart_v, bins=bins(-3.0, 3.0, discrete_factor)),
            np.digitize(_pole_angle, bins=bins(-0.5, 0.5, discrete_factor)),
            np.digitize(_pole_v, bins=bins(-2.0, 2.0, discrete_factor))
        ]
    elif _env_id == "Pendulum-v0":
        _cos, _sin, _theta, = _state
        discrete_state = [
            np.digitize(_cos, bins=bins(-1.0, 1.0, discrete_factor)),
            np.digitize(_sin, bins=bins(-1.0, 1.0, discrete_factor)),
            np.digitize(_theta, bins=bins(-8.0, 8.0, discrete_factor)),
        ]
    else:
        discrete_state = []

    """ Convert state to state_id (int type). """
    discrete_state_id = sum([x * (discrete_factor ** i) for i, x in enumerate(discrete_state)])
    return discrete_state_id


def summary(epoch_list, reward_list, penal_list):
    len_epoch = len(epoch_list)
    print(f"Results after {len(epoch_list)} episodes:")
    print(f"Average time-steps per episode: {sum(epoch_list) / len_epoch}")
    print(f"Average reward per episode: {sum(reward_list) / len_epoch}")
    print(f"Average penalties per episode: {sum(penal_list) / len_epoch}")


def visualize(data_list, title_str):
    for data in data_list:
        plt.plot(data)
    plt.legend(labels=['Random Policy', 'Q-learning', 'SARSA'], loc='best')
    plt.title(title_str)
    plt.show()


""" Part III: Runner """


def case_script(env_id, num_episode):
    """ """
    random_agent = RandomPolicyAgent(env_id)
    random_epoch_list, random_reward_list, random_penal_list = random_agent.run(num_episode)

    q_learning_agent = QLearningAgent(env_id)
    q_epoch_list, q_reward_list, q_penal_list = q_learning_agent.run(num_episode)

    sarsa_agent = SARSAgent(env_id)
    sarsa_epoch_list, sarsa_reward_list, sarsa_penal_list = sarsa_agent.run(num_episode)

    visualize([random_epoch_list, q_epoch_list, sarsa_epoch_list],
              f"Total time-steps per episode: {env_id}")

    visualize([random_reward_list, q_reward_list, sarsa_reward_list],
              f"Total rewards per episode: {env_id}")

    visualize([random_penal_list, q_penal_list, sarsa_penal_list],
              f"Total penalties per episode: {env_id}")


if __name__ == '__main__':
    # Prepare OpenAI Gym Environment
    taxi_env_id, cart_env_id, pendulum_env_id = "Taxi-v3", "CartPole-v1", "Pendulum-v0"
    case_script(taxi_env_id, 1000)
    case_script(cart_env_id, 100000)
    case_script(pendulum_env_id, 100000)
