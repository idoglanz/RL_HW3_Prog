import numpy as np
import random
import mountain_car_with_data_collection as sim
from LSPI import *
from QLearning import *

class Sample:
    def __init__(self, state, action, reward, next_state):
        self.curr_state = state
        self.curr_action = action
        self.reward = reward
        self.next_state = next_state
        self.n_sampled = 0
        self.curr_phi = np.array(state.shape)
        self.next_phi = np.array(next_state.shape)


class DataSet:
    def __init__(self, env, N):
        self.environment = env
        self.n_samples = N
        self.samples = []
        self.episodes = 0
        self.means = np.array(2)
        self.stds = np.array(2)
        self.phi = np.zeros((N, 3 * (len(self.environment.observation_space.sample()) + 1)))
        self.reward = np.zeros(N)
        self.phi_next = np.zeros((N, 3 * (len(self.environment.observation_space.sample()) + 1), 3))
        self.collect_data()

    def collect_data(self):
        for i in range(self.n_samples):
            curr_state = self.environment.observation_space.sample()
            action = self.environment.action_space.sample()

            self.environment.reset_specific(curr_state[0], curr_state[1])
            next_state, reward, is_done, _ = self.environment.step(action)
            self.samples.append(Sample(curr_state, action, reward, next_state))

    def RBF_kernal(self):
        temp_vectors = np.zeros((len(self.samples), 2))

        for i, sample in enumerate(self.samples):
            temp_vectors[i] = (sample.curr_state + sample.next_state)/2

        self.means = np.mean(temp_vectors, axis=0)
        self.stds = np.std(temp_vectors, axis=0)

    def normalize(self):
        self.RBF_kernal()

        for sample in self.samples:

            # Normalize wrt mean and standard deviation, map to RBF and add a bias term:
            sample.curr_phi = np.append(np.exp(-np.divide((sample.curr_state - self.means), self.stds)), 1)
            sample.next_phi = np.append(np.exp(-np.divide((sample.next_state - self.means), self.stds)), 1)
            # print('after:', sample.curr_phi)
        self.generate_phi()

    def normalize_sample(self, new_sample):
        return np.append(np.exp(-np.divide((new_sample - self.means), self.stds)), 1)

    def generate_phi(self):
        for i, sample in enumerate(self.samples):
            self.phi[i, sample.curr_action * 3:sample.curr_action * 3 + len(sample.curr_phi)] = sample.curr_phi
            self.reward[i] = sample.reward
            for position in range(3):
                self.phi_next[i, position * 3:position * 3 + len(sample.next_phi), position] = sample.next_phi

    def generate_phi_sample(self, new_sample):
        phi = np.zeros((3 * (len(self.environment.observation_space.sample()) + 1), 3))
        for action in range(3):
            phi[action * 3:action * 3 + len(new_sample), action] = new_sample
        return phi

    def state2phi(self, new_sample):
        new_sample = self.normalize_sample(new_sample)
        return self.generate_phi_sample(new_sample)




    #
    # def generate_batch(self, dataset):
    #     return np.arange(dataset.n_samples)


if __name__ == '__main__':
    env = sim.MountainCarWithResetEnv()
    data = DataSet(env, 100000)
    data.normalize()

    state = env.reset()
    is_done = False

    # positive_rewards = 0
    # for sample in data.samples:
    #     if sample.reward > 0:
    #         positive_rewards += 1
    #
    # print(positive_rewards)

    model = LSPIModel(env)
    theta = model.train(data)
    print('Theta:', theta)

    env.render()
    while not is_done:

        phi_state = data.state2phi(state)
        Q = np.dot(phi_state.T, theta)
        action = np.argmax(Q)

        state, r, is_done, _ = env.step(action)
        env.render()
        print('reward:', r, ', state:', state)

    env.close()