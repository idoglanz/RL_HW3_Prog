import numpy as np
import time
import matplotlib.pyplot as plt
# import mountain_car_with_data_collection as sim


class Sample:
    def __init__(self, state, action, reward, next_state, next_action=[]):
        self.curr_state = state
        self.curr_action = action
        self.reward = reward
        self.next_state = next_state
        self.next_action = next_action
        self.n_sampled = 0
        self.curr_phi = np.array(state.shape)
        self.next_phi = np.array(next_state.shape)


class DataSet:
    def __init__(self, env, N=10000, method='LSPI', theta=0, trajectories=1, iteration=1, epsilon_greedy_val=0):
        self.environment = env
        self.n_samples = N
        self.trajectories = trajectories
        self.samples = []
        self.episodes = 0
        self.means = np.array(2)
        self.stds = np.array(2)
        self.phi = np.zeros((N*trajectories, 3 * (len(self.environment.observation_space.sample()) + 1)))
        self.reward = np.zeros(N*trajectories)
        self.phi_next = np.zeros((N*trajectories, 3 * (len(self.environment.observation_space.sample()) + 1), 3))
        self.method = method
        self.iteration = iteration
        self.epsilon_greedy_val = epsilon_greedy_val

    def reset_data(self):
        self.samples = []

    def collect(self, theta=0, iteration=1):
        self.iteration = iteration
        if self.method == 'LSPI':
            self.collect_data()
        if self.method == 'Q_learning':
            self.collect_data_with_trajectory(theta)

    def collect_data(self):
        for i in range(self.n_samples):

            curr_state = self.environment.observation_space.sample()
            action = self.environment.action_space.sample()

            self.environment.reset_specific(curr_state[0], curr_state[1])
            next_state, reward, is_done, _ = self.environment.step(action)
            self.samples.append(Sample(curr_state, action, reward, next_state))

    def collect_data_with_trajectory(self, theta):
        self.reset_data()

        for _ in range(self.trajectories):
            curr_state = self.environment.reset()

            action = self.environment.action_space.sample()
            done = False
            trip = 0

            while not done and trip < self.n_samples:

                next_state, reward, done, _ = self.environment.step(action)
                next_action = self.epsilon_greedy(theta, next_state)

                self.samples.append(Sample(curr_state, action, reward, next_state, next_action))
                action = next_action
                # curr_state = next_state
                trip += 1

    def epsilon_greedy(self, theta, next_state):
        phi_next = self.state2phi(next_state)

        a_space = np.zeros(3)

        for position in range(3):
            a_space[position] = np.dot(phi_next[:, position], theta)

        max_action = np.argmax(a_space)
        probability = np.array([self.iteration, 1., 1., 1.])
        probability /= np.sum(probability)

        if self.epsilon_greedy_val:
            probability = np.array([1-self.epsilon_greedy_val, self.epsilon_greedy_val/3, self.epsilon_greedy_val/3, self.epsilon_greedy_val/3])

        action = np.random.choice([max_action, 0, 1, 2], 1, p=probability)  # TODO: prob choice.....
        # phi_next_argmax[row, :] = phi_next[row, :, choose_action]

        return action[0]

    def RBF_kernal(self):
        temp_vectors = np.zeros((len(self.samples), 2))

        for i, sample in enumerate(self.samples):
            temp_vectors[i] = (sample.curr_state + sample.next_state)/2

        self.means = np.mean(temp_vectors, axis=0)
        self.stds = np.std(temp_vectors, axis=0)
        # self.stds = 1

    def normalize(self):
        self.RBF_kernal()

        for sample in self.samples:
            # sample.curr_state = np.divide((sample.curr_state - self.means), self.stds)
            # sample.next_state = np.divide((sample.curr_state - self.means), self.stds)

            # Normalize wrt mean and standard deviation, map to RBF and add a bias term:
            # sample.curr_phi = np.append(np.append(np.append(np.exp(-(sample.curr_state[0]-2)*2), np.sign(sample.curr_state[1])), sample.curr_state**2),  1)
            # sample.next_phi = np.append(np.append(np.append(np.exp(-(sample.next_state[0]-2)*2), np.sign(sample.next_state[1])), sample.next_state**2),  1)

            sample.curr_phi = np.append(np.append(np.append(np.exp(np.dot((-np.abs(sample.curr_state[0] - 0.6)), 2)), np.sign(sample.curr_state[1])), sample.curr_state**2),  1)
            sample.next_phi = np.append(np.append(np.append(np.exp(np.dot((-np.abs(sample.next_state[0] - 0.6)), 2)), np.sign(sample.next_state[1])), sample.next_state**2),  1)

        self.generate_phi()

    def normalize_sample(self, new_sample):
        # return np.append(np.exp(-np.divide((np.abs(new_sample - self.means)), self.stds)), 1)
        # new_sample = np.divide(new_sample - self.means, self.stds)
        return np.append(np.append(np.append(np.exp(np.dot((-np.abs(new_sample[0] - 0.6)), 2)), np.sign(new_sample[1])), new_sample**2), 1)

        # return np.append(np.append(np.append(np.exp(-(new_sample[0]-2)*2), np.sign(new_sample[1])), new_sample**2), 1)

    def generate_phi(self):
        self.phi = np.zeros((len(self.samples), 3 * len(self.samples[0].curr_phi)))
        self.reward = np.zeros(len(self.samples))

        if self.method == 'Q_learning':
            self.phi_next = np.zeros((len(self.samples), 3 * len(self.samples[0].curr_phi)))

            for i, sample in enumerate(self.samples):
                self.phi[i, sample.curr_action * len(sample.curr_phi):sample.curr_action * len(sample.curr_phi) + len(
                    sample.curr_phi)] = sample.curr_phi
                self.phi_next[i, sample.next_action * len(sample.next_phi):sample.next_action * len(sample.next_phi) + len(
                    sample.next_phi)] = sample.next_phi
                self.reward[i] = sample.reward

        if self.method == 'LSPI':
            self.phi_next = np.zeros((len(self.samples), 3 * len(self.samples[0].curr_phi), 3))

            for i, sample in enumerate(self.samples):
                self.phi[i, sample.curr_action * len(sample.curr_phi):sample.curr_action * len(sample.curr_phi) + len(
                    sample.curr_phi)] = sample.curr_phi
                self.reward[i] = sample.reward
                for position in range(3):
                    self.phi_next[i, position * len(sample.next_phi):position * len(sample.next_phi) + len(sample.next_phi), position] = sample.next_phi

    def generate_phi_sample(self, new_sample):
        phi = np.zeros((3 * len(new_sample), 3))
        for action in range(3):
            phi[action * len(new_sample):action * len(new_sample) + len(new_sample), action] = new_sample
        return phi

    def state2phi(self, new_sample):
        new_sample = self.normalize_sample(new_sample)
        return self.generate_phi_sample(new_sample)


class PlaySimulation:
    def __init__(self, env, data, starts=10):
        self.env = env
        self.data = data
        self.start_states = []
        self.starts = starts

        self.initial_starts(starts)

    def play(self, theta, how_long_time=10):

        state = self.env.reset()
        is_done = False
        self.env.render()
        start_time = time.time()

        while not is_done:
            phi_state = self.data.state2phi(state)
            Q = np.dot(phi_state.T, theta)
            action = np.argmax(Q)
            state, r, is_done, _ = self.env.step(action)
            self.env.render()

            time.sleep(0.01)  # TODO: delete pause action
            # print('action: ', action, 'reward:', r, ', state:', state)
            if time.time() - start_time > how_long_time:
                break

        self.env.close()

    def success(self, theta, how_long_iterations=5000):
        # success_rate = np.zeros(self.starts)
        success_rate = 0.

        for i in range(self.starts):
            state = self.env.reset_specific(self.start_states[i, 0], self.start_states[i, 1])

            for _ in range(how_long_iterations):
                phi_state = self.data.state2phi(state)
                Q = np.dot(phi_state.T, theta)
                action = np.argmax(Q)
                state, r, is_done, _ = self.env.step(action)
                if is_done:
                    success_rate += 1
                    break

        return success_rate/self.starts * 100.

    def initial_starts(self, starts):
        self.start_states = np.zeros((starts, 2))

        for i in range(starts):
            self.start_states[i] = self.env.reset()


def plot_success(success_rate, max_iteration, average, N):

    mean_success = np.zeros((average, max_iteration))
    # mean_success = np.zeros(max_iteration)
    # for iterations in range(max_iteration):
    #     for success in enumerate(success_rate):
    #         mean_success[iterations] += success[1][iterations] / average
    for i, success in enumerate(success_rate):
        mean_success[i] = success[:max_iteration]
        # mean_success[i] = success[1]
    mean_success = np.mean(mean_success, axis=0)

    plt.plot(mean_success, label=['number of samples: ', + N])
    # plt.plot(mean_success)
    # plt.title()
    plt.xlabel('LSPI iterations')
    plt.ylabel('Average success')
    plt.ylim([0, 110])
    plt.grid()
    plt.legend()
