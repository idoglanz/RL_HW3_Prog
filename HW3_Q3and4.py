import numpy as np
import mountain_car_with_data_collection as sim
import time


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
    def __init__(self, env, N=10000, method='LSPI', theta=0, trajectories=1, iteration=1):
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

            # Normalize wrt mean and standard deviation, map to RBF and add a bias term:
            sample.curr_phi = np.append(np.append(np.append(np.exp(np.dot((-np.abs(sample.curr_state[0] - 0.6)), 2)), np.sign(sample.curr_state[1])), sample.curr_state**2),  1)
            sample.next_phi = np.append(np.append(np.append(np.exp(np.dot((-np.abs(sample.next_state[0] - 0.6)), 2)), np.sign(sample.next_state[1])), sample.next_state**2),  1)

        self.generate_phi()

    def normalize_sample(self, new_sample):
        # return np.append(np.exp(-np.divide((np.abs(new_sample - self.means)), self.stds)), 1)

        return np.append(np.append(np.append(np.exp(np.dot((-np.abs(new_sample[0] - 0.6)), 2)), np.sign(new_sample[1])), new_sample**2), 1)

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


class LSPIModel:
    def __init__(self, env, gamma=0.99, max_iterations=5, epsilon=0, theta_size=0):
        self.env = env
        self.gamma = gamma
        self.batch_size = 0
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.theta = np.random.rand(theta_size)

    def train(self, dataset):

        self.batch_size = len(dataset.samples)

        done = False
        iteration = 0

        while not done:
            # generate a phi vector for each possible action:
            max_phi_next = self.find_argmax_a(self.theta, dataset.phi_next)

            C = np.dot(dataset.phi.T, (dataset.phi - self.gamma * max_phi_next)) / self.batch_size
            d = np.dot(dataset.phi.T, dataset.reward) / self.batch_size

            prev_theta = self.theta
            self.theta = np.dot(np.linalg.inv(C), d)
            iteration += 1

            # if np.sum((prev_theta-self.theta)**2) <= self.epsilon or iteration > self.max_iterations:
            if iteration > self.max_iterations:
                done = True

        # print(iteration)

        return self.theta

    def find_argmax_a(self, theta, phi_next):
        a_space = np.zeros((len(phi_next), 3))
        phi_next_argmax = np.zeros((len(phi_next), len(theta)))

        for position in range(3):
            a_space[:, position] = np.dot(phi_next[:, :, position], theta)

        for row in range(len(phi_next)):
            phi_next_argmax[row, :] = phi_next[row, :, np.argmax(a_space[row])]

        return phi_next_argmax


class QLearning:
    def __init__(self, env, alpha=1, batch_size=1, max_iterations=100, trajectories=100, epsilon=0, theta_size=9):
        self.env = env
        self.alpha = alpha
        self.theta = np.random.rand(theta_size)

        self.batch_size = batch_size
        self.trajectories = trajectories
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def train(self):
        iteration = 1
        done = False
        data = DataSet(env, N=self.batch_size, trajectories=self.trajectories, method='Q_learning', iteration=iteration)
        while not done:
            data.collect(theta=self.theta, iteration=iteration)
            data.normalize()

            # self.theta = self.theta + (self.alpha/(self.batch_size*self.trajectories)) * np.dot(data.reward  * np.dot(data.phi_next, self.theta) - np.dot(data.phi, self.theta), data.phi)
            self.theta = self.theta + (self.alpha/len(data.samples)) *\
                         np.dot(data.reward * np.dot(data.phi_next, self.theta) - np.dot(data.phi, self.theta), data.phi)

            # print(self.theta)
            prev_theta = self.theta
            iteration += 1
            if iteration > self.max_iterations:
                done = True

        return self.theta, data


class PolicyGradient:
    def __init__(self, env, alpha=1, batch_size=1, max_iterations=100, trajectories=100, epsilon=0, theta_size=9):
        self.env = env
        self.alpha = alpha
        self.theta = np.random.rand(theta_size)

        self.batch_size = batch_size
        self.trajectories = trajectories
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def train(self):
        iteration = 1
        done = False
        while not done:
            data = DataSet(env, N=self.batch_size, trajectories=self.trajectories, method='Q_learning', theta=self.theta, iteration=iteration)
            data.normalize()

            self.theta = self.theta + (self.alpha/(len(data.samples))) * np.dot(data.reward  * np.dot(data.phi_next, self.theta) - np.dot(data.phi, self.theta), data.phi)

            # print(self.theta)
            prev_theta = self.theta
            iteration += 1
            if iteration > self.max_iterations:
                done = True

        return self.theta, data


class PlaySimulation:
    def __init__(self, env, data):
        self.env = env
        self.data = data

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

    def success(self, theta, starting=10, how_long_iterations=10000):
        success_rate = 0.
        for i in range(starting):
            state = self.env.reset()

            for _ in range(how_long_iterations):
                phi_state = self.data.state2phi(state)
                Q = np.dot(phi_state.T, theta)
                action = np.argmax(Q)
                state, r, is_done, _ = self.env.step(action)
                if is_done:
                    success_rate += 1
                    break

        return success_rate/starting * 100


def q3_testing_the_model(env):
    print('Q3 - testing the model')
    method = 'LSPI'

    for max_iterations in range(0, 10, 1):
        model = LSPIModel(env, gamma=0.999, max_iterations=max_iterations, epsilon=0, theta_size=15)
        data = DataSet(env, 10000, method=method)
        data.collect()
        data.normalize()
        theta = model.train(data)

        PlaySimulation(env, data).play(theta)
        # print(PlaySimulation(env, data).success(theta, starting=10))


def q4_testing_the_model(env):
    print('Q4 - testing the model')

    for max_iterations in range(10, 100, 10):
        model = QLearning(env, alpha=0.1, batch_size=1, trajectories=100, max_iterations=max_iterations, epsilon=1e-4, theta_size=15)
        theta, data = model.train()

        PlaySimulation(env, data).play(theta)
        print(PlaySimulation(env, data).success(theta, starting=10))


if __name__ == '__main__':
    env = sim.MountainCarWithResetEnv()

    q3_testing_the_model(env)
    q4_testing_the_model(env)



    # method = 'LSPI'
    # method = 'Q_learning'
    #
    # if method == 'LSPI':
    #     model = LSPIModel(env, gamma=0.999, max_iterations=200, epsilon=1e-10, theta_size=15)
    #     data = DataSet(env, 10000, method=method)
    #     data.collect()
    #     data.normalize()
    #     theta = model.train(data)
    #     print('Theta:', theta)
    #
    # if method == 'Q_learning':
    #     model = QLearning(env, alpha=0.1, batch_size=1, trajectories=100, max_iterations=100, epsilon=1e-4, theta_size=15)
    #     theta, data = model.train()
    #
    # PlaySimulation(env, data).play(theta)

