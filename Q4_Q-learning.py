import numpy as np
import mountain_car_with_data_collection as sim


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
        self.means = np.array([0.6,0])
        self.stds = np.array([0.01, 0.01])
        self.phi = np.zeros((N, 3 * (len(self.environment.observation_space.sample()) + 1)))
        self.reward = np.zeros(N)
        self.phi_next = np.zeros((N, 3 * (len(self.environment.observation_space.sample()) + 1), 3))

        # self.collect_data()

    def epsilon_greedy(self, phi_state, theta, epsilon):

        a_space = np.dot(phi_state.T, theta)

        if np.random.rand(1) > epsilon:
            return np.argmax(a_space)
        else:
            action = np.round(np.random.uniform(0, 2))
            return action

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
        # self.means = np.array([0,0])
        self.stds = np.std(temp_vectors, axis=0)
        # self.stds = np.array([1.8, 0.14])

    def normalize(self):
        self.RBF_kernal()

        for sample in self.samples:

            # Normalize wrt mean and standard deviation, map to RBF and add a bias term:
            sample.curr_phi = np.append(np.exp(-np.divide(np.abs(sample.curr_state - self.means)**2, self.stds**2)), 1)
            # sample.curr_phi = np.append(np.divide((sample.curr_state - self.means), self.stds), 1)
            sample.next_phi = np.append(np.exp(-np.divide(np.abs(sample.next_state - self.means)**2, self.stds**2)), 1)
            # sample.next_phi = np.append(np.divide((sample.next_state - self.means), self.stds), 1)

            # print('after:', sample.curr_phi)

        self.generate_phi()

    def normalize_sample(self, new_sample):
        return np.append(np.exp(-np.divide(np.abs(new_sample - self.means)**2, self.stds**2)), 1)
        # return np.append(np.divide((new_sample - self.means), self.stds), 1)

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


class Qlearn:
    def __init__(self, environment, alpha=0.1):
        self.env = environment
        self.alpha = alpha
        self.theta = np.random.rand(9)

    def collect_data(self, epsilon, max_samples=1000):
        iteration = 0
        done = False
        state = self.env.reset()
        games_played = 0
        while iteration < max_samples:
            phi_state = data.state2phi(state)
            action = data.epsilon_greedy(phi_state, self.theta, epsilon=epsilon)
            next_state, reward, done, _ = self.env.step(action)
            data.samples.append(Sample(state, action, reward, next_state))
            if done:
                state = self.env.reset()
                done = False
                games_played += 1
            iteration += 1

    def learn(self):
        self.theta += self.alpha * 

class LSPIModel:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.batch_size = 0
        self.max_iterations = 120
        self.epsilon = 1e-4

    def train(self, dataset, method='TD0'):

        self.batch_size = len(dataset.samples)
        theta = np.random.rand(9)
        done = False
        iteration = 0
        if method == 'TD0':

            while not done:
                d = np.dot(dataset.phi.T, dataset.reward) / self.batch_size

                # generate a phi vector for each possible action:
                max_phi_next = self.find_argmax_a(theta, dataset.phi_next)

                C = np.dot(dataset.phi.T, (dataset.phi - self.gamma * max_phi_next)) / self.batch_size

                prev_theta = theta
                theta = np.dot(np.linalg.inv(C), d)
                iteration += 1

                # if np.sum((prev_theta-theta)**2) <= self.epsilon or iteration > self.max_iterations:
                if iteration > self.max_iterations:
                    done = True

                if not np.mod(iteration, 20):
                    print(iteration)

            print(iteration)

        return theta

    def find_argmax_a(self, theta, phi_next):
        a_space = np.zeros((len(phi_next), 3))
        phi_next_argmax = np.zeros((len(phi_next), len(theta)))

        for position in range(3):
            a_space[:, position] = np.dot(phi_next[:, :, position], theta)

        for row in range(len(phi_next)):
            phi_next_argmax[row, :] = phi_next[row, :, np.argmax(a_space[row])]

        return phi_next_argmax

    #
    # def generate_batch(self, dataset):
    #     return np.arange(dataset.n_samples)


if __name__ == '__main__':
    env = sim.MountainCarWithResetEnv()
    data = DataSet(env, 100000)
    data.normalize()

    state = env.reset()
    # state = env.reset_specific(0.3, 0.0)

    is_done = False

    positive_rewards = 0
    for sample in data.samples:
        if sample.reward > 0:
            positive_rewards += 1

    print(positive_rewards)

    model = LSPIModel(env)
    theta = model.train(data)
    print('Theta:', theta)

    env.render()
    while not is_done:

        phi_state = data.state2phi(state)
        Q = np.dot(phi_state.T, theta)
        action = np.argmax(Q)

        print('Action:', action)
        state, r, is_done, _ = env.step(action)
        env.render()
        print('reward:', r, ', state:', state)

    env.close()