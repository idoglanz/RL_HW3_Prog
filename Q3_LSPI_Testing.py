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
        self.RBF_kernals = []
        self.n_kernals = 1
        self.size_phi = self.n_kernals + 2
        self.means = np.array(2)
        self.stds = np.array(2)
        self.phi = np.zeros((N, 3*self.size_phi))
        self.reward = np.zeros(N)
        self.phi_next = np.zeros((N, 3*self.size_phi, 3))

        self.collect_data()

    def collect_data(self):
        for i in range(self.n_samples):
            curr_state = self.environment.observation_space.sample()
            action = self.environment.action_space.sample()

            self.environment.reset_specific(curr_state[0], curr_state[1])
            next_state, reward, is_done, _ = self.environment.step(action)

            self.samples.append(Sample(curr_state, action, reward, next_state))

    def get_mean_and_std(self):
        temp_vectors = np.zeros((len(self.samples), 2))

        for i, sample in enumerate(self.samples):
            temp_vectors[i] = (sample.curr_state + sample.next_state)/2

        self.means = np.mean(temp_vectors, axis=0)
        self.stds = np.std(temp_vectors, axis=0)

    def generate_RBF(self, n):
        centers_position = np.linspace(0.6, 0.6, n)
        centers_velocity = np.linspace(0.07, 0.07, n)
        for i in range(n):
            self.RBF_kernals.append([centers_position[i], centers_velocity[i]])
        print(self.RBF_kernals)

    def map2phi(self, state):
        phi_vector = np.zeros(int(self.n_kernals))
        for i, kernel in enumerate(self.RBF_kernals):
            phi_vector[i] = np.sum(np.exp(-np.abs(state - kernel)**2/self.stds))
        # return np.append(phi_vector, [state[0]**2, state[1]**2, np.sign(state[1]), 1])
        return np.append(phi_vector, [np.sign(state[1]), 1])

    def normalize(self):
        self.get_mean_and_std()
        self.generate_RBF(self.n_kernals)

        for sample in self.samples:
            sample.curr_state = self.normalize_sample(sample.curr_state)
            sample.next_state = self.normalize_sample(sample.next_state)

            sample.curr_phi = self.map2phi(sample.curr_state)
            sample.next_phi = self.map2phi(sample.next_state)

        self.generate_phi_array()

    def normalize_sample(self, new_sample):
        return np.divide((new_sample - self.means), self.stds)

    def generate_phi_array(self):
        spacing = int(self.size_phi)
        for i, sample in enumerate(self.samples):
            self.phi[i, sample.curr_action * spacing:sample.curr_action * spacing + spacing] = sample.curr_phi
            self.reward[i] = sample.reward
            for position in range(3):
                self.phi_next[i, position * spacing:position * spacing + spacing, position] = sample.next_phi

    def add_action_space(self, new_sample):
        spacing = int(self.size_phi)
        phi = np.zeros((3*spacing, 3))
        for action in range(3):
            phi[action * spacing:action * spacing + spacing, action] = new_sample
        return phi

    def state2phi(self, new_sample):
        new_sample = self.normalize_sample(new_sample)
        return self.add_action_space(self.map2phi(new_sample))


class LSPIModel:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.batch_size = 0
        self.max_iterations = 10
        self.epsilon = 1e-6

    def train(self, dataset, method='TD0'):

        self.batch_size = len(dataset.samples)
        theta = np.random.rand(3*dataset.size_phi)

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

                if np.sum((prev_theta-theta)**2) <= self.epsilon or iteration > self.max_iterations:
                # if iteration > self.max_iterations:
                    done = True

                if not np.mod(iteration, 10):
                    print(np.sum((prev_theta-theta)**2))
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
    data = DataSet(env, 40000)
    data.normalize()

    state = env.reset()
    print('Initial state:', state)
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