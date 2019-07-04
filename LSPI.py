import numpy as np


class LSPIModel:
    def __init__(self, env, gamma=0.99, max_iterations=5, epsilon=0, theta_size=0):
        self.env = env
        self.gamma = gamma
        self.batch_size = 0
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.theta = np.random.rand(theta_size)

    def train(self, dataset, simulation):
        self.batch_size = len(dataset.samples)

        done = False
        iteration = 0
        success_rate = []

        while not done:
            # generate a phi vector for each possible action:
            max_phi_next = self.find_argmax_a(self.theta, dataset.phi_next)

            C = np.dot(dataset.phi.T, (dataset.phi - self.gamma * max_phi_next)) / self.batch_size
            d = np.dot(dataset.phi.T, dataset.reward) / self.batch_size

            prev_theta = self.theta
            self.theta = np.dot(np.linalg.inv(C), d)

            success_rate = np.append(success_rate, simulation.success(theta=self.theta))

            # print('success rate: ', success_rate, 'iteration', iteration)
            iteration += 1
            # if np.sum((prev_theta-self.theta)**2) <= self.epsilon or iteration > self.max_iterations:
            if iteration > self.max_iterations or success_rate[-1] > 90:
                done = True
                iteration_fill = iteration
                while iteration_fill < self.max_iterations:
                    success_rate = np.append(success_rate, success_rate[-1])
                    iteration_fill += 1

        return self.theta, [success_rate[-1], iteration], success_rate

    def find_argmax_a(self, theta, phi_next):
        a_space = np.zeros((len(phi_next), 3))
        phi_next_argmax = np.zeros((len(phi_next), len(theta)))

        for position in range(3):
            a_space[:, position] = np.dot(phi_next[:, :, position], theta)

        for row in range(len(phi_next)):
            phi_next_argmax[row, :] = phi_next[row, :, np.argmax(a_space[row])]

        return phi_next_argmax
