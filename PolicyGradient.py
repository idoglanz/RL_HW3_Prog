import numpy as np
from util import *


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
            data = DataSet(self.env, N=self.batch_size, trajectories=self.trajectories, method='Q_learning', theta=self.theta, iteration=iteration)
            data.normalize()

            self.theta = self.theta + (self.alpha/(len(data.samples))) * np.dot(data.reward  * np.dot(data.phi_next, self.theta) - np.dot(data.phi, self.theta), data.phi)

            # print(self.theta)
            prev_theta = self.theta
            iteration += 1
            if iteration > self.max_iterations:
                done = True

        return self.theta, data
