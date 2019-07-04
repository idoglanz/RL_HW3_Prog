import numpy as np
import tensorflow as tf
from util import *


# Function that apply the algorithm for future discount reward
def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = cumulative_rewards * discount_rate + rewards[step]
        discounted_rewards[step] = cumulative_rewards

    return discounted_rewards


#  This function apply the discount_rewards for every game in the n_game_per_iter, then normalize it
def discount_and_normalized_rewards(all_rewards, discount_rate):
    all_discount_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flatten = np.concatenate(all_discount_rewards)
    mean = flatten.mean()
    std = flatten.std()
    return [(discounted_rewards - mean)/std for discounted_rewards in all_discount_rewards]


def PolicyGradient(env):
    n_inputs = 2  # obs[0] - position  obs[1] - action
    n_outputs = 3  # Number of possible actions
    learning_rate = 0.01
    n_hidden = 10
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.3)  # Weights initializer
    b_initializer = tf.constant_initializer(0.1)  # Bias initializer


    tf.reset_default_graph()  # Reset the graph for new training
    X = tf.placeholder(tf.float32, shape=(None, n_inputs))  # obs placeholder
    y = tf.placeholder(tf.int64, shape=(None))  # choosed action in training placeholder
    hidden1 = tf.layers.dense(X, n_hidden, kernel_initializer=initializer, bias_initializer=b_initializer,
                              activation=tf.nn.tanh)
    logits = tf.layers.dense(hidden1, n_outputs, kernel_initializer=initializer, bias_initializer=b_initializer)
    outputs = tf.nn.softmax(logits)  # Probabilitys



    action = tf.multinomial(tf.log(outputs), num_samples=3)
    # This actually display 3 integers
    # Which one will be the index of the probability target, see documentation

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=y)  # Loss function

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # The optimizer
    grad_and_vars = optimizer.compute_gradients(loss)  # Get the gradients and vars of the loss function
    gradients = [grad for grad, var in grad_and_vars]  # Get the gradients
    gradients_placeholders = []  # To store the gradients
    grad_and_vars_feed = []  # To store the placeholders with the vars
    for grad, var in grad_and_vars:
        grad_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())  # The respective grad placeholder
        gradients_placeholders.append(grad_placeholder)  # Store the placeholder
        grad_and_vars_feed.append((grad_placeholder, var))  # Store the placeholder with his respective var

    training_op = optimizer.apply_gradients(grad_and_vars_feed)  # OP to apply the placeholder value with each var

    iterations = 100
    n_games_per_iter = 10  # Each iteration will play the game 10 times and store the rewards for every game
    discount_rate = 0.95
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        iter_height = -0.45
        for iteration in range(iterations):

            all_rewards = []  # Store all the 10 games rewards
            all_gradients = []  # Store all the 10 games gradients values
            rewards_sum = []  # Store the rewards for tracking improvements

            for game in range(n_games_per_iter):

                current_rewards = []  # Store the rewards for the game
                current_gradients = []  # Store the gradients of the game

                obs = env.reset()  # Reset enviroment

                while True:

                    previousHeight = obs[0]  # Store the previous height

                    all_acts = sess.run(action,
                                        feed_dict={X: obs.reshape(1, n_inputs)})  # run the tf.multinomial function

                    choosed_action = np.random.choice(all_acts.ravel())  # choose a random action

                    # Run the gradients
                    gradients_val = sess.run(gradients,
                                             feed_dict={X: obs.reshape(1, n_inputs), y: np.array([choosed_action])})

                    # Run the step with the choosed action
                    obs_, reward, done, info = env.step(choosed_action)

                    if obs_[0] > previousHeight:
                        # This is tricky, the env dont give any positive reward to the agent, its just -1 for each time step
                        # So i give a positive reward every time the agente reachs a higher height then the previou
                        reward += 1

                    if obs_[0] > iter_height:  # Compare with max height for all iterations
                        # I'm tracking the max height that ever ocurred
                        iter_height = obs_[0]  # Update if is the case
                        print("New maxHeight: {}".format(iter_height))  # Print
                        reward += 2  # I'm giving a even bigger reward when it's reaches the max height for all the iteration
                        # Hoping this helps the agente figure it out

                    current_rewards.append(reward)  # Append the step reward
                    rewards_sum.append(reward)
                    current_gradients.append(gradients_val)  # Append the step gradient val
                    obs = obs_  # Upadate the previous obs

                    if done:
                        all_rewards.append(current_rewards)  # Append all the steps rewards for the game
                        all_gradients.append(current_gradients)  # Append all the steps grad vals for the game
                        break

            all_rewards = discount_and_normalized_rewards(all_rewards,
                                                          discount_rate)  # Apply discount and normalize the rewards
            feed_dict = {}
            for var_index, grad_placeholder in enumerate(gradients_placeholders):
                # Calculate the mean of all the grad values for each game and each game step
                mean_gradient = np.mean([reward * all_gradients[game_index][step][var_index] for game_index, rewards \
                                         in enumerate(all_rewards) for step, reward in enumerate(rewards)], axis=0)

                feed_dict[grad_placeholder] = mean_gradient  # Store the value in the respective grad placeholder

            sess.run(training_op, feed_dict=feed_dict)  # Run the optimizer for the apply_gradients function

            if iteration % 50 == 0:
                print("Rewards: {}".format(sum(rewards_sum)))  # Print the sum of the rewards for all games
                saver.save(sess, './logs/pg_net7.ckpt')  # Save the iteration

# Code to see the agente playing the game

    with tf.Session() as sess:
        saver.restore(sess, './logs/pg_net7.ckpt')

        for game in range(30):

            obs = env.reset()

            while True:

                action = sess.run(outputs, feed_dict={X: obs.reshape(1, n_inputs)})
                choose_action = np.argmax(action, 1)
                obs, reward, done, info = env.step(choose_action[0])
                env.render()
                print(done)
                if done:
                    break

# class PolicyGradient:
#     def __init__(self, env, alpha=1, batch_size=1, max_iterations=100, trajectories=100, epsilon=0, theta_size=9):
#         self.env = env
#         self.theta = np.random.rand(theta_size)
#
#         self.batch_size = batch_size
#         self.trajectories = trajectories
#         self.max_iterations = max_iterations
#         self.epsilon = epsilon
#
#         self.n_inputs = 2  # obs[0] - position  obs[1] - action
#         self.n_outputs = 3  # Number of possible actions
#         self.learning_rate = 0.01
#         self.n_hidden = 10
#         self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.3)  # Weights initializer
#         self.b_initializer = tf.constant_initializer(0.1)  # Bias initializer
#
#     def train(self):
#         iteration = 1
#         done = False
#         while not done:
#             data = DataSet(self.env, N=self.batch_size, trajectories=self.trajectories, method='Q_learning', theta=self.theta, iteration=iteration)
#             data.normalize()
#
#             self.theta = self.theta + (self.alpha/(len(data.samples))) * np.dot(data.reward  * np.dot(data.phi_next, self.theta) - np.dot(data.phi, self.theta), data.phi)
#
#             # print(self.theta)
#             prev_theta = self.theta
#             iteration += 1
#             if iteration > self.max_iterations:
#                 done = True
#
#         return self.theta, data
