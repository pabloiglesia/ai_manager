import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from Environment import Environment
import math


class TrainingStatistics:
    """
       Class were all the statistics of the training will be stored.
       """

    def __init__(self):
        self.current_step = 0  # Current step since the beginning of training
        self.episode = 0  # Number of episode
        self.episode_steps = [0]  # Steps taken by each episode
        self.episode_picks = [0]  # Pick actions tried by each episode
        self.episode_total_reward = [0]  # Total reward of each episode
        self.episode_random_actions = [0]  # Number of random actions in each episode
        self.episode_succeed = []  # Array that stores whether each episode has ended successfully or not
        self.coordinates_matrix = self.generate_coordinates_matrix()

    def generate_coordinates_matrix(self):
        x_limit = Environment.X_LENGTH / 2
        y_limit = Environment.Y_LENGTH / 2

        matrix_width = 2 * math.ceil(x_limit / Environment.ACTION_DISTANCE)
        matrix_height = 2 * math.ceil(y_limit / Environment.ACTION_DISTANCE)

        return [([0] * matrix_height) for i in range(matrix_width)]

    def fill_coordinates_matrix(self, coordinates):
        try:
            matrix_width = len(self.coordinates_matrix[0])  # y
            matrix_height = len(self.coordinates_matrix)  # x

            x_idx = int(math.ceil(coordinates[0] / Environment.ACTION_DISTANCE) + (matrix_height / 2) - 1)
            y_idx = int(math.ceil(coordinates[1] / Environment.ACTION_DISTANCE) + (matrix_width / 2) - 1)

            self.coordinates_matrix[x_idx][y_idx] += 1
        except:
            print('Error while filling coordinates statistics matrix')

    def new_episode(self):
        self.episode += 1  # Increase the episode counter
        self.episode_steps.append(0)  # Append a new value to the next episode step counter
        self.episode_picks.append(0)  # Append a new value to the amount of picks counter
        self.episode_total_reward.append(0)  # Append a new value to the next episode total reward counter
        self.episode_random_actions.append(0)  # Append a new value to the next episode random actions counter

    def new_step(self):
        self.current_step += 1  # Increase step
        self.episode_steps[-1] += 1  # Increase current episode step counter

    def increment_picks(self):
        self.episode_picks[-1] += 1  # Increase of the statistics counter

    def add_reward(self, reward):
        self.episode_total_reward[-1] += reward

    def add_succesful_episode(self, successful):
        self.episode_succeed.append(successful)

    def random_action(self):
        self.episode_random_actions[-1] += 1

    def save(self, filename='trainings/rl_algorithm_stats.pkl'):
        def create_if_not_exist(filename):
            current_path = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(current_path, filename)
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            return filename
        filename = create_if_not_exist(filename)
        with open(filename, 'wb+') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def recover(filename='trainings/rl_algorithm_stats.pkl'):
        current_path = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(current_path, filename)
        print(filename)
        try:
            with open(filename, 'rb') as input:
                stats = pickle.load(input)
                return stats
        except IOError:
            print("There is no Training saved in this path.")


if __name__ == '__main__':

    IMAGE_PATH = "statistics/"

    if not os.path.isdir(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)

    stats = TrainingStatistics.recover()

    print("Steps performed: {}".format(stats.current_step))
    print("Episodes performed: {}".format(stats.episode))
    print("Amount of pick actions: {}".format(sum(stats.episode_picks)))
    successful_episodes = list(filter((False).__ne__, stats.episode_succeed))
    print("Percentage of successful episodes: {}".format(100 * len(successful_episodes) / len(stats.episode_succeed)))

    # Steps per episodes
    try:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(stats.episode_steps) + 1), stats.episode_steps)

        ax.set(xlabel='Episodes', ylabel='Number of steps',
               title='Evolution of number of steps per episode')
        ax.grid()

        fig.savefig(IMAGE_PATH + "steps_per_episode.png")
        plt.show()
    except:
        print("Error while plotting steps_per_episode")

    # Picks per episodes
    try:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(stats.episode_picks) + 1), stats.episode_picks)

        ax.set(xlabel='Episodes', ylabel='Number of picks',
               title='Evolution of number of pick actions per episode')
        ax.grid()

        fig.savefig(IMAGE_PATH + "picks_per_episode.png")
        plt.show()
    except:
        print("Error while plotting picks_per_episode")

    #  Reward per episodes
    try:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(stats.episode_total_reward) + 1), stats.episode_total_reward)

        ax.set(xlabel='Episodes', ylabel='Total reward',
               title='Evolution of total reward per episode')
        ax.grid()

        fig.savefig(IMAGE_PATH + "reward_per_episode.png")
        plt.show()
    except:
        print("Error while plotting reward_per_episode")

    #  Episode successful
    try:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(stats.episode_succeed) + 1), stats.episode_succeed)

        ax.set(xlabel='Episodes', ylabel='Episode successful',
               title='Episode successful')
        ax.grid()

        fig.savefig(IMAGE_PATH + "successful_episode.png")
        plt.show()
    except:
        print("Error while plotting successful_episode")

    #  Random actions
    try:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(stats.episode_random_actions) + 1), stats.episode_random_actions)

        ax.set(xlabel='Episodes', ylabel='Random Actions',
               title='Evolution of Random Actions')
        ax.grid()

        fig.savefig(IMAGE_PATH + "random_actions.png")
        plt.show()
    except:
        print("Error while plotting random_actions")

    #  Random actions
    try:
        # Create a dataset from the statistics gader
        df = pd.DataFrame(np.array(stats.coordinates_matrix))

        # Default heatmap: just a visualization of this square matrix
        p1 = sns.heatmap(df, cmap='coolwarm')
        p1.set(xlabel='Y coordinates', ylabel='X coordinates',
                   title='Robot movement heatmap')
        plt.show()
    except:
        print("Error while plotting robot heatmap")
