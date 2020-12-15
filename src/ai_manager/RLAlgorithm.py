# coding=utf-8
import math
import random
import os
import errno
import sys
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from Environment import Environment
from ImageProcessing.ImageModel import ImageModel
from ImageController import ImageController

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

import pickle

State = namedtuple(  # State information namedtuple
    'State',
    ('coordinate_x', 'coordinate_y', 'object_gripped', 'image_raw')
)

Experience = namedtuple(  # Replay Memory Experience namedtuple
    'Experience',
    ('state', 'coordinates', 'action', 'next_state', 'next_coordinates', 'reward', 'is_final_state')
)


class RLAlgorithm:
    """
    Class used to perform actions related to the RL Algorithm training. It can be initialized with custom parameters or
    with the default ones.

    To perform a Deep Reinforcement Learning training, the following steps have to be followed:

        1. Initialize replay memory capacity.
        2. Initialize the policy network with random weights.
        3. Clone the policy network, and call it the target network.
        4. For each episode:
           1. Initialize the starting state.
            2. For each time step:
                1. Select an action.
                    - Via exploration or exploitation
                2. Execute selected action in an emulator or in Real-life.
                3. Observe reward and next state.
                4. Store experience in replay memory.
                5. Sample random batch from replay memory.
                6. Preprocess states from batch.
                7. Pass batch of preprocessed states to policy network.
                8. Calculate loss between output Q-values and target Q-values.
                    - Requires a pass to the target network for the next state
                9. Gradient descent updates weights in the policy network to minimize loss.
                    - After time steps, weights in the target network are updated to the weights in the policy network.

    """

    def __init__(self, batch_size=32, gamma=0.999, eps_start=1, eps_end=0.01, eps_decay=0.0005, target_update=10,
                 memory_size=2000, lr=0.001, num_episodes=1000):
        """

        :param batch_size: Size of the batch used to train the network in every step
        :param gamma: discount factor used in the Bellman equation
        :param eps_start: Greedy strategy epsilon start (Probability of random choice)
        :param eps_end: Greedy strategy minimum epsilon (Probability of random choice)
        :param eps_decay: Greedy strategy epsilon decay (Probability decay of random choice)
        :param target_update: How frequently, in terms of episodes, target network will update the weights with the
        policy network weights
        :param memory_size: Capacity of the replay memory
        :param lr: Learning rate of the Deep Learning algorithm
        :param num_episodes:  Number of episodes on training
        """

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.memory_size = memory_size
        self.lr = lr
        self.num_episodes = num_episodes

        self.current_state = None  # Robot current state
        self.previous_state = None  # Robot previous state
        self.current_action = None  # Robot current action
        self.current_action_idx = None  # Robot current action Index
        self.episode_done = False  # True if the episode has just ended

        # This tells PyTorch to use a GPU if its available, otherwise use the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Torch devide
        self.em = self.EnvManager(self)  # Robot Environment Manager
        self.strategy = self.EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)  # Greede Strategy
        self.agent = self.Agent(self)  # RL Agent
        self.memory = self.ReplayMemory(self.memory_size)  # Replay Memory
        self.statistics = self.TrainingStatistics()  # Training statistics

        self.policy_net = self.DQN(self.em.image_tensor_size,
                                   self.em.num_actions_available()).to(self.device)  # Policy Q Network
        self.target_net = self.DQN(self.em.image_tensor_size,
                                   self.em.num_actions_available()).to(self.device)  # Target Q Network
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Target net has to be the same as policy network
        self.target_net.eval()  # Target net has to be the same as policy network
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)  # Q Networks optimizer

        print("Device: ", self.device)

    class Agent:
        """
        Class that contains all needed methods to control the agent through the environment and retrieve information of
        Its state
        """

        def __init__(self, rl_algorithm):
            """

            :param self: RLAlgorithm object
            """
            self.strategy = rl_algorithm.strategy  # Greedy Strategy
            self.num_actions = rl_algorithm.em.num_actions_available()  # Num of actions available
            self.device = rl_algorithm.device  # Torch device
            self.rl_algorithm = rl_algorithm

        def select_action(self, state, policy_net):
            """
            Method used to pick the following action of the robot
            Method used to pick the following action of the robot
            :param state: State RLAlgorithm namedtuple with all the information of the current state
            :param policy_net: DQN object used as policy network for the RL algorithm
            :return:
            """
            if self.rl_algorithm.episode_done:  # If the episode has just ended we reset the robot environment
                self.rl_algorithm.episode_done = False  # Put the variable episode_done back to False
                self.rl_algorithm.statistics.new_episode()

                # TODO: if self.self.episode >= self.self.num_episodes:
                self.rl_algorithm.current_action = 'random_state'  # Return random_state to reset the robot position
                self.rl_algorithm.current_action_idx = None
            else:
                rate = self.strategy.get_exploration_rate(self.rl_algorithm.statistics.current_step)  # We get the current epsilon value
                self.rl_algorithm.statistics.new_step()  # Add new steps statistics

                if rate > random.random():  # With a probability = rate we choose a random action (Explore environment)
                    action = random.randrange(self.num_actions)
                    self.rl_algorithm.statistics.random_action()  # Recolecting statistics
                else:  # With a probability = (1 - rate) we Explote the information we already have
                    print("No Random")
                    try:
                        with torch.no_grad():  # We calculate the action using the Policy Q Network
                            action = policy_net(state.image_raw, torch.tensor(
                                [[state.coordinate_x, state.coordinate_y]], device=self.device)).argmax(dim=1).to(self.device)  # exploit
                    except:
                        print("Ha habido un error")

                self.rl_algorithm.current_action = self.rl_algorithm.em.actions[action]
                self.rl_algorithm.current_action_idx = action

            return self.rl_algorithm.current_action  # We return the action as a string, not as int

    class DQN(nn.Module):
        """
        Class to create a Deep Q Learning Neural Network
        """

        def __init__(self, image_tensor_size, num_actions, kernel_size=5, stride=2):
            """

            :param image_tensor_size: Size of the input tensor
            :param num_actions: Number of actions, which is the output of the Neural Network
            :param kernel_size: Kernel Size
            :param stride: Stride parameter
            """
            super(RLAlgorithm.DQN, self).__init__()
            # Different Convolutional Steps to retrieve the image features
            # self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=stride)
            # self.bn1 = nn.BatchNorm2d(16)
            # self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
            # self.bn2 = nn.BatchNorm2d(32)
            # self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride)
            # self.bn3 = nn.BatchNorm2d(32)
            #
            # # Number of Linear input connections depends on output of conv2d layers
            # # and therefore the input image size, so compute it.
            # def conv2d_size_out(size, kernel_size=kernel_size, stride=stride):
            #     return (size - (kernel_size - 1) - 1) // stride + 1
            #
            # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_tensor_size(2))))  # Width of the conv output
            # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_tensor_size(3))))  # Height of the conv output
            # # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(img_width)))
            # # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(img_height)))
            #
            # # Linear Step where we include the image features and the current robot coordinates
            # linear_input_size = (convw * convh * 32) + 2
            self.linear1 = nn.Linear(image_tensor_size, int(image_tensor_size/2))
            self.linear2 = nn.Linear(int(image_tensor_size/2), int(image_tensor_size/4))
            self.linear3 = nn.Linear(int(image_tensor_size/4) + 2, num_actions)
            print(image_tensor_size)
            self.linear = nn.Linear(image_tensor_size + 2, num_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, image_raw, coordinates):
            # features1 = F.relu(self.bn1(self.conv1(image_raw)))
            # features2 = F.relu(self.bn2(self.conv2(features1)))
            # features3 = F.relu(self.bn3(self.conv3(features2)))
            #
            # linear_input = features3.view(features3.size(0), -1)
            # linear_input = torch.cat((linear_input, coordinates), 1)

            output = self.linear1(image_raw)
            output = self.linear2(output)
            output = torch.cat((output, coordinates), 1)
            # output = torch.cat((image_raw, coordinates), 1)
            return self.linear3(output)

    class EnvManager:
        """
        Class used to manage the RL environment. It is used to perform actions such as calculate rewards or retrieve the
        current state of the robot.
        """

        def __init__(self, rl_algorithm, image_size=256):
            """
            Initialization of an object
            :param rl_manager: RLAlgorithm object
            """
            self.device = rl_algorithm.device  # Torch device
            self.image_controller = ImageController()  # ImageController object to manage images
            self.actions = ['north', 'south', 'east', 'west', 'pick']  # Possible actions of the objects
            self.image_height = None  # Retrieved images height
            self.image_width = None  # Retrieved Images Width
            self.image = None  # Current image ROS message
            self.image_tensor = None  # Current image tensor

            self.image_model = ImageModel()
            self.model = self.image_model.load_best_model()
            self.feature_extraction_model = self.image_model.model.load_from_checkpoint(self.image_model.MODEL_CKPT_PATH
                                                                                        + self.model)
            self.image_tensor_size = self.image_model.get_size_features(
                self.feature_extraction_model)  # Size of the image after performing some transformations

            self.rl_algorithm = rl_algorithm
            self.image_size = image_size
            self.gather_image_state()  # Retrieve initial state image

        def calculate_reward(self, previous_image):
            """
            Method used to calculate the reward of the previous action and whether it is a final state or not
            :return: reward, is_final_state
            """
            current_coordinates = [self.rl_algorithm.current_state.coordinate_x,
                                   self.rl_algorithm.current_state.coordinate_y]  # Retrieve robot's current coordinates
            object_gripped = self.rl_algorithm.current_state.object_gripped  # Retrieve if the robot has an object gripped
            if Environment.is_terminal_state(current_coordinates, object_gripped):  # If is a terminal state
                self.rl_algorithm.episode_done = True  # Set the episode_done variable to True to end up the episode
                episode_done = True
                if object_gripped:  # If object_gripped is True, the episode has ended successfully
                    reward = 100
                    self.rl_algorithm.statistics.add_succesful_episode(True)  # Saving episode successful statistic
                    self.rl_algorithm.statistics.increment_picks()    # Increase of the statistics cpunter
                    rospy.loginfo("Episode ended: Object gripped!")
                    self.image_controller.record_image(previous_image, True)  # Saving the falure state image
                else:  # Otherwise the robot has reached the limits of the environment
                    reward = -10
                    self.rl_algorithm.statistics.add_succesful_episode(False)  # Saving episode failure statistic
                    rospy.loginfo("Episode ended: Environment limits reached!")
            else:  # If it is not a Terminal State
                episode_done = False
                if self.rl_algorithm.current_action == 'pick':  # if it is not the first action and action is pick
                    reward = -10
                    self.image_controller.record_image(previous_image, False)  # Saving the falure state image
                    self.rl_algorithm.statistics.increment_picks()  # Increase of the statistics counter
                else:  # otherwise
                    self.rl_algorithm.statistics.fill_coordinates_matrix(current_coordinates)
                    reward = -1

            self.rl_algorithm.statistics.add_reward(reward)  # Add reward to the algorithm statistics
            return reward, episode_done

        def gather_image_state(self):
            """
            This method gather information about the ur3 robot state by reading several ROS topics
            :param img_controller: class which will allow us to save sensor_msgs images
            """
            previous_image = self.image
            self.image, self.image_width, self.image_height = self.image_controller.get_image()  # We retrieve state image
            self.image_tensor = self.get_processed_screen(self.image)
            return previous_image

        def get_processed_screen(self, image):
            """
            Method used to transformate the image to a spected tensor that Neural Network is specting
            :param image_raw: Image
            :return:
            """
            features = self.image_model.evaluate_image(image, self.feature_extraction_model)
            features = torch.from_numpy(features)
            return features.to(self.device)


        def num_actions_available(self):
            """
            Returns the number of actions available
            :return: Number of actions available
            """
            return len(self.actions)

    class EpsilonGreedyStrategy:
        """
        Class used to perform the Epsilon greede strategy
        """

        def __init__(self, start, end, decay):
            """
            Initialization
            :param start: Greedy strategy epsilon start (Probability of random choice)
            :param end: Greedy strategy minimum epsilon (Probability of random choice)
            :param decay: Greedy strategy epsilon decay (Probability decay of random choice)
            """
            self.start = start
            self.end = end
            self.decay = decay

        def get_exploration_rate(self, current_step):
            """
            It calculates the rate depending on the actual step of the execution
            :param current_step: step of the training
            :return:
            """
            return self.end + (self.start - self.end) * \
                   math.exp(-1. * current_step * self.decay)

    class QValues:
        """
        It returns the predicted q-values from the policy_net for the specific state-action pairs that were passed in.
        states and actions are the state-action pairs that were sampled from replay memory.
        """

        @staticmethod
        def get_current(policy_net, states, coordinates, actions):
            """
            With the current state of the policy network, it calculates the q_values of
            :param policy_net: policy network used to decide the actions
            :param states: Set of state images (Preprocessed)
            :param coordinates: Set of robot coordinates
            :param actions: Set of taken actions
            :return:
            """
            return policy_net(states, coordinates).gather(dim=1, index=actions.unsqueeze(-1))

        @staticmethod
        def get_next(target_net, next_states, next_coordinates, is_final_state):
            """
            Calculate the maximum q-value predicted by the target_net among all possible next actions.
            If the action has led to a terminal state, next reward will be 0. If not, it is calculated using the target
            net
            :param target_net: Target Deep Q Network
            :param next_states: Next states images
            :param next_coordinates: Next states coordinates
            :param is_final_state: Tensor indicating whether this action has led to a final state or not.
            :return:
            """
            batch_size = next_states.shape[0]  # The batch size is taken from next_states shape
            # q_values is initialized with a zeros tensor of batch_size and if there is GPU it is loaded to it
            q_values = torch.zeros(batch_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            non_final_state_locations = (is_final_state == False)  # Non final state index locations are calculated
            non_final_states = next_states[non_final_state_locations]  # non final state images
            non_final_coordinates = next_coordinates[non_final_state_locations]  # non final coordinates
            # Max q values of the non final states are calculated using the target net
            q_values[non_final_state_locations] = target_net(non_final_states, non_final_coordinates).max(dim=1)[
                0].detach()
            return q_values

    class ReplayMemory:
        """
        Class used to create a Replay Memory for the RL algorithm
        """

        def __init__(self, capacity):
            """
            Initialization of ReplayMemory
            :param capacity: Capacity of Replay Memory
            """
            self.capacity = capacity
            self.memory = []  # Actual memory. it will be filled with Experience namedtuples
            self.push_count = 0  # will be used to keep track of how many experiences have been added to the memory

        def push(self, experience):
            """
            Method used to fill the Replay Memory with experiences
            :param experience: Experience namedtuple
            :return:
            """
            if len(self.memory) < self.capacity:  # if memory is not full, new experience is appended
                self.memory.append(experience)
            else:  # If its full, we add a new experience and take the oldest out
                self.memory[self.push_count % self.capacity] = experience
            self.push_count += 1  # we increase the memory counter

        def sample(self, batch_size):
            """
            Returns a random sample of experiences
            :param batch_size: Number of randomly sampled experiences returned
            :return: random sample of experiences (Experience namedtuples)
            """
            return random.sample(self.memory, batch_size)

        def can_provide_sample(self, batch_size):
            """
            returns a boolean telling whether or not we can sample from memory. Recall that the size of a sample
            we’ll obtain from memory will be equal to the batch size we use to train our network.
            :param batch_size: Batch size to train the network
            :return: boolean telling whether or not we can sample from memory
            """
            return len(self.memory) >= batch_size

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
            self.episode_random_actions = [0]  # Total reward of each episode
            self.episode_succeed = []  # Array that stores whether each episode has ended successfully or not
            self.coordinates_matrix = self.generate_coordinates_matrix()

        def generate_coordinates_matrix(self):
            x_limit = Environment.X_LENGTH / 2
            y_limit = Environment.Y_LENGTH / 2

            matrix_width = 2 * math.ceil(x_limit/Environment.ACTION_DISTANCE)
            matrix_height = 2 * math.ceil(y_limit/Environment.ACTION_DISTANCE)

            return [([0]*matrix_height) for i in range(matrix_width)]

        def fill_coordinates_matrix(self, coordinates):
            try:
                matrix_width = len(self.coordinates_matrix[0])  # y
                matrix_height = len(self.coordinates_matrix)  # x

                x_idx = int(math.ceil(coordinates[0] / Environment.ACTION_DISTANCE) + (matrix_height / 2) - 1)
                y_idx = int(math.ceil(coordinates[1] / Environment.ACTION_DISTANCE) + (matrix_width / 2) - 1)

                self.coordinates_matrix[x_idx][y_idx] += 1
            except:
                rospy.loginfo('Error while filling coordinates statistics matrix')

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

    def extract_tensors(self, experiences):
        """
        Converts a batch of Experiences to Experience of batches and returns all the elements separately.
        :param experiences: Batch of Experienc objects
        :return: A tuple of each element of a Experience namedtuple
        """
        batch = Experience(*zip(*experiences))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        coordinates = torch.cat(batch.coordinates)
        next_coordinates = torch.cat(batch.next_coordinates)
        is_final_state = torch.cat(batch.is_final_state)

        return states, coordinates, actions, rewards, next_states, next_coordinates, is_final_state

    @staticmethod
    def get_average_steps(period, values):
        values = values[-period:]
        return sum(values) / len(values)

    def plot(self, average_steps_period=20):
        int_values = [1 if value else 0 for value in self.statistics.episode_succeed]

        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(int_values)

        success_percentage = self.get_average_steps(average_steps_period, self.statistics.episode_steps)
        plt.plot(success_percentage)
        plt.pause(0.001)
        if is_ipython: display.clear_output(wait=True)

    def save_training(self, filename='trainings/rl_algorithm.pkl'):

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

        rospy.loginfo("Saving training...")

        filename = create_if_not_exist(filename)

        self.em.image_model = None
        self.em.feature_extraction_model = None

        with open(filename, 'wb+') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        rospy.loginfo("Saving Statistics...")
        print(self.statistics.coordinates_matrix)

        statistics_filename = "{}_stats.pkl".format(filename.split('.pkl')[0])
        statistics_filename = create_if_not_exist(statistics_filename)
        with open(statistics_filename, 'wb+') as output:  # Overwrites any existing file.
            pickle.dump(self.statistics, output, pickle.HIGHEST_PROTOCOL)

        rospy.loginfo("Training saved!")

    @staticmethod
    def recover_training(filename='trainings/rl_algorithm.pkl'):
        current_path = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(current_path, filename)

        try:
            with open(filename, 'rb') as input:
                rl_algorithm = pickle.load(input)
                # rospy.loginfo("Training recovered. Next step will be step number {}"
                #               .format(rl_algorithm.statistics.current_step))
                rl_algorithm.em.image_model = ImageModel()
                rl_algorithm.em.feature_extraction_model = rl_algorithm.em.image_model.model.load_from_checkpoint(
                    rl_algorithm.em.image_model.MODEL_CKPT_PATH + rl_algorithm.em.model)

                return rl_algorithm
        except IOError:
            rospy.loginfo("There is no Training saved. New object has been created")
            return RLAlgorithm()

    def train_net(self):
        """
        Method used to train both the train and target Deep Q Networks. We train the network minimizing the loss between
        the current Q-values of the action-state tuples and the target Q-values. Target Q-values are calculated using
        thew Bellman's equation:

        q*(state, action) = Reward + gamma * max( q*(next_state, next_action) )
        :return:
        """
        # If there are at least as much experiences stored as the batch size
        if self.memory.can_provide_sample(self.batch_size):
            experiences = self.memory.sample(self.batch_size)  # Retrieve the experiences
            states, coordinates, actions, rewards, next_states, next_coordinates, is_final_state = self.extract_tensors(
                experiences)  # We split the batch of experience into different tensors
            # To compute the loss, current_q_values and target_q_values have to be calculated
            current_q_values = self.QValues.get_current(self.policy_net, states, coordinates, actions)
            # next_q_values is the maximum Q-value of each future state
            next_q_values = self.QValues.get_next(self.target_net, next_states, next_coordinates, is_final_state)
            target_q_values = (next_q_values * self.gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))  # Loss is calculated
            self.optimizer.zero_grad()  # set all the gradients to 0 (initialization) so that we don't accumulate
            # gradient throughout all the backpropagation
            loss.backward(retain_graph=True)  # Compute the gradient of the loss with respect to all the weights and biases in the
            # policy net
            self.optimizer.step()  # Updates the weights and biases with the gradients computed

        if self.statistics.episode % self.target_update == 0:  # If target_net has to be updated in this episode
            self.target_net.load_state_dict(self.policy_net.state_dict())  # Target net is updated

    def next_training_step(self, current_coordinates, object_gripped):
        """
        This method implements the Reinforcement Learning algorithm to control the UR3 robot.  As the algorithm is prepared
        to be executed in real life, rewards and final states cannot be received until the action is finished, which is the
        beginning of next loop. Therefore, during an execution of this function, an action will be calculated and the
        previous action, its reward and its final state will be stored in the replay memory.
        :param current_coordinates: Tuple of float indicating current coordinates of the robot
        :param object_gripped: Boolean indicating whether or not ann object has been gripped
        :return: action taken
        """
        self.previous_state = self.current_state  # Previous state information to store in the Replay Memory
        previous_action = self.current_action  # Previous action to store in the Replay Memory
        previous_action_idx = self.current_action_idx  # Previous action index to store in the Replay Memory
        previous_image = self.em.gather_image_state()  # Gathers current state image

        self.current_state = State(current_coordinates[0], current_coordinates[1], object_gripped,
                                   self.em.image_tensor)  # Updates current_state

        # Calculates previous action reward an establish whether the current state is terminal or not
        previous_reward, is_final_state = self.em.calculate_reward(previous_image)
        action = self.agent.select_action(self.current_state,
                                          self.policy_net)  # Calculates action

        # Random_state actions are used just to initialize the environment to a random position, so it is not taken into
        # account while storing state information in the Replay Memory.
        # If previous action was a random_state and it is not the first step of the training
        if previous_action != 'random_state' and self.statistics.current_step > 1:
            self.memory.push(  # Pushing experience to Replay Memory
                Experience(  # Using an Experience namedtuple
                    self.previous_state.image_raw,  # Initial state image
                    torch.tensor([[self.previous_state.coordinate_x, self.previous_state.coordinate_y]],
                                 device=self.device),  # Initial coordinates
                    torch.tensor([previous_action_idx], device=self.device),  # Action taken
                    self.current_state.image_raw,  # Final state image
                    torch.tensor([[self.current_state.coordinate_x,
                                   self.current_state.coordinate_y]],
                                 device=self.device),  # Final coordinates
                    torch.tensor([previous_reward], device=self.device),  # Action reward
                    torch.tensor([is_final_state], device=self.device)  # Episode ended
                ))

            # Logging information
            rospy.loginfo("Step: {}, Episode: {}, Previous reward: {}, Previous action: {}".format(
                self.statistics.current_step - 1,
                self.statistics.episode,
                previous_reward,
                previous_action))

            self.train_net()  # Both policy and target networks gets trained

        return action
