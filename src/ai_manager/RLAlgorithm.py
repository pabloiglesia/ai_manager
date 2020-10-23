import math
import random
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Environment import Environment
from ImageController import ImageController

import torchvision.transforms as T
from PIL import Image

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display


class RLAlgorithm:

    def __init__(self, batch_size=256, gamma=0.999, eps_start=1, eps_end=0.01, eps_decay=0.001, target_update=10,
                 memory_size=100000, lr=0.001, num_episodes=1000):

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.memory_size = memory_size
        self.lr = lr
        self.num_episodes = num_episodes

        self.current_state = None
        self.current_action = None
        self.episode_done = False
        self.episode = 0
        self.episode_steps = []
        self.episode_secced = []

        self.State = namedtuple(  # Replay Memory Experience namedtuple
            'State',
            ('coordinate_x', 'coordinate_y', 'object_gripped', 'image_raw')
        )
        self.Experience = namedtuple(  # Replay Memory Experience namedtuple
            'Experience',
            ('state', 'action', 'next_state', 'reward')
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.em = self.EnvManager(self)
        self.strategy = self.EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)
        self.agent = self.Agent(self)
        self.memory = self.ReplayMemory(self.memory_size)

        self.policy_net = self.DQN(self.em.image_tensor_size,
                                   self.em.num_actions_available()).to(self.device)
        self.target_net = self.DQN(self.em.image_tensor_size,
                                   self.em.num_actions_available()).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

    class Agent:
        def __init__(self, rl_algorithm):
            self.current_step = 0
            self.strategy = rl_algorithm.strategy
            self.num_actions = rl_algorithm.em.num_actions_available()
            self.device = rl_algorithm.device
            self.rl_algorithm = rl_algorithm

        def select_action(self, state, policy_net):
            if self.rl_algorithm.episode_done:
                self.rl_algorithm.episode_done = False
                self.rl_algorithm.episode += 1
                # TODO: if self.rl_algorithm.episode >= self.rl_algorithm.num_episodes:
                return 'random_state'
            else:
                rate = self.strategy.get_exploration_rate(self.current_step)
                self.current_step += 1

                if rate > random.random():
                    action = random.randrange(self.num_actions)
                else:
                    print(
                        "______________________________________________ ACCION CALCULADA _______________________________________")
                    # input_list = np.append(state.image_raw, state.coordinate_x)
                    # input_list = np.append(input_list, state.coordinate_y)

                    with torch.no_grad():
                        action = policy_net(state.image_raw).argmax(dim=1).to(self.device)  # exploit
                self.rl_algorithm.current_action = self.rl_algorithm.em.actions[action]
                return self.rl_algorithm.current_action

    class DQN(nn.Module):
        def __init__(self, image_tensor_size, num_actions, kernel_size=5, stride=2):

            super(RLAlgorithm.DQN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=stride)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride)
            self.bn3 = nn.BatchNorm2d(32)

            # Number of Linear input connections depends on output of conv2d layers
            # and therefore the input image size, so compute it.
            def conv2d_size_out(size, kernel_size=kernel_size, stride=stride):
                return (size - (kernel_size - 1) - 1) // stride + 1

            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_tensor_size(2))))
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_tensor_size(3))))

            # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(img_width)))
            # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(img_height)))
            linear_input_size = convw * convh * 32
            self.head = nn.Linear(linear_input_size, num_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            linear_input = x.view(x.size(0), -1)
            return self.head(linear_input)

    # class DQN(nn.Module):
    #     def __init__(self, img_height, img_width, num_actions):
    #         super(RLAlgorithm.DQN, self).__init__()
    #
    #         self.fc1 = nn.Linear(in_features=(img_height * img_width) + 2, out_features=24)
    #         self.fc2 = nn.Linear(in_features=24, out_features=32)
    #         self.out = nn.Linear(in_features=32, out_features=num_actions)
    #
    #     # Called with either one element to determine next action, or a batch
    #     # during optimization. Returns tensor([[left0exp,right0exp]...]).
    #     def forward(self, t):
    #         # t = t.flatten(start_dim=1)
    #         t = F.relu(self.fc1(t))
    #         t = F.relu(self.fc2(t))
    #         t = self.out(t)
    #         return t

    class EnvManager:
        def __init__(self, rl_manager):
            self.device = rl_manager.device
            self.image_controller = ImageController(capacity=rl_manager.memory_size)
            self.actions = ['north', 'south', 'east', 'west', 'pick']
            self.image_height = None
            self.image_width = None
            self.image_msg = None
            self.image_tensor_size = None
            self.rl_manager = rl_manager
            self.gather_image_state()

        def calculate_reward(self):
            current_coordinates = [self.rl_manager.current_state.coordinate_x,
                                   self.rl_manager.current_state.coordinate_x]
            object_gripped = self.rl_manager.current_state.object_gripped
            if Environment.is_terminal_state(current_coordinates, object_gripped):
                self.rl_manager.episode_done = True
                if object_gripped:
                    rospy.loginfo("Episode ended: Object gripped!")
                    # TODO: Save image as success
                    return 10
                else:
                    rospy.loginfo("Episode ended: Environment limits reached!")
                    return -10
            else:
                if self.rl_manager.current_action == 'pick':
                    # TODO: Save image as failure
                    return -10
                else:
                    return -1

        def num_actions_available(self):
            return len(self.actions)

        def gather_image_state(self):
            """
            This method gather information about the ur3 robot state by reading several ROS topics
            :param img_controller: class which will allow us to save sensor_msgs images
            """
            rgb_array, self.image_width, self.image_height = self.image_controller.get_image()  # We retrieve state image
            # self.image_msg = rgb_array
            # self.image_msg = rgb_array.flatten()
            self.image_msg = self.get_processed_screen(rgb_array)
            self.image_tensor_size = self.image_msg.size

        def get_processed_screen(self, image_raw):
            resize = T.Compose([T.ToPILImage(),
                                T.Resize(40, interpolation=Image.CUBIC),
                                T.ToTensor()])
            image_raw = torch.from_numpy(image_raw)
            return resize(image_raw).unsqueeze(0).to(self.device)
            # Resize, and add a batch dimension (BCHW)
            # list = np.concatenate(image_raw, axis=0).tolist()
            # return [value for row in list for value in row]

    class EpsilonGreedyStrategy:
        def __init__(self, start, end, decay):
            self.start = start
            self.end = end
            self.decay = decay

        def get_exploration_rate(self, current_step):
            return self.end + (self.start - self.end) * \
                   math.exp(-1. * current_step * self.decay)

    class QValues:
        @staticmethod
        def get_current(policy_net, states, actions):
            return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

        @staticmethod
        def get_next(target_net, next_states):
            final_state_locations = next_states.flatten(start_dim=1) \
                .max(dim=1)[0].eq(0).type(torch.bool)
            non_final_state_locations = (final_state_locations == False)
            non_final_states = next_states[non_final_state_locations]
            batch_size = next_states.shape[0]
            values = torch.zeros(batch_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
            return values

    class ReplayMemory:
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.push_count = 0

        def push(self, experience):
            if len(self.memory) < self.capacity:
                self.memory.append(experience)
            else:
                self.memory[self.push_count % self.capacity] = experience
            self.push_count += 1

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def can_provide_sample(self, batch_size):
            return len(self.memory) >= batch_size

    def extract_tensors(self, experiences):
        # Convert batch of Experiences to Experience of batches
        batch = self.Experience(*zip(*experiences))

        print(batch.reward)

        t1 = torch.cat(batch.state)
        t3 = torch.cat(batch.reward)
        t2 = torch.cat(batch.action)
        t4 = torch.cat(batch.next_state)

        return t1, t2, t3, t4

    @staticmethod
    def get_success_percentage(period, values):
        values = torch.tensor(values, dtype=torch.float)
        if len(values) >= period:
            moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
            moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
            return moving_avg.numpy()
        else:
            moving_avg = torch.zeros(len(values))
            return moving_avg.numpy()

    def plot(self, values, success_percentage_period):
        int_values = [1 if value else 0 for value in values]

        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(int_values)

        success_percentage = self.get_success_percentage(success_percentage_period, int_values)
        plt.plot(success_percentage)
        plt.pause(0.001)
        print("Episode", len(int_values), "\n", success_percentage_period, "episode moving avg: ",
              success_percentage[-1])
        if is_ipython: display.clear_output(wait=True)

    def train_net(self):
        if self.memory.can_provide_sample(self.batch_size):
            print("Entrenando red")
            experiences = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states = self.extract_tensors(experiences)

            current_q_values = self.QValues.get_current(self.policy_net, states, actions)
            next_q_values = self.QValues.get_next(self.target_net, next_states)
            target_q_values = (next_q_values * self.gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            self.optimizer.zero_grad()  # set all the gradients to 0 (initialization) so that we dont accumulate gradient throughout all the backpropagation
            loss.backward()  # Compute the gradient of the loss with respect to all the weights and biases in the policy net
            self.optimizer.step()  # Updates the weights and biases with the gradients computed
            print("Fin de entrenamiento")

        if self.episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
