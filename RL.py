import ctypes as ct
from enum import Enum

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from stable_baselines3 import PPO, A2C

from utils import run_iter_func

step_size = 0.001

movements = [[step_size, 0, 0],
             [0, step_size, 0],
             [0, 0, step_size],
             [step_size, step_size, 0],
             [step_size, 0, step_size],
             [step_size, step_size, 0],
             [step_size, step_size, step_size]]


class RL_Algorithm(Enum):
    A2C = 1,
    PPO = 2


algorithm = RL_Algorithm.A2C

class AlgorithmClass:
    def __init__(self, alg):
        self.env = gym.make('CollisionAvoidanceEnv-v0')
        if alg == RL_Algorithm.A2C:
            try:
                self.model = PPO.load("ppo_cartpole")
                print("Saved model doesn't exist. Creating new model.")
            except FileNotFoundError:
                self.model = PPO("MlpPolicy", self.env, verbose=1)
        elif alg == RL_Algorithm.PPO:
            try:
                self.model = A2C.load("a2c_cartpole")
                print("Saved model doesn't exist. Creating new model.")
            except FileNotFoundError:
                self.model = A2C("MlpPolicy", self.env, verbose=1)


class Function:
    def __init__(self):
        mylib = ct.CDLL(
            './libuntitled1.so')

        self.iteration = mylib.start_collision_detection
        # Define the return type of the C function
        self.iteration.restype = ct.c_long

        # Define arguments of the C function
        self.iteration.argtypes = [
            ct.c_int32,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double)
        ]


r_history = []


class CollisionAvoidanceEnv(gym.Env):
    """
    Custom Environment for collision avoidance with high-dimensional state and action spaces.
    """

    def __init__(self):
        super(CollisionAvoidanceEnv, self).__init__()
        # Define action and observation space
        # Each state consists of 9 real numbers (positions of 3 objects)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # Each action consists of 48 real numbers (motions for these objects)
        self.action_space = spaces.MultiDiscrete([7, 7])

        # Initialize state
        self.reset_state()
        self.function = Function()
        self.max_reward = 0
        self.max_reward_state = 0
        self.max_reward_action = 0

        # For early Stop
        self.window_size = 100  # Size of the window for the moving average
        self.improvement_threshold = 0.001  # Threshold for considering improvement significant
        self.patience = 10  # How many checks without significant improvement to wait before stopping

        # Initialize variables for tracking the moving average and stopping criteria
        self.moving_averages = []
        self.no_improvement_count = 0
        self.total_rewards = 0
        self.num_steps = 0
        self.total_steps = 0

    def step(self, action):
        # Apply action
        # Here you would need to define how an action modifies your environment's state

        # Check for collision and calculate reward
        movement = movements[action[0]] + movements[action[1]]
        reward = self.calculate_reward(movement)

        if reward <= 0:
            return self.state, reward, True, {}

        self.total_rewards += reward
        self.num_steps += 1
        self.total_steps += 1

        if self.num_steps >= self.window_size:
            moving_avg = self.total_rewards / self.num_steps
            if len(self.moving_averages) >= 1:
                improvement = moving_avg - self.moving_averages[-1]
                if improvement < self.improvement_threshold:
                    self.no_improvement_count += 1
                r_history.append(moving_avg)
            self.moving_averages.append(moving_avg)
            self.no_improvement_count = 0
            self.total_rewards = 0
            self.num_steps = 0

        done = False

        if self.no_improvement_count >= self.patience:
            done = True
            print(f"Total steps were: {self.total_steps}")
            self.no_improvement_count = 0
            self.total_rewards = 0
            self.num_steps = 0
            self.total_steps = 0

        # Check if the episode is done (you'll need to define the criteria)
        if reward > self.max_reward:
            self.max_reward = reward
            self.max_reward_state = self.state.copy()  # Make a copy of the state
            self.max_reward_action = action.copy()  # Make a copy of the action

        for i in range(6):
            self.state[i] = movement[i] + self.state[i]

        return self.state, reward, done, {}

    def reset(self):
        # Reset the environment state
        self.moving_averages = []
        self.reset_state()
        return self.state

    def render(self, mode='human'):
        # Render the environment to the screen (optional for your case)
        pass

    def close(self):
        # Perform any necessary cleanup
        pass

    def calculate_reward(self, movement):
        # Define how to calculate the reward
        # In your case, it's related to execution time for collision checking
        inputs = [0] * 15
        for i in range(6):
            inputs[i] = movement[i]

        for i in range(6, 15):
            inputs[i - 6] = self.state[i - 6]

        execution_time = run_iter_func(inputs)
        if execution_time == -1 or execution_time == 0:
            return execution_time
        return execution_time / 3000000

    def reset_state(self):
        count = 0
        while True:
            self.state = self.observation_space.sample()
            reward = self.calculate_reward([0] * 6)
            if reward != -1:
                print(f'count for resetting state was: {count}')
                return
            count += 1


from gym.envs.registration import register

register(
    id='CollisionAvoidanceEnv-v0',
    entry_point='RL:CollisionAvoidanceEnv',
)

rl_alg = AlgorithmClass(algorithm)

model = rl_alg.model
env = rl_alg.model

# Train the agent
total_timesteps = 250000
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save("ppo_cartpole")

x = list(range(len(r_history)))
y = r_history
plt.plot(x, y)
plt.savefig("learn_graph.pdf")

r_history = []

# Load the trained model
model = PPO.load("ppo_cartpole")
obs = env.reset()

for i in range(100000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()

print("Maximum Reward:", env.max_reward)
print("State leading to max reward:", env.max_reward_state)
print("Action leading to max reward:", env.max_reward_action)
env.close()

x = list(range(len(r_history)))
y = r_history
plt.plot(x, y)
plt.savefig("test_graph.pdf")
