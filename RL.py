from stable_baselines3 import PPO
import gym
from gym import spaces
import numpy as np
import ctypes as ct
from utils import run_iter_func

step_size = 0.001

movements = [[step_size, 0, 0],
             [0, step_size, 0],
             [0, 0, step_size],
             [step_size, step_size, 0],
             [step_size, 0, step_size],
             [step_size, step_size, 0],
             [step_size, step_size, step_size]]


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
        self.state = np.zeros(9, dtype=np.float32)
        self.function = Function()
        self.max_reward = 0
        self.max_reward_state = 0
        self.max_reward_action = 0

    def step(self, action):
        # Apply action
        # Here you would need to define how an action modifies your environment's state

        # Check for collision and calculate reward
        movement = movements[action[0]] + movements[action[1]]
        reward = self.calculate_reward(movement)

        # Check if the episode is done (you'll need to define the criteria)
        if reward > self.max_reward:
            self.max_reward = reward
            self.max_reward_state = self.state.copy()  # Make a copy of the state
            self.max_reward_action = action.copy()  # Make a copy of the action

        r_history.append(reward)

        for i in range(6):
            self.state[i] = movement[i] + self.state[i]

        return self.state, reward, False, {}

    def reset(self):
        # Reset the environment state
        self.state = np.zeros(9, dtype=np.float32)
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
        return execution_time / 3000000


from gym.envs.registration import register

register(
    id='CollisionAvoidanceEnv-v0',
    entry_point='RL:CollisionAvoidanceEnv',
)

env = gym.make('CollisionAvoidanceEnv-v0')

# Initialize the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
total_timesteps = 25000
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save("ppo_cartpole")

# Load the trained model
model = PPO.load("ppo_cartpole")
obs = env.reset()

for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()

print("Maximum Reward:", env.max_reward)
print("State leading to max reward:", env.max_reward_state)
print("Action leading to max reward:", env.max_reward_action)
env.close()

import matplotlib.pyplot as plt

x = list(range(len(r_history)))
y = r_history
plt.plot(x, y)
plt.savefig("graph.png")
