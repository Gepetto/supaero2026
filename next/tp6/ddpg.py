# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer

# HYPERPARAM
SEED = 1
ENV_ID = "Pendulum-v1"
NUM_ENVS = 1
LEARNING_RATE = 3e-4
BUFFER_SIZE = int(1e6)
TOTAL_TIMESTEPS = 1000000
TRAIN_FREQUENCY = 2
BATCH_SIZE = 256
GAMMA = 0.99
TARGET_NETWORK_FREQUENCY = 2
TAU = 0.005
EXPLORATION_NOISE = 0.1


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


# TRY NOT TO MODIFY: seeding
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# env setup
envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, SEED, 0, False, "TP6")])
assert isinstance(
    envs.single_action_space, gym.spaces.Box
), "only continuous action space is supported"

actor = Actor(envs)
qf1 = QNetwork(envs)
qf1_target = QNetwork(envs)
target_actor = Actor(envs)
target_actor.load_state_dict(actor.state_dict())
qf1_target.load_state_dict(qf1.state_dict())
q_optimizer = torch.optim.Adam(list(qf1.parameters()), lr=LEARNING_RATE)
actor_optimizer = torch.optim.Adam(list(actor.parameters()), lr=LEARNING_RATE)

envs.single_observation_space.dtype = np.float32
rb = ReplayBuffer(
    BUFFER_SIZE,
    envs.single_observation_space,
    envs.single_action_space,
    "cpu",
    handle_timeout_termination=False,
)

# start the game
obs, _ = envs.reset(seed=SEED)
h_rwd = []

for global_step in range(TOTAL_TIMESTEPS):
    # ALGO LOGIC: put action logic here
    if global_step < TOTAL_TIMESTEPS // 40:
        actions = np.array(
            [envs.single_action_space.sample() for _ in range(envs.num_envs)]
        )
    else:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs))
            actions += torch.normal(0, actor.action_scale * EXPLORATION_NOISE)
            actions = actions.numpy().clip(
                envs.single_action_space.low, envs.single_action_space.high
            )

    # execute the game
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    # record rewards for plotting purposes
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            h_rwd.append(info["episode"]["r"])

    # save data to replay buffer; handle `final_observation`
    real_next_obs = next_obs.copy()
    for idx, trunc in enumerate(truncations):
        if trunc:
            real_next_obs[idx] = infos["final_observation"][idx]
    rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

    # CRUCIAL step easy to overlook
    obs = next_obs

    # ALGO LOGIC: training.
    if global_step > TOTAL_TIMESTEPS // 40:
        data = rb.sample(BATCH_SIZE)
        with torch.no_grad():
            next_state_actions = target_actor(data.next_observations)
            qf1_next_target = qf1_target(data.next_observations, next_state_actions)
            next_q_value = data.rewards.flatten() + (
                1 - data.dones.flatten()
            ) * GAMMA * (qf1_next_target).view(-1)

        qf1_a_values = qf1(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

        # optimize the model
        q_optimizer.zero_grad()
        qf1_loss.backward()
        q_optimizer.step()

        if global_step % TRAIN_FREQUENCY == 0:
            actor_loss = -qf1(data.observations, actor(data.observations)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

        # update the target network
        if global_step % TARGET_NETWORK_FREQUENCY == 0:
            for param, target_param in zip(
                actor.parameters(), target_actor.parameters()
            ):
                target_param.data.copy_(
                    TAU * param.data + (1 - TAU) * target_param.data
                )
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(
                    TAU * param.data + (1 - TAU) * target_param.data
                )

envs.close()
