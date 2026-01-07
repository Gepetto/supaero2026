"""
Example of Q-table learning with a simple discretized 1-pendulum environment.
-- concerge in 1k  episods with pendulum(1)
-- Converge in 10k episods with cozmo model
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from myenvs import EnvMountainCarFullyDiscrete

### --- Random seed
RANDOM_SEED = 1188  # int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)

### --- Environment

env = EnvMountainCarFullyDiscrete()

### --- Hyper paramaters
NEPISODES = 400000  # Number of training episodes
NSTEPS = 50  # Max episode length
LEARNING_RATE = 0.85  #
DECAY_RATE = 0.99  # Discount factor

assert isinstance(env.observation_space, gym.spaces.discrete.Discrete)
assert isinstance(env.action_space, gym.spaces.discrete.Discrete)
Q = np.zeros([env.observation_space.n, env.action_space.n])  # Q-table initialized to 0

h_rwd = []  # Learning history (for plot).
for episode in range(1, NEPISODES):
    x, _ = env.reset()
    rsum = 0.0
    for steps in range(NSTEPS):
        u = np.argmax(
            Q[x, :] + np.random.randn(1, env.action_space.n) / episode
        )  # Greedy action with noise
        x2, reward, done, trunc, info = env.step(u)

        # Compute reference Q-value at state x respecting HJB
        Qref = reward + DECAY_RATE * np.max(Q[x2, :])

        # Update Q-Table to better fit HJB
        Q[x, u] += LEARNING_RATE * (Qref - Q[x, u])
        x = x2
        rsum += reward
        if done or trunc:
            break

    h_rwd.append(rsum)
    if not episode % 20:
        print(
            "Episode #%d done with average cost %.2f" % (episode, sum(h_rwd[-20:]) / 20)
        )

### DISPLAY

# ### Render a trial


def rendertrial(env, maxiter=100):
    """Roll-out from random state using greedy policy."""
    s, _ = env.reset()
    traj = [s]
    for i in range(maxiter):
        a = np.argmax(Q[s, :])
        s, r, done, _ = env.step(a)
        traj.append(s)
        if done:
            break
    return traj


envrender = EnvMountainCarFullyDiscrete(render_mode="human")
traj = rendertrial(env)

plt.ion()

# ### Plot the learning curve
print("Total rate of success: %.3f" % (sum(h_rwd) / NEPISODES))
plt.plot(np.cumsum(h_rwd) / range(1, len(h_rwd) + 1))


# ### Plot Q-Table as value function pos-vs-vel
def indexes_to_continuous_states(indexes):
    return [
        env.undiscretize_state(env.index_to_discrete_state(s))
        for s in indexes
        if env.is_index_in_range(s)
    ]


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
qvs = indexes_to_continuous_states(range(env.observation_space.n))
values = [
    np.max(Q[s, :]) for s in range(env.observation_space.n) if env.is_index_in_range(s)
]
vplot = axs[0].scatter([qv[0] for qv in qvs], [qv[1] for qv in qvs], c=values)
plt.colorbar(vplot, ax=axs[0])
policies = [
    np.argmax(Q[s, :])
    for s in range(env.observation_space.n)
    if env.is_index_in_range(s)
]
axs[1].scatter([qv[0] for qv in qvs], [qv[1] for qv in qvs], c=policies)


qv_traj = indexes_to_continuous_states(traj)
plt.plot([qv[0] for qv in qv_traj], [qv[1] for qv in qv_traj], "-")
