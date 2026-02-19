"""
Example of Q-table learning with a simple discretized 1-pendulum environment
using a linear Q network.
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from myenvs import EnvMountainCarFullyDiscrete

### --- Random seed
RANDOM_SEED = 1188  # int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

### --- Hyper paramaters
NEPISODES = 20000  # Number of training episodes
NSTEPS = 50  # Max episode length
LEARNING_RATE = 0.6  # Step length in optimizer
DECAY_RATE = 0.99  # Discount factor

### --- Environment

env = EnvMountainCarFullyDiscrete()  # render_mode='human')

### --- Q-value networks


class QValueNetwork(nn.Module):
    def __init__(self, env, learning_rate=LEARNING_RATE):
        assert isinstance(env.observation_space, gym.spaces.discrete.Discrete)
        assert isinstance(env.action_space, gym.spaces.discrete.Discrete)

        super(QValueNetwork, self).__init__()

        # Linear layer from input size NX to output size NU
        self.fc = nn.Linear(env.observation_space.n, env.action_space.n)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, x):
        qvalue = self.fc(x)
        return qvalue

    def predict_action(self, x, noise=0):
        qvalue = self.forward(x)
        if noise != 0:
            qvalue += torch.randn(qvalue.shape) * noise
        u = torch.argmax(qvalue, dim=1)
        return u

    def update(self, x, qref):
        self.optimizer.zero_grad()
        qvalue = self.forward(x)
        loss = F.mse_loss(qvalue, qref)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def plot(self):
        # ### Plot Q-Table as value function pos-vs-vel
        def indexes_to_continuous_states(indexes):
            return [
                env.undiscretize_state(env.index_to_discrete_state(s))
                for s in indexes
                if env.is_index_in_range(s)
            ]

        Q = self.fc.weight.T
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        qvs = indexes_to_continuous_states(range(env.observation_space.n))
        values = [
            float(torch.max(Q[s, :]))
            for s in range(env.observation_space.n)
            if env.is_index_in_range(s)
        ]
        vplot = axs[0].scatter([qv[0] for qv in qvs], [qv[1] for qv in qvs], c=values)
        plt.colorbar(vplot, ax=axs[0])
        policies = [
            int(torch.argmax(Q[s, :]))
            for s in range(env.observation_space.n)
            if env.is_index_in_range(s)
        ]
        axs[1].scatter([qv[0] for qv in qvs], [qv[1] for qv in qvs], c=policies)


qvalue = QValueNetwork(env)


def one_hot(ix, env):
    """Return a one-hot encoded tensor.

    - ix: index or batch of indices
    - n: number of classes (size of the one-hot vector)
    """
    ix = torch.tensor(ix).long()
    if ix.dim() == 0:  # If a single index, add batch dimension
        ix = ix.unsqueeze(0)
    return F.one_hot(ix, num_classes=env.observation_space.n).to(torch.float32)


### --- History of search
h_rwd = []  # Learning history (for plot).

### --- Training
for episode in range(1, NEPISODES):
    x, _ = env.reset()
    rsum = 0.0

    for step in range(NSTEPS - 1):
        u = int(
            qvalue.predict_action(one_hot(x, env), noise=1 / episode)
        )  # ... with noise
        x2, reward, done, trunc, info = env.step(u)

        # Compute reference Q-value at state x respecting HJB
        # Q2 = sess.run(qvalue.qvalue, feed_dict={qvalue.x: onhot(x2)})
        qnext = qvalue(one_hot(x2, env))
        qref = qvalue(one_hot(x, env))
        qref[0, u] = reward + DECAY_RATE * torch.max(qnext)

        # Update Q-table to better fit HJB
        # sess.run(qvalue.optim, feed_dict={qvalue.x: onehot(x), qvalue.qref: Qref})
        qvalue.update(x=one_hot(x, env), qref=qref)

        rsum += reward
        x = x2
        if done or trunc:
            break

    h_rwd.append(rsum)
    if not episode % 20:
        print("Episode #%d done with %d sucess" % (episode, sum(h_rwd[-20:])))

### DISPLAY

# ### Render a trial


def rendertrial(env, maxiter=100):
    """Roll-out from random state using greedy policy."""
    s, _ = env.reset()
    traj = [s]
    for i in range(maxiter):
        a = int(qvalue.predict_action(one_hot(s, env)))
        s, r, done, trunc, _ = env.step(a)
        traj.append(s)
        if done or trunc:
            break
    return traj


envrender = EnvMountainCarFullyDiscrete(render_mode="human")
traj = rendertrial(envrender, 10)

plt.ion()

# ### Plot the learning curve
print("Total rate of success: %.3f" % (sum(h_rwd) / NEPISODES))
plt.plot(np.cumsum(h_rwd) / range(1, len(h_rwd) + 1))

# ### Plot Q-Table as value function pos-vs-vel
qvalue.plot()
