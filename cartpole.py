import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np


# Define a simple neural network with one hidden layer
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # CartPole environment has 4 input features
        self.fc2 = nn.Linear(128, 2)  # CartPole actions: left or right

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x


# Initialize environment, network, and optimizer
env = gym.make('CartPole-v1')
policy_net = PolicyNetwork()
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)


def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy_net(state)
    # We will use probabilities to select actions
    action = torch.multinomial(probs, 1).item()
    return action


def train(episodes):
    for episode in range(1, episodes+1):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False

        # Collect samples for this episode
        while not done:
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            log_prob = torch.log(policy_net(torch.from_numpy(state).float())[action])
            log_probs.append(log_prob)
            rewards.append(reward)

        # Update policy after the episode is done
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + (0.99 ** pw) * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        # Perform backprop
        optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        optimizer.step()

        # Print results
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards)}")


# Train the agent
train(500)

# Close the environment
env.close()


# if __name__ == "__main__":
#
#     if False:
#         env = gym.make('CartPole-v0', render_mode='rgb_array')
#
#         episodes = 5
#         for episode in range(1, episodes+1):
#             env.reset()
#             done = False
#             score = 0
#
#             while not done:
#                 action = env.action_space.sample()
#                 obs, reward, done, truncated, info = env.step(action)
#                 score += reward
#
#             print(f"Episode:{episode} Score:{score}")
#         env.close()
#


