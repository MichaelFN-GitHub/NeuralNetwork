import gym
import numpy as np
import matplotlib.pyplot as plt
import torch

env = gym.make("CartPole-v1")

num_inputs = 4
num_actions = 2

model = torch.nn.Sequential(
    torch.nn.Linear(num_inputs, 128, bias=False, dtype=torch.float32),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_actions, bias=False, dtype=torch.float32),
    torch.nn.Softmax(dim=1)
)


def run_episode(max_steps_per_episode=10000, render=False):
    states, actions, probs, rewards = [], [], [], []
    state, _ = env.reset()
    # print("Initial state shape:", state)  # Check the initial shape of state
    state = np.array(state)  # Ensure state is a NumPy array
    for _ in range(max_steps_per_episode):
        if render:
            env.render()
        state_expanded = np.expand_dims(state, 0)
        tensor_state = torch.from_numpy(state_expanded).float()  # Convert to tensor and specify the data type
        action_probs = model(tensor_state)[0]
        action = np.random.choice(num_actions, p=np.squeeze(action_probs.detach().numpy()))
        nstate, reward, done, info, _ = env.step(action)  # Correct: unpacking 4 values
        if done:
            break
        states.append(state)
        actions.append(action)
        probs.append(action_probs.detach().numpy())
        rewards.append(reward)
        state = np.array(nstate)  # Ensure nstate is a NumPy array
    return np.vstack(states), np.vstack(actions), np.vstack(probs), np.vstack(rewards)


s, a, p, r = run_episode()
print(f"Total reward: {np.sum(r)}")

eps = 0.0001

def discounted_rewards(rewards,gamma=0.99,normalize=True):
    ret = []
    s = 0
    for r in rewards[::-1]:
        s = r + gamma * s
        ret.insert(0, s)
    if normalize:
        ret = (ret-np.mean(ret))/(np.std(ret)+eps)
    return ret

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_on_batch(x, y):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    optimizer.zero_grad()
    predictions = model(x)
    loss = -torch.mean(torch.log(predictions) * y)
    loss.backward()
    optimizer.step()
    return loss

alpha = 1e-4

history = []
for epoch in range(300):
    states, actions, probs, rewards = run_episode()
    one_hot_actions = np.eye(2)[actions.T][0]
    gradients = one_hot_actions-probs
    dr = discounted_rewards(rewards)
    gradients *= dr
    target = alpha*np.vstack([gradients])+probs
    train_on_batch(states,target)
    history.append(np.sum(rewards))
    if epoch%100==0:
        print(f"{epoch} -> {np.sum(rewards)}")

plt.plot(history)

_ = run_episode(render=True)


