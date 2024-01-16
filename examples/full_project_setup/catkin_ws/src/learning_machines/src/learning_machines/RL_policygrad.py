import numpy as np

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from irobobo_extensions import get_observation, get_reward, get_simulation_done
from irobobo_extensions import POSSIBLE_ACTIONS, do_action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


S_SIZE = 8 # 8 numbers per observation - 8 irs sensors
A_SIZE = 4 # 4 actions - forward, right, left, back


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class PolicyGradientModel:
    def __init__(self, rob, hidden_size):
        self.rob = rob
        self.policy = Policy(S_SIZE, A_SIZE, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)


    def _reinforce(self, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
        scores_deque = deque(maxlen=100)
        scores = []

        for i_episode in range(1, n_training_episodes+1):
            saved_log_probs = []
            rewards = []
            state = get_observation(self.rob)[0]
            for t in range(max_t):
                action, log_prob = policy.act(state)
                saved_log_probs.append(log_prob)
                
                do_action(self.rob, action)
                state, reward, done = get_observation(self.rob)[0], get_reward(self.rob), get_simulation_done(self.rob)
                
                rewards.append(reward)
                if done:
                    break 
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
            
            returns = deque(maxlen=max_t) 
            n_steps = len(rewards) 
            
            for t in range(n_steps)[::-1]:
                disc_return_t = (returns[0] if len(returns)>0 else 0)
                returns.appendleft( gamma*disc_return_t + rewards[t]   )    
                
            eps = np.finfo(np.float32).eps.item()
            
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)
            policy_loss = []
            for log_prob, disc_return in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob * disc_return)
            policy_loss = torch.cat(policy_loss).sum()
            
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            
            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
        return scores

def train(self, n_episodes, max_steps_num, gamma):
    scores = self._reinforce(self.policy, self.optimizer,
                    n_training_episodes=n_episodes,
                      max_t=max_steps_num, gamma=gamma)
    return scores

def predict(self, state):
    predicted_action, proba = self.policy.act(state)
    return POSSIBLE_ACTIONS[predicted_action], proba


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param policy: The Reinforce agent
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    
    for step in range(max_steps):
      action, _ = policy.act(state)
      new_state, reward, done, info = env.step(action)
      total_rewards_ep += reward
        
      if done:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward


# evaluate_agent(eval_env, cartpole_hyperparameters["max_t"], cartpole_hyperparameters["n_evaluation_episodes"], cartpole_policy)