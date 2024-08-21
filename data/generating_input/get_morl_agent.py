import numpy as np
from copy import deepcopy

import torch as th



class MO_Agent:
    
    def __init__(self, agent, weights, previous_state = None):
        self.agent = agent
        self.weights = weights
        self.previous_state = previous_state


    @th.no_grad()
    def get_q_vals(self, obs: np.ndarray) -> np.array:
        """Calculates Q-values given an observation and weight.

        Args:
            obs: observation

        Returns: the array of Q-values.
        """
        obs = th.as_tensor(obs).float().to(self.agent.device)
        w = th.as_tensor(self.weights).float().to(self.agent.device)
        
        #With GPIPD, there are multiple neural nets, which predict the q-values
        # The action is selected based on the highest q-value across all NNs (9)
        #Here, I take mx q value across all NNs for each action
        if self.agent.__class__.__name__ == 'GPIPD':
            M = th.stack(self.agent.weight_support)
            obs_m = obs.repeat(M.size(0), *(1 for _ in range(obs.dim())))
            q_values = self.agent.q_nets[0](obs_m, M)
            scalarized_q_values = th.einsum("r,bar->ba", w, q_values)
            scalarized_q_values = th.max(scalarized_q_values, 0).values

        else:
            q_values = self.agent.q_net(obs, w)
            scalarized_q_values = th.einsum("r,bar->ba", w, q_values)
        return scalarized_q_values.detach().numpy().flatten()
    
    def eval(self, obs) -> int:
        """Returns the action with maximum Q-value assuming the agent object 
            has an eval method (only compatible with morl-baselines).
        
        Args:
            obs: observation
            
        Returns a scalar value - action"""

        return self.agent.eval(obs, self.weights)
    
    def epsilon_greedy(self, obs) -> int:
        """Epsilon-greedily select an action given an observation and weight.
            assuming the agent object has an act method (only compatible with morl-baselines).

        Args:
            obs: observation

        Returns: an integer representing the action to take.
        """
        obs = th.as_tensor(obs).float().to(self.agent.device)
        w = th.as_tensor(self.weights).float().to(self.agent.device)

        return self.agent.act(obs, w)


    def get_features(self, env) -> dict:
        """Gets environment-specific features from the environment - 
            only compatible with highway env for now
            
        Args
            env: environment of the first agent
            
        Returns: diciotnary with position of the agent"""
        return {"position": deepcopy(env.road.vehicles[0].destination)}

    def pre_disagreement(self, env):
        """Copies the environemnt image before the disagreement"""
        return deepcopy(env)

    def post_disagreement(self, agent1, pre_params=None):
        env = pre_params
        self.previous_state = agent1.previous_state
        return env


    def da_states_functionality(self, trace, params=None):
        trace.a2_max_q_val = max(max(params), trace.a2_max_q_val)
        trace.a2_min_q_val = min(min(params), trace.a2_min_q_val)

    def update_trace(self, trace, t, states, scores):
        a2_s_a_values = [x.action_values for x in states]
        a1_values_for_a2_states = [
            self.get_q_vals(x.state) for x in states]
        trace.a2_s_a_values.append(a2_s_a_values)
        trace.a2_trajectories.append(states)
        trace.a2_rewards.append(scores)
        trace.disagreement_indexes.append(t)
        trace.a1_values_for_a2_states.append(a1_values_for_a2_states)




