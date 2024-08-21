
from get_morl_agent import MO_Agent
import numpy as np
import torch
import numpy as np
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
import pickle




class State:
    def __init__(self, episode, timestep, state_rep, frame, action, importance):
        self.episode = episode
        self.timestep = timestep
        self.state_rep = state_rep
        self.frame = frame
        self.action = action
        self.importance = importance


class Trajectory:
    def __init__(self, importance, trajectory, episode, second_highest_imp):
        self.importance = importance
        self.trajectory = trajectory
        self.episode = episode
        self.second_highest = second_highest_imp


class Highlights:

    '''Class representing highlights, for given agent with pareto-weights, return highlights for the given environment. 
    Also calculates the distances between the states'''
    def __init__(self, agent, weights, num_episodes, env):
        self.agent = agent #MORL agent
        self.weights = weights #np array with pareto-optimal wieghts for the agent (one weight combination for each solution in the solution set)
        self.num_episodes = num_episodes 
        self.env = env #environment to execute the agent

    def get_important_trajectories(self, weight):
        agent = MO_Agent(self.agent, weight)

        trajectories = []
        for e in range(self.num_episodes):
            print(f'Running Episode number: {e}')
            trajectory = []
            #Initial state
            obs, _ = self.env.reset()
            frame = self.env.render()
            t, r, done, infos = 0, 0, False, {}

            #check importance of that state
            q_vals = agent.get_q_vals(obs)
            importance = max(q_vals) - min(q_vals)
            

            while not done:
                a = agent.eval(obs)
                state = State(e, t, obs, frame, a, importance)
                trajectory.append(state)
                obs, r, terminated, truncated, info = self.env.step(a)
                frame = self.env.render()
                if terminated or truncated:
                    done = True
                q_vals = agent.get_q_vals(obs) 
                importance = max(q_vals) - min(q_vals)
                t = t+1
            
            importances = [state.importance for state in trajectory]
            highest_imp = max(importances)
            most_imp_state = importances.index(highest_imp)
            highlight_trajectory = trajectory[most_imp_state-2:most_imp_state+3]
            second_highest = sorted(set(importances), reverse=True)[1]
            second_highest_state = importances.index(second_highest)
            trajectories.append(Trajectory(highest_imp, highlight_trajectory, e, [second_highest, second_highest_state]))

        return trajectories


    def get_top_n(self, trajectories, n=5):

        indexes = sorted(range(len(trajectories)), key=lambda i: [trajectory.importance for trajectory in trajectories][i], reverse=True)[:n]
        top_n_trajectories = [trajectories[idx] for idx in indexes]
        return top_n_trajectories



    def get_highlights_morl(self, n = 5):
        '''For given sets of pareto weights and a mo agent outputs a set of n highlights for each policy'''

        highlights_all = {}

        for i, weight in enumerate(self.weights):
            trajectories = self.get_important_trajectories(weight)
            top = self.get_top_n(trajectories, n)
            highlights_all[f'Policy_{i}'] = top
        
        return highlights_all

    def frobenius_norm(self, highlights, save = False):
        distance_matrix = np.empty((len(highlights),len(highlights)))

        for i, policy_i in enumerate(highlights.keys()):

            imp_states_i = []
            for high_policy in highlights[policy_i]:
                state = high_policy.trajectory[2].state_rep
                imp_states_i.append(state)
            imp_states_i = np.array(imp_states_i).T

            for j, policy_j in enumerate(highlights.keys()):
                imp_states_j = []
                for high_policy in highlights[policy_j]:
                    state = high_policy.trajectory[2].state_rep
                    imp_states_j.append(state)
                imp_states_j = np.array(imp_states_j).T

                frobenius_distances = np.linalg.norm(imp_states_i - imp_states_j, 'fro')
                distance_matrix[i][j] = frobenius_distances
            if save:
                pickle.dump(distance_matrix, open('frobenius_distance_matrix_states.pkl', 'wb'))

        return distance_matrix



