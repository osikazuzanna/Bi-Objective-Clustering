import mo_gymnasium as mo_gym
import pickle
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from full_highlights import Highlights
from gymnasium.wrappers import FlattenObservation
import numpy as np
from scipy.spatial import distance


def create_distance_dict(obj_values, distance_matrix_states, save = False, env_name = None):
    '''calculates euclidean distances for the solutions in the objective space, give objective values
    and stacks them with distances caluclated bsaed on highlights after normalizing values in both spaces
    '''
    obj_values = np.vstack(obj_values)

    distance_obj = distance.cdist(obj_values, obj_values, 'euclidean')

    normalized_obj_dist = (distance_obj - distance_obj.min().min()) / (distance_obj.max().max() - distance_obj.min().min())
    normalized_policy_dist = (distance_matrix_states - distance_matrix_states.min().min()) / (distance_matrix_states.max().max() - distance_matrix_states.min().min())
    distances = {'objectives': normalized_obj_dist, 'policies': normalized_policy_dist}
    if save:
        pickle.dump(distances, open(f'distances/distances_{env_name}.pkl', 'wb'))
    return distances

def main():
    environments = ['mo-reacher-v4', 'mo-highway-v0', 'mo-lunar-lander-v2', 'minecart-v0']
    agents = ['gpi_pd_discrete-mo-reacher-v4.tar', 'gpi_pd_mo-highway.tar', 'GPI-PD-mo-lunar.tar', 'GPI-PD-minecart.tar']


    for i in range(len(environments)):
        #for every environment, we load the agent, and pareto-weights
        env = mo_gym.make(environments[i], render_mode = 'rgb_array')
        if env == 'mo-highway':
            env = FlattenObservation(env)
        agent_gpipd = GPIPD(env)

        agent_gpipd.load(path=agents[i])  

        #Load the pareto weights
        weights_gpipd = pickle.load(open(f'data/generating_input/solution_sets/pareto_weights_{environments[i]}.pkl', 'rb'))

        #arguments of the funstion
        num_episodes = 50

        #running highlights to create trajectories
        highlights = Highlights(agent_gpipd, weights_gpipd, num_episodes, env)
        highlights_trajectories = highlights.get_highlights_morl()

        #calculating frobenius distances between the trajectories highlighted by the algorithm
        frobenius_distance = Highlights.frobenius_norm(highlights_trajectories)
        #saving distances
        pickle.dump(frobenius_distance, open(f'data/generating_input/highlights/{environments[i]}_frobenius_distance_matrix_states.pkl', 'wb'))

        #load objective values for the pareto weights
        obj_values = pickle.load(open(f'data/generating_input/solution_sets/pareto_front_{environments[i]}.pkl', 'rb'))

        #create a distance dictionary with euclidean distances and frobenius distances, normalizsed
        distance_dict = create_distance_dict(obj_values, frobenius_distance)
        #save the dctionary to be used in PAN
        pickle.dump(distance_dict, open(f'data/input/distances_{environments[i]}.pkl', 'wb'))

if __name__ == "__main__":
    main()