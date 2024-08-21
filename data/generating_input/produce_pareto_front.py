import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.common.pareto import filter_pareto_dominated
import pickle
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD



def sample_weights(env, num_weights = 1000):
    reward_dim = env.reward_space.shape[0]
    weights = equally_spaced_weights(reward_dim, n=num_weights)
    return weights

def produce_whole_front(weights, agent, env, num_eval_episodes, discounted_vec_return = False):

    '''Given different weights combination, outputs vector of objective values for each.
    The objective vector can be either vector return (default) or discounted vector return'''


    if discounted_vec_return == True:
        current_front = [
        agent.policy_eval(env, weights=ew, num_episodes=num_eval_episodes)[4]
        for ew in weights]
    else:
        current_front = [
        agent.policy_eval(env, weights=ew, num_episodes=num_eval_episodes)[3]
        for ew in weights]
    
    return current_front


def get_pareto_front(whole_front, weights):
    '''Pareto Filters out the whole front and returns the pareto front with the weights'''
    # filter out the dominated ones

    pareto_front = list(filter_pareto_dominated(whole_front))
    

    #identify index for pareto solutions within the whole front
    rows = [[(current_f==pareto_f).all() for current_f in whole_front].index(True) for pareto_f in pareto_front]
    pareto_weights = [weights[w] for w in rows]

    return pareto_front, pareto_weights

def produce_fronts(environments, agents):
    for i, environment in enumerate(environments):

        env = mo_gym.make(environment)
        env = MORecordEpisodeStatistics(env, gamma=0.98)


        reward_dim = env.reward_space.shape[0]
        num_weights = 1000


        num_eval_episodes = 5

        agent = GPIPD(
            env,
            log=False,
            project_name="MORL-Baselines",
            experiment_name="GPIPD",
        )

        agent.load(path=agents[i])  



        weights = equally_spaced_weights(reward_dim, n=num_weights)

        whole_front = produce_whole_front(weights, agent, env, num_eval_episodes)
        pareto_front, pareto_weights = get_pareto_front(whole_front, weights)

        
        pickle.dump(pareto_front, open(f'pareto_front_{environment}.pkl', 'wb'))
        pickle.dump(pareto_weights, open(f'pareto_weights_{environment}.pkl', 'wb'))
        pickle.dump(whole_front, open(f'whole_front_{environment}.pkl', 'wb'))

