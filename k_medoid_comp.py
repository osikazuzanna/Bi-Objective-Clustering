from src.pareto_analysis import PanClustering
import wandb
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV
import copy

def visualise_and_compare(env_names):
    '''Visualises the solutions with clusters and calculates the k-medoid clusters'''

    for env_name in env_names:
        #open the distance metric to create an instance of pan clustering
        distances = pickle.load(open(f'data/input/distances_{env_name}.pkl', 'rb'))
        #initialise pan clustering for the env
        cluster = PanClustering(n, g,distances, local_opt=True)
        #open the pan results of pan clustering

        PAN_results = pickle.load(open(f'data/output/PAN_results_{env_name}.pkl', 'rb'))
        #get the population of clustering
        P = PAN_results['P']
        #get the final hypervolume coming from PAN
        hyp = PAN_results['hype']

        evaluated = cluster.eval(P)
        evaluated = 1-evaluated
        
        #do the kmedoids for the same distances (for both, objective and behaviour)


        population = copy.deepcopy(cluster.iterative_kmed())

        #filter only suitable k-medoid clusterings to compare with PAN
        eval_pop = copy.deepcopy(cluster.select(population))

        #calculate hypervolume
        hyp_kmed = cluster.calculate_hypervolume(eval_pop)
        evaluated_kmed = cluster.eval(eval_pop)
        evaluated_kmed = 1-evaluated_kmed

        plt.scatter(evaluated[:, 0], evaluated[:,1], color = 'red', marker = 'o', label = f'PAN Clustering (hyp: {round(hyp, 3)})')


        plt.scatter(evaluated_kmed[:, 0], evaluated_kmed[:,1], color = 'blue', marker = 's', label = f'K-medoid Clustering (hyp: {round(hyp_kmed, 3)})')

        print(f'Env: {env_name}..........PAN hypervolume: {hyp}, kmed hypervolume: {hyp_kmed} ')

        plt.xlabel('sil index: objectives')
        plt.ylabel('sil index: behaviour')
        plt.legend(loc='lower left')
        plt.show()


def main():

    env_names = [ 'reacher' , 'highway', 'lunar', 'minecart'
                ]
    n = 10
    g = 500


    visualise_and_compare(env_names)

if __name__ == "__main__":
    main()
