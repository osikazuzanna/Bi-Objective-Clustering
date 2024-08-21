from src.pareto_analysis import PanClustering
import wandb
import pickle


def main():

    env_names = ['reacher' , 'highway', 
        'lunar'
    'minecart'
                ]

    n = 10
    params = {'reacher' : {'g':100, 'pr':0.7, 'pu': 0.4, 'ps': 0.4, 'pm': 0.6},
    'highway': {'g':200, 'pr':0.7, 'pu': 0.2, 'ps': 0.2, 'pm': 0.6},
    'lunar': {'g':500, 'pr':0.9, 'pu': 0.6, 'ps': 0.6, 'pm': 0.8},
    'minecart': {'g':200, 'pr':0.9, 'pu': 0.6, 'ps': 0.6, 'pm': 0.8}
    }


    for env_name in env_names:
        distances = pickle.load(open(f'other_envs/distances_{env_name}.pkl', 'rb'))
        for seed in range(seed):
            wandb.init(project='bi_clustering', entity='osikaz', config={"seed": seed}, name=f'env:{env_name}_pop:{n}_g:{g}_seed:{seed}')
            g = params[env_name]['g']
            pr = params[env_name]['pr']
            pu = params[env_name]['pu']
            ps = params[env_name]['ps']
            pm = params[env_name]['pm']
            cluster = PanClustering(n, g, distances, pr=0.9,pu=0.6, ps=0.6,pm=0.8,  local_opt=True)
            P, hype, hypervolumes = cluster.run()
            all_results = {'P':P, 'hype': hype, 'hypervolumes': hypervolumes}
            pickle.dump(all_results, open(f'data/output/PAN_results_{env_name}.pkl', 'wb'))
            wandb.finish()



if __name__ == "__main__":
    main()