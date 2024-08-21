import numpy as np
from pymoo.indicators.hv import HV
import random
import math
import copy
from sklearn.metrics import silhouette_score
import wandb
import kmedoids
import pickle
from sklearn_extra.cluster import KMedoids

import kmedoids


class PanClustering:
    def __init__(self, n, g, distances_matrix, local_opt = False, validation_criteria = 'silhoutte', seed = 234, pr = 0.7, pu = 0.2, ps =0.2, pm = 0.6, log = False):
        '''
        n - population size (of the partition)
        g - number of generations to run
        validation criteria - to evaluate goodness of clusters in both spaces
        solution_set - is a class which has solutions i both spaces (objectives and sets of states), and can calculate the distances between them
        '''
        self.n = n
        self. g = g
        self.distances_matrix = distances_matrix
        self.validation_criteria = validation_criteria
        self.n_policies = distances_matrix['objectives'].shape[0]
        self.pr = pr
        self.pu = pu 
        self.pm = pm
        self.ps = ps
        self.log = log
        self.local_opt = local_opt
        random.seed(seed)
        np.random.seed(seed)


    def create_solution(self, invalid = False):
        '''creates a solution (partition): 2 dim np.array, where each position in the array is an np.array with idx of solutions (policies) from the solution set
        if invalid = True, an invalid partitioning is created, that is, a partitioning with either only one cluster or clusters with one solution'''
        #create a list for partitions
        partitions = []
        #create a list of randomly shuffled policies
        #get the number of policies to be partitioned


        # n = self.solution_set['objectives'].shape[0]
        policies = [i for i in range(self.n_policies)]
        random.shuffle(policies)
        if invalid==False:
            #choose a random number of clusters (currently only for valid)
            n_clusters = random.randint(2, int(self.n_policies/2))

            for i in range(n_clusters):
                #until we consider the last cluster
                if i+1 != n_clusters:
                    #choose a partition index, to partition the policy list and create first cluster
                    part_idx = random.randint(2, len(policies) - (n_clusters - (i+1))*2)
                    #take the policies
                    partition = policies[0: part_idx]
                    #append the first partition
                    partitions.append(partition)
                    #remove the policies already put in the partition list
                    policies = policies[part_idx:]
                else: #if this is the last partition, put all the remaining policies
                    partition = policies
                    partitions.append(partition)
        
        else:
            #with 0.5 prob, put all the solutions into one cluster
            if random.random() > 0.5:
                partitions.append(policies)
            else:
                partitions = [[policy] for policy in policies]


        return partitions


    def is_invalid(self, solution):
        '''Check if the solution is invalid, that is it contains only one cluster, or at laest one of the clusters has only one solution'''
        isinvalid = False
        #only one cluster:
        if len(solution) == 1:
            isinvalid = True
        #cluster has one solution
        cluster_length = [len(cl) for cl in solution]
        if 1 in cluster_length or 0 in cluster_length:
            isinvalid = True
        return isinvalid

    def sample_population(self):
        '''samples initial population of partitionings'''
        population = []
        for i in range(self.n):
            solution = self.create_solution()
            population.append(solution)
        return population

    def mating_selection(self, P):
        '''Selects two parent solutions (different partitions) from the current population for mating'''
        
        parents = random.sample(P, 2)
        p1 = parents[0]
        p2 = parents[1]

        return p1, p2

    def recombine(self, p1, p2):
        '''recombinates parent solutions with each other'''

        len_p1 = len(p1)
        len_p2 = len(p2)

        #find two split points, based on the lenght of the shorter parent
        split_1 = random.randint(1, min(len_p1, len_p2)-1)
        split_2 = random.randint(split_1+1, min(len_p1, len_p2))

        #recombine p1
        recombined_p1 = p1[:split_1]
        for cl in p2[split_1:split_2+1]:
            recombined_p1.append(cl)

        for cl in p1[split_2:]:
            recombined_p1.append(cl)

        #check if there any solutions are clustered double
        for i in range(len(recombined_p1)-1):
            for j in range(i+1, len(recombined_p1)):
                recombined_p1[j] = [x for x in recombined_p1[j] if x not in recombined_p1[i]]


        #recombine p2
        recombined_p2 = p2[:split_1]
        for cl in p1[split_1:split_2+1]:
            recombined_p2.append(cl) 
        for cl in p2[split_2:]:
            recombined_p2.append(cl)

        #check if there any solutions are clustered doublr
        for i in range(len(recombined_p2)-1):
            for j in range(i+1, len(recombined_p2)):
                recombined_p2[j] = [x for x in recombined_p2[j] if x not in recombined_p2[i]]
        
        #check if all the policies are clustered - first solution
        
        
        p1_flat = [x for xs in recombined_p1 for x in xs]

        if len(p1_flat) < self.n_policies:
            missing_cluster = []
            for i in range(0, self.n_policies):
                if i not in p1_flat:
                    missing_cluster.append(i)
            recombined_p1.append(missing_cluster)

        #check if all the policies are clustered - first solution

        p2_flat = [x for xs in recombined_p2 for x in xs]

        if len(p2_flat) < self.n_policies:
            missing_cluster = []
            for i in range(0,self.n_policies):
                if i not in p2_flat:
                    missing_cluster.append(i)
            recombined_p2.append(missing_cluster)
            
        #remove empty clusters
        recombined_p1 = [cl for cl in recombined_p1 if cl]
        recombined_p2 = [cl for cl in recombined_p2 if cl]

        return recombined_p1, recombined_p2


    def move(self, solution):
        '''Moves randomly chosen solution from one cluster to another'''

        #random cluster to take the solution from
        if len(solution) != 1:
            random_cluster = random.choice(solution)

            while len(random_cluster) == 0:
                random_cluster = random.choice(solution)

            #random solution
            random_solution = random.choice(random_cluster)

            #remove the random solution from the random cluster chose
            idx_random_cluster = solution.index(random_cluster)
            solution[idx_random_cluster].remove(random_solution)

            #choose a random cluster to put the solution in
            new_clusters = [i for i, _ in enumerate(solution)]
            #make sure it's not the same cluster the solution had belonged to
            new_clusters.remove(idx_random_cluster)

            random_cluster = random.choice(new_clusters)
            #put the solution in the cluster
            solution[random_cluster].append(random_solution)

        return solution
    
    def merge(self, solution):
        '''Merges two randomly chosen clusters together'''

        if len(solution) >= 2:
            random_clusters = random.sample(solution, 2)

            joined_cluster = random_clusters[0] + random_clusters[1]
            for random_cl in random_clusters:
                solution.remove(random_cl)
            solution.append(joined_cluster)
        
        return solution
    
    def split(self, solution):
        '''Splits one randomly chosen cluster'''

        random_cluster = random.choice(solution)
        solution.remove(random_cluster)
        random.shuffle(random_cluster)
        if len(random_cluster) > 1:
        # Choose a random split point, ensuring it's not at the very start or end
            split_point = random.randint(1, len(random_cluster) - 1)
            first_cluster = random_cluster[:split_point]
            second_cluster = random_cluster[split_point:]
        else:
        # For lists with 0 or 1 elements
            first_cluster = random_cluster
            second_cluster = []
        solution.append(first_cluster)
        solution.append(second_cluster)    
        return solution

    


    def mutate(self, solution):
        '''mutates given solution'''

        if np.random.rand() <= self.pm:
            solution = self.move(solution)
            #remove empty clusters
            solution = [cl for cl in solution if cl]

        if np.random.rand() <= self.pu:
            solution = self.merge(solution)
            #remove empty clusters
            solution = [cl for cl in solution if cl]

        if np.random.rand() <= self.ps:
            solution = self.split(solution)
            #remove empty clusters
            solution = [cl for cl in solution if cl]


        return solution


    def variate(self, P):
        '''variates given population'''
        parent_population = copy.deepcopy(P)
        offspring = []
        for i in range(math.ceil(self.n/2)):
            o1 = self.create_solution(invalid=True)
            o2 = self.create_solution(invalid=True)

            while self.is_invalid(o1) or self.is_invalid(o2):
                #randomly selects two parents from P
                p1, p2 = copy.deepcopy(self.mating_selection(parent_population))
                #set offspring to parents
                o1_p = p1
                o2_p = p2
                #with probability pr, recombine the parents
                if np.random.rand() <= self.pr:
                    o1_p, o2_p = self.recombine(p1, p2) 
                o1_p = self.mutate(o1_p)
                o2_p = self.mutate(o2_p)
                if self.is_invalid(o1) and not self.is_invalid(o1_p):
                    o1 = o1_p
                if self.is_invalid(o2) and not self.is_invalid(o2_p):
                    o2 = o2_p

            offspring.append(o1)
            offspring.append(o2)
        
        joint_population = parent_population + offspring

        return joint_population

    def select(self, joint_population):
        '''given the joint population of size 2*n, 
        iteratively removes solutions, which add the least to the performence measured in terms of hypervolume'''
        #calculate total hypervolume (with all the solutions included)
        total_hyp = self.calculate_hypervolume(joint_population)
        #caclulate hypervolumes without each of the partitions
        hyp_minus_pi = [self.calculate_hypervolume(joint_population[:i] + joint_population[i+1:]) for i, _ in enumerate(joint_population)]
        #calculate the difference
        hyp_diff = [total_hyp - item for item in hyp_minus_pi]
        partitions_sorted = sorted(range(len(hyp_diff)), key=lambda index: hyp_diff[index], reverse=True)
        current_population_idx = partitions_sorted[:self.n]
        current_population = [joint_population[idx] for idx in current_population_idx]
        return current_population

    def calculate_hypervolume(self, population):
        '''calculates hypervolume for 2 objectives (minimization problem): 
        1) minimize sillhuette index in the obj space
         2) minimize silluette index in the policy space '''
        #first evaluate population - that is calculate the sillhuette indexes for each partition
        evaluated_population = self.eval(population)
        hypervolume = HV(ref_point=[2, 2]) #TODO check with Jazmin
        hypervolume = hypervolume(evaluated_population)
        return hypervolume

    
    def eval(self, population):
        '''calculates the objectives (silhouette indexes adapted for minimization: -(S(C) - 1))'''
        evaluated_population = []
        for partition in population:
            sil_index = self.compute_sil_index(partition)
            evaluated_population.append(sil_index)
        return np.array(evaluated_population)

    def compute_sil_index(self, partition):
        '''the silhouette score of the clustering is an average of all
        silhouettes scored for each data point i.
        s(i) = b(i) - a(i)/max{a(i), b(i)}, where
        b(i) is the  average distance from ith data point to the data points in the closest cluster (we identify the closest cluster based on the 
        distances between clusters' medoids)
        a(i) is the average distance from ith data point to the other data points in the same cluster'''
        #get the distances
        #calculate sil index for all data points and evg them
        # tranform the partition into a 1D list, where each idx is the policy and the value of the element in place of the idx is the cluster
        # eg from a list: [[2,5],[0,1,3,4]] to labels: [0,0,1,0,0,1]
        
        c_labels = [0]*self.n_policies
        
        for i, cluster in enumerate(partition):
            for idx in cluster:
                c_labels[idx] = i

        sil_obj = silhouette_score(self.distances_matrix['objectives'], c_labels, metric='precomputed')

        sil_beh = silhouette_score(self.distances_matrix['policies'], c_labels, metric='precomputed')
        sil_score = [-(sil_obj-1), -(sil_beh-1)]

        return sil_score
    
    def iterative_kmed(self):
        kmed_pop = []
        seed = 0
        while len(kmed_pop) < self.n:
            seed += 1
            for i in range(2, round(self.n_policies/2)):
                clustering_obj = kmedoids.fasterpam(self.distances_matrix['objectives'], i, random_state=seed)
                clustering_beh = kmedoids.fasterpam(self.distances_matrix['policies'], i, random_state=seed)
                clusters = np.unique(clustering_obj.labels)
                obj_clusters = []
                for cl in clusters:
                    indexes = np.where(clustering_obj.labels == cl)[0].tolist()
                    obj_clusters.append(indexes)
                if not self.is_invalid(obj_clusters):
                    kmed_pop.append(obj_clusters)

                clusters = np.unique(clustering_beh.labels)
                beh_clusters = []
                for cl in clusters:
                    indexes = np.where(clustering_beh.labels == cl)[0].tolist()
                    beh_clusters.append(indexes)
                if not self.is_invalid(beh_clusters):
                    kmed_pop.append(beh_clusters)

        return kmed_pop


    def calculate_medoids(self, distance_matrix, cluster):

        c_labels = [0]*self.n_policies
        
        for i, cluster in enumerate(cluster):
            for idx in cluster:
                c_labels[idx] = i
        unique_labels = np.unique(c_labels)
        medoids = []


        for label in unique_labels:
            # Find the indexes of points in the current cluster
            indexes = np.where(c_labels == label)[0]
            
            # Subset the distance matrix to only those points
            cluster_distance_matrix = distance_matrix[np.ix_(indexes, indexes)]
            
            # Sum distances within the cluster for each point
            distance_sums = np.sum(cluster_distance_matrix, axis=1)
            
            # The medoid is the point with the minimum sum of distances to others
            medoid_index = indexes[np.argmin(distance_sums)]
            medoids.append(medoid_index)

        return np.array(medoids)


    # def localopt(self, P_set, distances):

    #     Plocal = []
    #     for partition in P_set:
    #         medoids = self.calculate_medoids(distances, partition)

    #         local_cl = kmedoids.fasterpam(distances, medoids=medoids).labels
    #         clusters = np.unique(local_cl)
    #         optimized_clustering = []
    #         for cl in clusters:
    #             indexes = np.where(local_cl == cl)[0].tolist()
    #             optimized_clustering.append(indexes)
    #         if not self.is_invalid(optimized_clustering):
    #             Plocal.append(optimized_clustering)

    #     return Plocal

    def localopt(self, P_set, distances):
        partitions = []
        for partition in P_set:
            local_cl = kmedoids.fasterpam(distances, len(partition)).labels
            clusters = np.unique(local_cl)
            optimized_clustering = []

            for cl in clusters:
                indexes = np.where(local_cl == cl)[0].tolist()
                optimized_clustering.append(indexes)
            if not self.is_invalid(optimized_clustering):
                partitions.append(optimized_clustering)
        return partitions




    def check(self, member):
        '''Check every member of the population is corrent'''
        if self.is_invalid(member):
            print(f'Partition: {member} is invalid')
        elif sum(len(sublist) for sublist in member) > self.n_policies:
            print(f'The partition {member} has too many solutions')
        elif sum(len(sublist) for sublist in member) < self.n_policies:
            print(f'The partition {member} has no enough solutions')
        elif len(set([item for sublist in member for item in sublist])) != len([item for sublist in member for item in sublist]):
            print(f'Partiotin : {member} doesnt have all solutions distinct')
        else:
            print(f'All fine')


    def run(self):
        hypervolumes = []
        P = self.sample_population()
        print(f'Starting hypervolume: {self.calculate_hypervolume(P)}')
        for g in range(self.g):
            P_set = self.variate(P)
            if self.local_opt:
                Po = self.localopt(P_set, self.distances_matrix['objectives'])
                Pb = self.localopt(P_set, self.distances_matrix['policies'])
                P_set = P_set + Pb + Po

            # for member in P_set:
            #     self.check(member)
            P = self.select(P_set)

            hyp = self.calculate_hypervolume(P)
            print(f'Hypervolume in {g} iteration: {hyp}')
            hypervolumes.append(hyp)
            if self.log == True:
                wandb.log({'hypervolume': hyp, 'generation': g})
        return P, hyp, hypervolumes
    





