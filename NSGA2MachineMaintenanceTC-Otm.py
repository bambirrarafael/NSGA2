import random
import numpy as np
import pandas as pd
from copy import deepcopy as deep_copy
import matplotlib.pyplot as plt
import csv


class ProblemParameters:

    def __init__(self, ncal):
        self.n_var = 500
        self.n_cal = ncal
        # =============== imputs ===============
        self.crossover_rate = 0.8
        self.mutation_rate = 0.05
        self.cuts = 10
        self.Problem = MachineMaintenance()      
        # =============== end imputs =============
        self.pop_size = int(np.sqrt(self.n_cal))    # set bound on population size
        if self.pop_size > 200:
            self.pop_size = 200
        if self.pop_size % 2 == 1:
            self.pop_size = self.pop_size - 1
        self.n_gen = self.n_cal // self.pop_size
        self.population = []    # initial and remaining population
        self.fathers = []
        self.sons = []
        self.everybody = []     # union of all individuals od fathers and sons
        self.frontiers = [[]]     # saves the population of each of frontier


class NSGA2(ProblemParameters):

    def __init__(self, ncal):
        self.n_var = 500
        self.n_cal = ncal
        ProblemParameters.__init__(self, ncal=self.n_cal)
        self.initialize_population()
        self.plot_results()
        self.optimize()
        self.plot_results()

    def create_new_population(self):    # Creates the union of fathers and sons
        self.everybody = []
        for i in range(0, self.pop_size):
            self.everybody.append(deep_copy(self.population[i]))
        for i in range(0, self.pop_size):
            self.everybody.append(deep_copy(self.sons[i]))

    def cross_over(self):
        # A crossover method with another individual using n cut points
        n = self.cuts
        #
        # erase sons of previous gen
        self.sons = []
        for i in range(self.pop_size):  # create empty sons
            '''ind = Individual(ncal=self.n_cal)
            self.sons.append(deep_copy(ind))'''
            self.sons.append(Individual(ncal=self.n_cal))
        #
        # loop on population
        for i in range(0, self.pop_size, 2):
            if random.random() <= self.crossover_rate:
                points = random.sample(range(1, self.n_var - 1), n)
                points.sort()
                points.insert(0, 0)
                points.append(self.n_var)
                selfUse = True
                for j in range(1, len(points)):
                    fromPos = points[j - 1]
                    toPos = points[j]
                    for k in range(fromPos, toPos):
                        if selfUse:
                            self.sons[i].variables[k] = self.fathers[i].variables[k]
                            self.sons[i + 1].variables[k] = self.fathers[i + 1].variables[k]
                        else:
                            self.sons[i + 1].variables[k] = self.fathers[i].variables[k]
                            self.sons[i].variables[k] = self.fathers[i + 1].variables[k]
                    selfUse = not selfUse
            else:
                self.sons[i].variables = self.fathers[i].variables
                self.sons[i + 1].variables = self.fathers[i + 1].variables

    def crowd_distance(self, population, front):
        everybody = self.everybody
        aux_sort = []
        l = len(front)
        for i in front:
            everybody[i].distance = 0.0
        for m in range(2):
            front = self.sort_by_fitness(front, m)
            f_max = everybody[front[l - 1]].fitness[m]
            f_min = everybody[front[0]].fitness[m]
            everybody[front[0]].distance = 1e99
            everybody[front[l - 1]].distance = 1e99
            width_of_interval = (f_max - f_min)
            for i in range(1, l - 2):
                fitness_of_individual_a = everybody[front[i - 1]].fitness[m]
                fitness_of_individual_b = everybody[front[i + 1]].fitness[m]
                size_of_cuboid_of_i = (fitness_of_individual_b - fitness_of_individual_a)/width_of_interval
                distance = everybody[front[i]].distance + size_of_cuboid_of_i
                everybody[front[i]].distance = distance.copy()
        for i in range(l):
            aux_sort.append(everybody[front[i]].distance)
        order = np.argsort(-np.array(aux_sort), kind='quicksort')
        for i in range(l-1):
            if len(population) < self.pop_size:
                population.append(everybody[front[order[i]]])
            if len(population) == self.pop_size:
                break
        return population

    def evaluate_sons(self):
        for i in range(self.pop_size):
            self.sons[i].calc_fitness()  # set fitness

    def fast_non_dominated_sort(self):
        frontiers = [[]]
        everybody = self.everybody
        S = []                          # the set of solutions that solution p dominates
        n = []                          # number of solutions that dominates solution p
        for p in range(0, len(everybody)):
            S.append([])
            n.append(0)
            for q in range(0, len(everybody)):
                if everybody[p].dominates(everybody[q]):
                    S[p].append(q)
                elif everybody[q].dominates(everybody[p]):
                    n[p] = n[p] + 1
            if n[p] == 0:
                everybody[p].rank_frontier = 0
                frontiers[0].append(p)

        i = 0
        while frontiers[i]:
            next_front = []
            for p in frontiers[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if n[q] == 0:
                        everybody[q].rank_frontier = i + 1
                        next_front.append(q)
            i = i + 1
            frontiers.append(next_front)
        del frontiers[len(frontiers)-1]
        self.everybody = everybody
        self.frontiers = frontiers

    def initialize_population(self):    # ok!
        '''# ________________________________________________________________________________________________________
        # gambs to add old run as the new population
        old_results = pd.read_csv('newpop.csv', header=None)
        # end gambs =============================================================================================='''
        for i in range(self.pop_size):
            ind = Individual(ncal=self.n_cal)
            self.population.append(deep_copy(ind))  # define it as object
            '''# ____________________________________________________________________________________________________
            # gambs to add old run as the new population
            self.population[i].variables = np.array(old_results.iloc[i])
            self.population[i].calc_fitness()
            # end gambs =========================================================================================='''
            self.population[i].create_individual()  # create string of variables
            self.population[i].calc_fitness()  # set fitness
        #
        # inset the strong individuals
        self.population[-3].variables = np.zeros([500]) + 1
        self.population[-3].calc_fitness()
        self.population[-2].variables = np.zeros([500]) + 2
        self.population[-2].calc_fitness()
        self.population[-1].variables = np.zeros([500])
        self.population[-1].calc_fitness()

    def mutation(self):     # ok!
        number_of_swaps = 50
        which_individuals = random.sample(range(0, self.pop_size), int(self.pop_size * self.mutation_rate))
        # which_individuals.sort()
        for i in which_individuals:
            #
            # define which machines are going to swap the maintenance planning
            which_machines = random.sample(range(0, 500), number_of_swaps)
            # which_machines.sort()
            #
            # swap the positions
            for j in which_machines:
                old_maintenance_plan = self.sons[i].variables[j]
                while self.sons[i].variables[j] == old_maintenance_plan:
                    self.sons[i].variables[j] = random.randint(0, 2)

    def optimize(self):
        for i in range(self.n_gen):
            self.selection()
            self.cross_over()
            self.mutation()
            self.evaluate_sons()
            self.create_new_population()
            self.fast_non_dominated_sort()        # create union and evaluate rank
            self.remain_pop()
            '''if i % 10 == 0:
                self.plot_results()'''
            # self.plot_results()
            print('Generation : ' + str(i))

    def plot_results(self):
        aux_1 = []
        aux_2 = []
        aux_front = []
        for i in range(0, len(self.population)):
            aux_1.append(self.population[i].fitness[0])
        for i in range(0, len(self.population)):
            aux_2.append(self.population[i].fitness[1])
            aux_front.append(self.population[i].rank_frontier)
        '''plt.axis([self.Problem.plot_limit_f1[0], self.Problem.plot_limit_f1[1], self.Problem.plot_limit_f2[0],
                  self.Problem.plot_limit_f2[1]])'''
        plt.scatter(aux_1, aux_2)
        plt.xlabel('Maintenance cost')
        plt.ylabel('Failure_cost')
        plt.title(str(self.Problem.name)+' for '+str(self.n_var)+' variable(s)')
        plt.show()

    def remain_pop(self):
        # sort everybody for the rank and crowd distance of individuals
        population = []
        for front in self.frontiers:
            if len(front) + len(population) > self.pop_size:
                population = self.crowd_distance(population=population, front=front)
                break
            for i in front:
                population.append(deep_copy(self.everybody[i]))
            if len(population) == self.pop_size:
                break
        self.population = deep_copy(population)

    def selection(self):
        self.fathers = []
        for i in range(self.pop_size):
            u = self.population[random.randint(0, self.pop_size - 1)]
            v = self.population[random.randint(0, self.pop_size - 1)]
            if u.dominates(other=v):
                self.fathers.append(deep_copy(u))
            elif v.dominates(other=u):
                self.fathers.append(deep_copy(v))
            else:
                r = random.random()
                if r >= 0.5:
                    self.fathers.append(deep_copy(u))
                else:
                    self.fathers.append(deep_copy(v))

    def sort_by_fitness(self, front, m):
        sorted_front = []
        sorted_fitness = []
        for i in front:
            sorted_fitness.append(self.everybody[i].fitness[m])
        order = np.argsort(sorted_fitness, kind='quicksort')
        for i in range(len(front)):
            sorted_front.append(front[order[i]])
        return sorted_front


class Individual(ProblemParameters):

    def __init__(self, ncal):
        self.n_var = 500
        self.n_cal = ncal
        ProblemParameters.__init__(self, ncal=self.n_cal)
        self.fitness = np.zeros(2)
        self.variables = np.zeros(self.n_var)
        self.rank_frontier = 0
        self.distance = 0

    def create_individual(self):
        for i in range(self.n_var):
            self.variables[i] = random.randint(0, 2)

    def calc_fitness(self):
        self.fitness[0] = self.Problem.evaluate_maintenance_cost(x=self.variables)
        self.fitness[1] = self.Problem.evaluate_failure_cost(x=self.variables)

    def dominates(self, other):  # check - ok
        u = self.fitness
        v = other.fitness
        flag = False
        if (u[0] < v[0] and u[1] < v[1]) or (u[0] <= v[0] and u[1] < v[1]) or (u[0] < v[0] and u[1] <= v[1]):
            flag = True
        elif u[0] == v[0] and u[1] == v[1]:
            flag = False
        if flag:
            return True
        return False


class MachineMaintenance:

    def __init__(self):
        self.name = 'Machine Maintenance Problem'
        self.n_var = 500
        self.plot_limit_f1 = [0, 2000]
        self.plot_limit_f2 = [1350, 1500]
        #
        # Import parameters data from all csv files as global variables
        self.groups = pd.read_csv('./data/ClusterDB.csv', names=['Cluster', 'eta', 'beta'])
        self.equipments = pd.read_csv('./data/EquipDB.csv', names=['ID', 't0', 'Cluster', 'Failure cost'])
        self.maintenance_planning = pd.read_csv('./data/MPDB.csv', names=['Maintenance type', 'k', 'Maintenance cost'])
        #
        # Adjust Equipment Data Base
        self.equipments['eta'] = 0
        self.equipments['beta'] = 0
        for eq, r in self.equipments.iterrows():   # todo - improve this gambs - remove warning
            self.equipments['eta'].loc[eq] = self.groups['eta'].loc[r['Cluster'] - 1]
            self.equipments['beta'].loc[eq] = self.groups['beta'].loc[r['Cluster'] - 1]
        #
        # Define failure probabilities for all equipments
        self.p = np.zeros([500, 3])  # Pre-allocate variable
        self.define_failure_probability_for_all_equipments()  # ok!

    def define_failure_probability_for_all_equipments(self):
        # Calculate failure probability
        for i in range(500):
            for j in range(3):
                machine = self.equipments.loc[i]
                plan = self.maintenance_planning.loc[j]
                f_i_t0_k_delta_t = self.evaluate_failure_function(t=machine['t0'] + plan['k'] * 5, eta=machine['eta'],
                                                                  beta=machine['beta'])
                f_i_t0 = self.evaluate_failure_function(t=machine['t0'], eta=machine['eta'], beta=machine['beta'])
                numerator = f_i_t0_k_delta_t - f_i_t0
                denominator = 1 - f_i_t0
                self.p[i][j] = numerator / denominator
        return self.p

    def evaluate_failure_function(self, t, eta, beta):
        f = 1 - np.exp(-(t / eta) ** beta)
        return f

    def evaluate_maintenance_cost(self, x):  # function 1 - min
        machine_maintenance_cost = []
        for i in range(500):
            plan = self.maintenance_planning.loc[x[i]]
            machine_maintenance_cost.append(plan['Maintenance cost'])
        total_maintenance_cost = np.sum(machine_maintenance_cost)
        return total_maintenance_cost

    def evaluate_failure_cost(self, x):  # function 2 - min - ok!
        machine_failure_cost = []
        for i in range(500):
            machine = self.equipments.loc[i]
            plan = self.maintenance_planning.loc[x[i]]
            machine_failure_cost.append(machine['Failure cost'] * self.p[i][int(plan['Maintenance type'] - 1)])
        total_expected_failure_cost = np.sum(machine_failure_cost)
        return total_expected_failure_cost

# todo - Set up Hyper Volume index
    
pareto = NSGA2(ncal=250000)
print('=== End ===')

with open('Pareto_Frontier run 3', "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for line in range(200):
        writer.writerow(pareto.population[line].variables)

