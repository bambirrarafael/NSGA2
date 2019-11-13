import random
import numpy as np
import pandas as pd
from copy import deepcopy as deep_copy
import sympy.combinatorics.graycode as gc
import matplotlib.pyplot as plt


# define global variables
global_data_NEWAVE = './data/NEWAVE_DATA.xlsx'
global_data_portfolio = './data/PORTFOLIO_DATA.xlsx'
global_PLD = np.array(pd.read_excel(global_data_NEWAVE, 'PLD', header=None))
global_GSF = np.array(pd.read_excel(global_data_NEWAVE, 'GSF', header=None))
global_bought_energy = np.array(pd.read_excel(global_data_portfolio, 'BOUGHT'))
global_bought_price = np.array(pd.read_excel(global_data_portfolio, 'BOUGHT_PRICE'))
global_sold_energy = np.array(pd.read_excel(global_data_portfolio, 'SOLD'))
global_sold_price = np.array(pd.read_excel(global_data_portfolio, 'SOLD_PRICE'))
global_gFIS_HIDR = np.array(pd.read_excel(global_data_portfolio, 'HIDR'))

class ProblemParameters:

    def __init__(self, nvar, ncal):
        self.n_var = nvar
        self.n_cal = ncal
        # =============== imputs ===============
        self.crossover_rate = 0.7
        self.mutation_rate = 0.005
        self.bits_per_variable = 8
        self.Problem = EnergyRiskAndReturn(self.n_var)       # todo - think a better way to change this
        # =============== end imputs =============
        self.pop_size = int(np.sqrt(self.n_cal))    # set bound on population size
        if self.pop_size > 50:
            self.pop_size = 50
        if self.pop_size % 2 == 1:
            self.pop_size = self.pop_size - 1
        self.n_gen = self.n_cal // self.pop_size
        self.population = []    # initial and remaining population
        self.fathers = []
        self.sons = []
        self.everybody = []     # union of all individuals od fathers and sons
        self.frontiers = [[]]     # saves the population of each of frontier


class NSGA2(ProblemParameters):

    def __init__(self, nvar, ncal):
        self.n_var = nvar
        self.n_cal = ncal
        ProblemParameters.__init__(self, nvar=self.n_var, ncal=self.n_cal)
        self.initialize_population()
        self.plot_results()
        self.optimize()
        self.plot_results()

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

    def create_new_population(self):    # Creates the union of fathers and sons
        self.everybody = []
        for i in range(0, self.pop_size):
            self.everybody.append(deep_copy(self.population[i]))
        for i in range(0, self.pop_size):
            self.everybody.append(deep_copy(self.sons[i]))

    def cross_over(self):
        self.sons = []
        for i in range(self.pop_size):  # create empty sons
            ind = Individual(nvar=self.n_var, nbits=self.bits_per_variable, ncal=self.n_cal)
            self.sons.append(deep_copy(ind))
        for i in range(0, self.pop_size, 2):  # cross over and fill their binary code
            for j in range(self.n_var):
                r = random.randint(1, self.bits_per_variable)
                gray_code_father_1 = gc.bin_to_gray(self.fathers[i].binary_code[j])
                gray_code_father_2 = gc.bin_to_gray(self.fathers[i + 1].binary_code[j])
                gray_code_son_1 = [gray_code_father_1[0:r], gray_code_father_2[r:]]
                gray_code_son_1 = ''.join(gray_code_son_1)
                gray_code_son_2 = [gray_code_father_2[0:r], gray_code_father_1[r:]]
                gray_code_son_2 = ''.join(gray_code_son_2)
                self.sons[i].binary_code.append(gc.gray_to_bin(gray_code_son_1))
                self.sons[i + 1].binary_code.append(gc.gray_to_bin(gray_code_son_2))

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

    def sort_by_fitness(self, front, m):    # todo fix error on sorting
        sorted_front = []
        sorted_fitness = []
        for i in front:
            sorted_fitness.append(self.everybody[i].fitness[m])
        order = np.argsort(sorted_fitness, kind='quicksort')
        for i in range(len(front)):
            sorted_front.append(front[order[i]])
        return sorted_front

    def evaluate_sons(self):
        for i in range(self.pop_size):
            self.sons[i].binary_to_real(self.sons[i].binary_code)  # decript
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

    def initialize_population(self):
        for i in range(self.pop_size):
            ind = Individual(nvar=self.n_var, nbits=self.bits_per_variable, ncal=self.n_cal)
            self.population.append(deep_copy(ind))  # define it as object
            self.population[i].create_individual()  # create binary string
            self.population[i].binary_to_real(self.population[i].binary_code)  # decript
            self.population[i].calc_fitness()  # set fitness

    def mutation(self):
        for i in range(self.pop_size):
            for j in range(self.n_var):
                for k in range(self.bits_per_variable):
                    r = random.random()
                    if r <= self.mutation_rate:
                        gray_code_son = gc.bin_to_gray(self.sons[i].binary_code[j])
                        if gray_code_son[k] == '1':
                            gray_code_son = list(gray_code_son)
                            gray_code_son[k] = '0'
                            gray_code_son = ''.join(gray_code_son)
                        else:
                            gray_code_son = list(gray_code_son)
                            gray_code_son[k] = '1'
                            gray_code_son = ''.join(gray_code_son)
                        self.sons[i].binary_code[j] = gc.gray_to_bin(gray_code_son)

    def optimize(self):
        for i in range(self.n_gen):
            self.selection()
            self.cross_over()
            self.mutation()
            self.evaluate_sons()
            self.create_new_population()
            self.fast_non_dominated_sort()        # create union and evaluate rank
            self.remain_pop()
            if i % 10 == 0:
                self.plot_results()
            #self.plot_results()
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
        plt.axis([self.Problem.plot_limit_f1[0], self.Problem.plot_limit_f1[1], self.Problem.plot_limit_f2[0],
                  self.Problem.plot_limit_f2[1]])
        plt.scatter(aux_1, aux_2)
        plt.xlabel('Function 1')
        plt.ylabel('Function 2')
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


class Individual(ProblemParameters):

    def __init__(self, nvar, nbits, ncal):
        self.n_var = nvar
        self.n_bits = nbits
        self.n_cal = ncal
        ProblemParameters.__init__(self, nvar=self.n_var, ncal=self.n_cal)
        self.fitness = np.zeros(2)
        self.variables = np.zeros(self.n_var)
        self.binary_code = []
        self.rank_frontier = 0
        self.distance = 0

    def create_individual(self):
        for i in range(self.n_var):
            self.binary_code.append(gc.random_bitstring(self.n_bits))

    def binary_to_real(self, binary_str):
        for j in range(self.n_var):
            binary = int(binary_str[j])
            decimal, i, n = 0, 0, 0
            while binary != 0:
                dec = binary % 10
                decimal = decimal + dec * pow(2, i)
                binary = binary // 10
                i += 1
            self.variables[j] = self.Problem.upper_bound[j] - ((((pow(2, self.n_bits) - 1) - decimal) * (
                        self.Problem.upper_bound[j] - self.Problem.lower_bound[j])) / (pow(2, self.n_bits) - 1))

    def calc_fitness(self):
        self.fitness[0] = self.Problem.function_1(x=self.variables)
        self.fitness[1] = self.Problem.function_2(x=self.variables)
        penalties = self.Problem.constrains(x=self.variables)
        self.fitness = self.fitness + penalties

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


class Shaffer1MO:

    def __init__(self, nvar):
        self.name = 'Shaffer n° 1 M.O.'
        self.n_var = nvar
        if self.n_var > 1:      # safety display
            print('wrong number of variables for this function')
            print('=-=-=-=-=-=- n_var should be 1 -=-=-=-=-=-=')
            quit()
        self.upper_bound = np.ones(self.n_var) * 5.0
        self.lower_bound = np.ones(self.n_var) * 0.0
        self.plot_limit_f1 = [0.0, 4.5]
        self.plot_limit_f2 = [0.0, 4.5]
        self.f = [0, 0]

    def function_1(self, x):
        s = 0
        for i in range(len(x)):
            s += x[i]**2
        self.f = s
        return self.f

    def function_2(self, x):
        s = 0
        for i in range(len(x)):
            s += (x[i] - 2)**2
        self.f = s
        return self.f

    def constrains(self, x):
        penalties = 0
        return penalties


class Shaffer2MO:

    def __init__(self, nvar):
        self.name = 'Shaffer n° 2 M.O.'
        self.n_var = nvar
        if self.n_var > 1:      # safety display
            print('wrong number of variables for this function')
            print('=-=-=-=-=-=- n_var should be 1 -=-=-=-=-=-=')
            quit()
        self.upper_bound = np.array([10.0])
        self.lower_bound = np.array([-5.0])
        self.plot_limit_f1 = [-1.0, 1.0]
        self.plot_limit_f2 = [0.0, 16.0]
        self.f = [0, 0]

    def function_1(self, x):
        s = 0
        for i in range(len(x)):
            if x[i] <= 1:
                s += -x
            elif 1 < x[i] <= 3:
                s += x - 2
            elif 3 < x[i] <= 4:
                s += 4 - x
            else:
                s += x-4
        self.f = s
        return self.f

    def function_2(self, x):
        s = 0
        for i in range(len(x)):
            s += (x[i] - 5)**2
        self.f = s
        return self.f

    def constrains(self, x):
        penalties = 0
        return penalties


class FonsecaFleming:

    def __init__(self, nvar):
        self.name = 'Fonseca and Fleming'
        self.n_var = nvar
        self.upper_bound = 4 * np.ones(self.n_var)
        self.lower_bound = -4 * np.ones(self.n_var)
        self.plot_limit_f1 = [0.0, 1.0]
        self.plot_limit_f2 = [0.0, 1.0]
        self.f = [0, 0]

    def function_1(self, x):
        s = 0
        for i in range(len(x)):
            s += (x[i] - (1/np.sqrt(len(x))))**2
        self.f = 1 - np.exp(-s)
        return self.f

    def function_2(self, x):
        s = 0
        for i in range(len(x)):
            s += (x[i] + (1/np.sqrt(len(x))))**2
        self.f = 1 - np.exp(-s)
        return self.f

    def constrains(self, x):
        penalties = 0
        return penalties


class Kursawe:
    def __init__(self, nvar):
        self.name = 'Kursawe'
        self.n_var = nvar
        if self.n_var > 3:  # safety display
            print('wrong number of variables for this function')
            print('=-=-=-= n_var should be less than 3 =-=-=-=')
            quit()
        self.upper_bound = np.ones(self.n_var) * 5.0
        self.lower_bound = np.ones(self.n_var) * -5.0
        self.plot_limit_f1 = [-20.0, -14.0]
        self.plot_limit_f2 = [-12.0, 2.0]
        self.f = [0, 0]

    def function_1(self, x):
        s = 0
        for i in range(len(x)-1):
            s += -10*np.exp(-0.2*(np.sqrt(x[i]**2 + x[i+1]**2)))
        self.f = s
        return self.f

    def function_2(self, x):
        s = 0
        for i in range(len(x)):
            s += abs(x[i])**0.8 + 5*np.sin(x[i]**3)
        self.f = s
        return self.f

    def constrains(self, x):
        penalties = 0
        return penalties


class ChakongHaimes:

    def __init__(self, nvar):
        self.name = 'Chakong and Haimes'
        self.n_var = nvar
        if self.n_var != 2:  # safety display
            print('wrong number of variables for this function')
            print('=-=-=-=-=-=- n_var should be 2 -=-=-=-=-=-=')
            quit()
        self.upper_bound = np.ones(self.n_var) * 20.0
        self.lower_bound = np.ones(self.n_var) * -20.0
        self.plot_limit_f1 = [0.0, 250.0]
        self.plot_limit_f2 = [-250.0, 0.0]
        self.f = [0, 0]

    def function_1(self, x):
        s = 2 + (x[0] - 2)**2 + (x[1] - 1)**2
        self.f = s
        return self.f

    def function_2(self, x):
        s = 9*x[0] - (x[1] - 1)**2
        self.f = s
        return self.f

    def constrains(self, x):   # todo - fix constrains
        g = np.zeros(2)     # there are 2 constrains
        if x[0]**2 + x[1]**2 > 255:    # constrain number 1
            g[0] = abs((x[0]**2 + x[1]**2) - 255)    # penalty
        if x[0] - 3*x[1] + 10 > 0:
            g[1] = abs(x[0] - 3*x[1] + 10)
        penalties = np.sum(g)
        return penalties


class OsyczkaKunku:

    def __init__(self, nvar):
        self.name = 'Osyczka and Kunku'
        self.n_var = nvar
        if self.n_var != 6:  # safety display
            print('wrong number of variables for this function')
            print('=-=-=-=-=-=- n_var should be 6 -=-=-=-=-=-=')
            quit()
        self.upper_bound = np.array([10.0, 10.0, 5.0, 6.0, 5.0, 10.0])
        self.lower_bound = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self.plot_limit_f1 = [-300.0, 0.0]
        self.plot_limit_f2 = [0.0, 80.0]
        self.f = [0, 0]

    def function_1(self, x):
        s = -25*(x[0] - 2)**2 - (x[1] - 2)**2 - (x[2] - 1)**2 - (x[3] - 4)**2 - (x[5] - 1)**2
        self.f = s
        return self.f

    def function_2(self, x):
        s = 0
        for i in range(len(x)):
            s += x[i]**2
        self.f = s
        return self.f

    def constrains(self, x):   # todo - fix constrains
        g = np.zeros(6)     # there are 6 constrains
        if x[0] + x[1] - 2 < 0:                     # constrain number 1
            g[0] = abs((x[0] + x[1] - 2) - 0)       # penalty 1
        if 6 - x[0] - x[1] < 0:                     # constrain number 2
            g[1] = abs((6 - x[0] - x[1]) - 0)
        if 2 - x[1] + x[0] < 0:                     # constrain number 3
            g[2] = abs((2 - x[1] + x[0]) - 0)
        if 2 - x[0] + 3*x[1] < 0:                   # constrain number 4
            g[3] = abs((2 - x[0] + 3*x[1]) - 0)
        if 4 - (x[2] - 3)**2 - x[3] < 0:                    # constrain number 5
            g[4] = abs((4 - (x[2] - 3)**2 - x[3]) - 0)
        if (x[4] - 3)**2 + x[5] - 4 < 0:                    # constrain number 6
            g[5] = abs(((x[4] - 3)**2 + x[5] - 4) - 0)
        penalties = np.sum(g)
        return penalties


class EnergyRiskAndReturn:

    def __init__(self, nvar):
        # Parameters of simulation
        self.name = 'Energy Risk and Return'
        self.n_var = nvar
        self.upper_bound = np.ones(self.n_var) * 200
        self.lower_bound = np.ones(self.n_var) * -100
        self.plot_limit_f1 = [-250000000.0, 100000.0]
        self.plot_limit_f2 = [-1000000.0, 300000000.0]
        self.f = [0, 0]
        #
        # Parameters of the functions
        self.horizon = 12               # months
        self.alpha = 0.05               # CVaR's limit
        self.inflation = 0.0048         # VPL's rate
        self.price_delta_contract = 200     # R$/MWh
        #
        # Inputs from portfolio and NEWAVE
        self.newave_data = global_data_NEWAVE
        self.portfolio_data = global_data_portfolio
        self.PLD = global_PLD
        self.GSF = global_GSF
        self.bought_energy = global_bought_energy
        self.bought_price = global_bought_price
        self.sold_energy = global_sold_energy
        self.sold_price = global_sold_price
        self.gFIS_HIDR = global_gFIS_HIDR
        self.n_scen = len(self.PLD)
        self.n_worst_scenarios = int(self.alpha * self.n_scen)
        self.n_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]     # number of hours of each month
        self.gen_cost_HIDR = 3.3105     # US$/MWm [h]

    def function_1(self, x):    # Return    - OK
        #
        # Pre-allocate variables
        aux_bought = np.zeros([1, self.horizon])
        aux_sold = np.zeros([1, self.horizon])
        aux_gFIS_HIDR = np.zeros([1, self.horizon])
        aux_income = np.zeros(np.shape(self.PLD))
        net_present_value = np.zeros(len(self.PLD))
        #
        # Calculate auxiliary variables of portfolio - constants
        for i in range(len(self.bought_price)):
            aux_bought = aux_bought + self.bought_energy[i, :]*self.bought_price[i]
        for i in range(len(self.sold_price)):
            aux_sold = aux_sold + self.sold_energy[i, :]*self.sold_price[i]
        for i in range(len(self.gFIS_HIDR)):
            aux_gFIS_HIDR = aux_gFIS_HIDR + self.gFIS_HIDR[i, :]*self.gen_cost_HIDR
        #
        # Calculate the monthly income for each scenario
        for i in range(self.n_scen):
            # but first, calculate portfolio's exposition
            resources = np.sum(self.bought_energy, axis=0) + self.GSF[i, :]*np.sum(self.gFIS_HIDR, axis=0)
            exposition = resources - np.sum(self.sold_energy, axis=0) - x
            #
            # and then the income
            aux_income[i, :] = self.n_hours*(self.price_delta_contract*x + aux_sold + exposition*self.PLD[i, :]
                                             - aux_bought - aux_gFIS_HIDR)
            net_present_value[i] = np.npv(self.inflation, aux_income[i, :])
        expected_income = np.mean(net_present_value)
        self.f = - expected_income  # maximize it
        return self.f

    def function_2(self, x):    # risk or CVaR  - Ok
        #
        # Pre-allocate variables
        aux_bought = np.zeros([1, self.horizon])
        aux_sold = np.zeros([1, self.horizon])
        aux_gFIS_HIDR = np.zeros([1, self.horizon])
        aux_income = np.zeros(np.shape(self.PLD))
        #
        # Calculate auxiliary variables of portfolio - constants
        for i in range(len(self.bought_price)):
            aux_bought = aux_bought + self.bought_energy[i, :]*self.bought_price[i]
        for i in range(len(self.sold_price)):
            aux_sold = aux_sold + self.sold_energy[i, :]*self.sold_price[i]
        for i in range(len(self.gFIS_HIDR)):
            aux_gFIS_HIDR = aux_gFIS_HIDR + self.gFIS_HIDR[i, :]*self.gen_cost_HIDR
        #
        # Calculate the monthly income for each scenario
        for i in range(self.n_scen):
            # but first, calculate portfolio's exposition
            resources = np.sum(self.bought_energy, axis=0) + self.GSF[i, :]*np.sum(self.gFIS_HIDR, axis=0)
            exposition = resources - np.sum(self.sold_energy, axis=0) - x
            #
            # and then the income
            aux_income[i, :] = self.n_hours*(self.price_delta_contract * x + aux_sold + exposition *self.PLD[i, :]
                                             - aux_bought - aux_gFIS_HIDR)
        horizon_income = np.sum(aux_income, axis=1)
        expected_income = np.mean(horizon_income)
        #
        # Calculate the worst incomes
        sorted_horizon_income = np.sort(horizon_income)
        worst_incomes = sorted_horizon_income[:self.n_worst_scenarios]
        expected_tail_loss = np.mean(worst_incomes)
        conditional_value_at_risk = expected_income - expected_tail_loss
        self.f = conditional_value_at_risk
        return self.f

    def constrains(self, x):
        penalties = 0.0
        return penalties


'''# test Energy risk and return functions
def test_functions():
    x = np.linspace(-100, 200, 500)
    income = []
    cvar = []
    for k in range(len(x)):
        paramaters = ProblemParameters(1, 1000)
        income.append(paramaters.Problem.function_1(x[k]))
        cvar.append(paramaters.Problem.function_2(x[k]))
        print(str(k))
    income = np.array(income)

    plt.plot(x, -income, label='- Income', c='b')
    plt.plot(x, cvar, label='CVaR', c='r')
    plt.xlabel('Change of contracting')
    plt.ylabel('Amount of Cash [R$]')
    plt.title('Intrinsic Risk for 1 variable')
    plt.legend()
    plt.show()
    return income, cvar


income, cvr = test_functions()'''

pareto = NSGA2(nvar=12, ncal=10000)
print('=== End ===')


# ====================================================

