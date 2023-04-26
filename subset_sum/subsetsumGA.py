# imports
from leap_ec import Individual
from leap_ec import ops, util
from leap_ec.decoder import IdentityDecoder
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.binary_rep.problems import ScalarProblem
import numpy as np
import random
from toolz import pipe


# implementation based on 2012 paper by Bhasin and Singla
class SubsetSumGAFitness(ScalarProblem):
    def __init__(self, ss_list, target_sum):
        super().__init__(maximize=False)
        self.ss_list = np.array(ss_list)
        self.target_sum = target_sum

    def eval_sum(self, ind):
        return np.dot(ind, self.ss_list)

    # TODO: define fitness function and see if this works
    def evaluate(self, ind):
        ss_sum = self.eval_sum(ind)
        s = 0
        if self.target_sum - ss_sum >= 0:
            s = 1
        return s * (self.target_sum - ss_sum) + (1 - s) * ss_sum


# simple GA implementation for subset sum problem
class SubsetSumGASimple:
    def __init__(
        self,
        n: int,
        p_m: float,
        p_c: float,
        trn_size: int,
        weights: list,
        target: int,
        csv_out: str,
    ):
        # set up initial class variables
        self.N = n
        self.p_m = p_m
        self.p_c = p_c
        self.trn_size = trn_size
        self.ss_list = weights
        self.ss_len = len(weights)
        self.target = target
        self.csv_out = csv_out

        # run GA
        self.runGA()

    def runGA(self):
        # create initial parent population
        parents = Individual.create_population(
            self.N,
            initialize=create_binary_sequence(self.ss_len),
            decoder=IdentityDecoder(),
            problem=SubsetSumGAFitness(self.ss_list, self.target),
        )

        # evaluate initial population
        parents = Individual.evaluate_population(parents)
        generation_counter = util.inc_generation()

        # open output CSV for writing
        # out_f = open(self.csv_out, "w")
        ss_sol_list = []
        sol_found = 0

        # loop through generations of GA
        while sol_found == 0:
            # create and accumulate offspring
            offspring = pipe(
                parents,
                ops.tournament_selection(k=self.trn_size),
                ops.clone,
                mutate_bitflip(probability=self.p_m),
                ops.uniform_crossover(p_xover=self.p_c),
                ops.evaluate,
                ops.pool(size=len(parents)),  # accumulate offspring
                # probe.AttributesCSVProbe(stream=out_f, do_fitness=True, do_genome=True),
            )

            # update population and increment to next generation
            parents = offspring

            # update ss_sol
            ss_sol_list = [np.dot(self.ss_list, i.genome) for i in parents]
            if self.target in ss_sol_list:
                sol_found = 1

            # increment generation
            generation_counter()

        # close output CSV
        # out_f.close()
        self.solution = parents[ss_sol_list.index(self.target)].genome


# GA implementation based on Wong's paper
# Uses difference degree to determine mutation/crossover values
class SubsetSumGAWong:
    def __init__(self, n: int, d_deg: float, mu: float, weights: list, target: int):
        # set up initial class variables
        self.N = n
        self.d_deg = d_deg
        self.mu = mu
        self.ss_list = weights
        self.ss_len = len(weights)
        self.target = target

        # run GA
        self.runGA()

    # perform crossover based on probability
    def crossover(self, pair, p_c: float):
        if np.random.uniform(low=0.0, high=1.0) <= p_c:
            crossover_index = np.random.randint(low=0, high=self.ss_len - 1)
            tmp0, tmp1 = pair[0].genome.copy(), pair[1].genome.copy()
            pair[0].genome[crossover_index:] = tmp0[crossover_index:]
            pair[1].genome[crossover_index:] = tmp1[crossover_index:]
            return Individual.evaluate_population(pair)
        else:
            return None

    # perform mutation based on probability
    def mutation(self, pair, p_m: float):
        for i in pair:
            for j in range(self.ss_len):
                if np.random.uniform(low=0.0, high=1.0) <= p_m:
                    i.genome[j] = 1 - i.genome[j]
        return Individual.evaluate_population(pair)


    def updatePopulation(self, parents, d_thresh):
        # initialize number of children
        children = []
        indices = [i for i in range(len(parents))]

        # enter loop to generate child population
        # TODO reconfigure parents array
        while (1):

            # get parent pairs, and find difference degrees
            parent_pairs = [np.random.choice(indices, 2, replace=False) for i in range(int((self.N - len(children))/2))]
            n_ds = [np.sum(parents[i[0]].genome == parents[i[1]].genome) for i in parent_pairs]
            d_degs = [i/self.ss_len for i in n_ds]

            # loop over pairs and examine
            for i in range(len(parent_pairs)):
                
                # perform crossover on viable pairs
                if d_degs[i] > d_thresh:
                    child_pair = self.crossover([parents[parent_pairs[i][0]], parents[parent_pairs[i][1]]], 1.0)
                    children.append(child_pair[0])
                    children.append(child_pair[1])

            if len(children) == self.N:
                break
            else:
                for i in range(len(parent_pairs)):
                    
                    # perform mutation on viable pairs
                    if d_degs[i] < d_thresh:
                        mutated_pair = self.mutation([parents[parent_pairs[i][0]], parents[parent_pairs[i][1]]], 1.0)
                        parents[parent_pairs[i][0]], parents[parent_pairs[i][1]] = mutated_pair[0], mutated_pair[1]

        # return updated population
        return children

    def runGA(self):
        # create initial parent population
        parents = Individual.create_population(
            self.N,
            initialize=create_binary_sequence(self.ss_len),
            decoder=IdentityDecoder(),
            problem=SubsetSumGAFitness(self.ss_list, self.target),
        )

        # evaluate initial population
        parents = Individual.evaluate_population(parents)
        generation_counter = util.inc_generation()

        # open output CSV for writing
        # out_f = open(self.csv_out, "w")
        ss_sol_list = []
        sol_found = 0

        # loop through generations of GA
        while sol_found == 0:
            # update population
            parents = self.updatePopulation(parents, self.d_deg)

            # update ss_sol
            ss_sol_list = [np.dot(self.ss_list, i.genome) for i in parents]
            if self.target in ss_sol_list:
                sol_found = 1

            # increment generation
            self.d_deg = self.d_deg*self.mu
            generation_counter()

        # close output CSV
        # out_f.close()
        self.solution = parents[ss_sol_list.index(self.target)].genome
