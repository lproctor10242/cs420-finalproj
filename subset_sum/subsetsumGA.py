# imports
import os
import numpy as np
from toolz import pipe
from leap_ec import Individual, context, test_env_var
from leap_ec import ops, probe, util
from leap_ec.decoder import IdentityDecoder
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.binary_rep.problems import ScalarProblem
import argparse
import sys

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
        return (s*(self.target_sum - ss_sum) + (1 - s)*ss_sum)

class SubsetSumGADriver():

    def __init__(self, n: int, p_m: float, p_c: float, trn_size: int, weights: list, target: int, csv_out: str):

        # set up initial class variables
        self.N = n
        self.p_m = p_m
        self.p_c = p_c
        self.trn_size = trn_size
        self.ss_list = weights
        self.ss_len = len(weights)
        self.target = target
        self.max_generation = 30
        self.csv_out = csv_out

        # run GA
        self.runGA()

    def runGA(self):

        # create initial parent population
        parents = Individual.create_population(
            self.N,
            initialize=create_binary_sequence(self.ss_len),
            decoder=IdentityDecoder(),
            problem=SubsetSumGAFitness(self.ss_list, self.target)
        )

        # evaluate initial population
        parents = Individual.evaluate_population(parents)
        generation_counter = util.inc_generation()

        # open output CSV for writing
        #out_f = open(self.csv_out, "w")
        ss_sol_list = []
        sol_found = False

        # loop through generations of GA
        while sol_found == False:

            # create and accumulate offspring
            offspring = pipe(
                parents,
                ops.tournament_selection(k=self.trn_size),
                ops.clone,
                mutate_bitflip(probability=self.p_m),
                ops.uniform_crossover(p_xover=self.p_c),
                ops.evaluate,
                ops.pool(size=len(parents)),  # accumulate offspring
                #probe.AttributesCSVProbe(stream=out_f, do_fitness=True, do_genome=True),
            )

            # update population and increment to next generation
            parents = offspring

            # update ss_sol
            ss_sol_list = [np.dot(self.ss_list, i.genome) for i in parents]
            if self.target in ss_sol_list:
                sol_found = True

            # increment generation
            generation_counter()

        # close output CSV
        #out_f.close()
        self.solution = parents[ss_sol_list.index(self.target)].genome