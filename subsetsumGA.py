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
class SubsetSumGA(ScalarProblem):

    def __init__(self, ss_list, target_sum):
        super().__init__(maximize=False)
        self.ss_list = np.array(ss_list)
        self.target_sum = target_sum
        print(self.ss_list)

    def eval_sum(self, ind):
        return np.dot(ind, self.ss_list)

    # TODO: define fitness function and see if this works
    def evaluate(self, ind):
        ss_sum = self.eval_sum(ind)
        s = 0
        if self.target_sum - ss_sum >= 0:
            s = 1
        return (s*(self.target_sum - ss_sum) + (1 - s)*ss_sum)


if __name__ == "__main__":

    # set up argument parsing
    parser = argparse.ArgumentParser(description="Subset Sum Solver")
    parser.add_argument("--n", default=50, help="GA Population Size", type=int)
    parser.add_argument(
        "--p_m", default=0.01, help="GA Probability of Mutation", type=float
    )
    parser.add_argument(
        "--p_c", default=0.3, help="GA Probability of Crossover", type=float
    )
    parser.add_argument("--trn_size", default=2, help="GA Tournament Size", type=int)
    parser.add_argument(
        "--csv_output", required=True, help="GA CSV Output File Name", type=str
    )
    parser.add_argument("--list", required=True, default=[], help="GA Subset Sum List", type=int, nargs="+")
    parser.add_argument("--target", required=True, default=0, help="GA Subset Sum Bound", type=int)

    # parse arguments
    args = parser.parse_args()
    N = args.n
    p_m = args.p_m
    p_c = args.p_c
    trn_size = args.trn_size
    ss_list = args.list
    ss_len = len(args.list)
    target = args.target
    max_generation = 30 # TODO: change this later

    # create initial parent population
    parents = Individual.create_population(
        N,
        initialize=create_binary_sequence(ss_len),
        decoder=IdentityDecoder(),
        problem=SubsetSumGA(ss_list, target)
    )

    # evaluate initial population
    parents = Individual.evaluate_population(parents)
    generation_counter = util.inc_generation()

    # open output CSV for writing
    out_f = open(args.csv_output, "w")

    # loop through generations of GA
    while generation_counter.generation() < max_generation:

        # create and accumulate offspring
        offspring = pipe(
            parents,
            ops.tournament_selection(k=trn_size),
            ops.clone,
            mutate_bitflip(probability=p_m),
            ops.uniform_crossover(p_xover=p_c),
            ops.evaluate,
            ops.pool(size=len(parents)),  # accumulate offspring
            probe.AttributesCSVProbe(stream=out_f, do_fitness=True, do_genome=True),
        )

        # update population and increment to next generation
        parents = offspring
        generation_counter()

    # close output CSV
    out_f.close()