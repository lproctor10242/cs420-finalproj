# Authored by Sam Dahms, with script sourced from Dr. Catherine Schuman
# 
# This script performs cryptanalysis on the Merkle-Hellman Knapsack Cipher using Genetic Algorithms

from leap_ec import Individual, context, test_env_var
from leap_ec import ops, probe, util
from leap_ec.decoder import IdentityDecoder
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.binary_rep.problems import ScalarProblem

from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import sys
import time
from toolz import pipe

# Implementation of a custom genetic problem using leap
class Problem(ScalarProblem):
    def __init__(self, public_key, cipher):
        super().__init__(maximize=False)

        self.pk = public_key
        self.pk_l = len(public_key)

        self.c = cipher
        
    def evaluate(self, ind):

        # convert individual genome into binary string
        binstring = np.array2string(ind, separator='')
        binstring = binstring[1:]
        binstring = binstring[:-1] 
        
        # calculate fitness and return
        scores = []
        for i in range(0,len(self.c)):
            binarray = np.array( list(binstring[i*self.pk_l : (i+1)*self.pk_l]), dtype=int )
            pkarray = np.array( self.pk, dtype=float )
            # multiply each bit of the genome by its respective element in the public key array
            score = np.multiply(binarray, pkarray)
            # sum that to get the score
            scores.append( np.sum(score) )

        c = np.array(self.c)
        s = np.array(scores)
        
        # subtract the scores from their respective cipher value and return the absolute value
        return np.sum( np.absolute(np.subtract(c,s)) )

# handler for the genetic analysis
class geneticCryptanalysis:
    def __init__(self, 
                plaintext: str,
                public_key: list[int], 
                ciphertext: list[int], 
                pop: int ,
                p_m: float,
                p_c: float,
                trn_size: int,
                max_iter: int,
                csv_output: str,
                verbose: str
                ) -> None:
        
        block_count = len(ciphertext)

        # self.pt is not the plaintext, but a list of plaintext blocks
        blocksize = int( round( len(plaintext) / block_count ) )
        self.pt = []
        for i in range (0, block_count):
            if i == block_count-1:
                self.pt.append( plaintext[blocksize*i:] )
            else:
                self.pt.append( plaintext[blocksize*i:blocksize*(i+1)] )
        
        if self.pt[-1] == []:
            self.pt.remove([])

        # self.c is not the cipher, but a list of cipher blocks
        blocksize = int( round( len(ciphertext) / block_count ) )
        self.c = []
        for i in range(0,block_count):
            if i == block_count-1:
                self.c.append( ciphertext[blocksize*i:] )
            else:
                self.c.append( ciphertext[blocksize*i:blocksize*(i+1)] )
        
        if self.c[-1] == []:
            self.c.remove([])

        self.pk = public_key
        self.n = pop
        self.pm = p_m
        self.pc = p_c
        self.trn = trn_size
        self.max_iter = max_iter

        self.csv_out = csv_output
        self.v = verbose

        # start a timer
        start = time.time()
       
        # parallelize to solve each block simultaneously
        with Pool() as pool:
            solutions = list(pool.map(self.geneticDecrypt, zip(list(range(len(self.c))), self.c)))

        # stop the timer
        if self.v:
            end = time.time()
            print(f'\nTime to complete: {end-start}\n')

        # concatenate all of the solutions
        self.solution = ''.join(x for x in solutions)

        return None

    def geneticDecrypt(self, t: tuple[int, list[int]]) -> str:
        i = t[0]
        c = t[1]
        
        if self.v:
            print(f'Beginning Genetic Decryption Process for Block {i}')

        genome_length = len(self.pk)
        parents = Individual.create_population(self.n,
                                            initialize=create_binary_sequence(genome_length),
                                            decoder=IdentityDecoder(),
                                            problem=Problem(self.pk, c)
                                            )

        # Evaluate initial population
        parents = Individual.evaluate_population(parents)
        
        # begin generation counts
        generation_counter = util.inc_generation()

        #open csv for writing
        out_f = open("cryptanalysis_csvs/" + self.csv_out + f"-b{i}.csv", "w")
        
        # tracks if solution has been found by evaluating the genome of the best invdividual of the last population
        best_genome = ''
        while best_genome != self.pt[i]:
            # create offspring from the parents, mutate, and evaluate
            offspring = pipe(parents,
                            ops.tournament_selection(k=self.trn),
                            ops.clone,
                            mutate_bitflip(probability=self.pm),
                            ops.uniform_crossover(p_xover=self.pc),
                            ops.evaluate,
                            # accumulate offspring, then write [generation, fitness, genome] of each new offspring to csv file
                            ops.pool(size=len(parents)),
                            probe.AttributesCSVProbe(stream=out_f, do_fitness=True, do_genome=True)
                            )
            
            # evaluate best individual of a population to help determine if we've found the solution
            best = probe.best_of_gen(offspring) 
            best_genome, best_fitness = str(best).split('] ')
            best_genome = best_genome.replace('[','')
            best_genome = best_genome.replace(' ','')
            best_genome = best_genome[::-1]

            parents = offspring

            # increment to the next generation
            generation_counter()  
            if generation_counter.generation() == self.max_iter:
                print(f"No solution for block {i} found in {generation_counter.generation()} generations :(")
                return ('0'*genome_length)
        
        if self.v:
            print(f"Solution for block {i} found in {generation_counter.generation()} generations!")
        
        out_f.close() 
        return best_genome

# for testing purposes
def xnor (s1,s2):
    return ''.join(str(~(ord(a) ^ ord(b))+2) for a,b in zip(s1,s2))

# for testing purposes
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

