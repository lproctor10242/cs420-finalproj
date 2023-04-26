from merkle_hellman.merklehellman import MerkleHellman

from leap_ec import Individual, context, test_env_var
from leap_ec import ops, probe, util
from leap_ec.decoder import IdentityDecoder
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.binary_rep.problems import ScalarProblem

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
                pop: int = 100,
                p_m: float = 0.1,
                p_c: float = 0.1,
                trn_size: int = 5
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
        self.solution = ''

        return None

    def geneticDecrypt(self) -> None:
        start = time.time()
        for i, c in enumerate(self.c):
            genome_length = len(self.pk)*len(c)
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
            out_f = open("csvs/" + "block" +str(i)+".csv", "w")
            
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
                if best_fitness == '0.0':
                    best_genome = best_genome.replace('[','')
                    best_genome = best_genome.replace(' ','')
                    best_genome = best_genome[::-1]

                parents = offspring

                # increment to the next generation
                generation_counter()  

            self.solution += best_genome
            print(f"Solution for block {i} found in {generation_counter.generation()} generations")
            out_f.close() 
        
        end = time.time()
        print(f'Time to complete: {end-start}')
        return None

# for testing purposes
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# for testing purposes
def xnor (s1,s2):
    return ''.join(str(~(ord(a) ^ ord(b))+2) for a,b in zip(s1,s2))


if __name__ == '__main__':

    mh = MerkleHellman(8)
    plaintext = ord("a")
    print("Plaintext:", plaintext)
    cipher = mh.encrypt(plaintext)
    print("Cipher:", cipher)
    plaintext = mh.decrypt(cipher)
    print("Decrypted Cipher:", plaintext)

    #os.system('python3 -m merkle_hellman.merklehellman')

    message = 'ab'
    binary = ''.join('{0:08b}'.format(ord(x), 'b') for x in message)

    gc = geneticCryptanalysis( 
                        plaintext = binary,
                        public_key = [737275508093970824447, 1092420682627872412413, 982379061352518594124, 16930508469857996252, 988778712680290931315, 1197792263373058555151, 116348984040649682725, 873292266069408452212],
                        ciphertext = [2051416755507679062323, 2406561930041580650289],
                        pop = 50,
                        p_m = 0.1,
                        p_c = 0.8,
                        trn_size = 5
                        )

    gc.geneticDecrypt()
    print( f"message binary: {binary}")
    print( f"found solution: {gc.solution}" )
    