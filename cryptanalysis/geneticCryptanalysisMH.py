import numpy as np
from toolz import pipe

from leap_ec import Individual, context, test_env_var
from leap_ec import ops, probe, util
from leap_ec.decoder import IdentityDecoder
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.binary_rep.problems import ScalarProblem

import pandas as pd
import time
import sys

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
        for i, c in enumerate(self.c):
            g = binstring[i*self.pk_l : (i+1)*self.pk_l]
            score = 0
            for j, pk in enumerate(self.pk):
                score += self.pk[j]*int(g[j])
            scores.append(score)

        c = np.array(self.c)
        s = np.array(scores)
        
        return np.sum( np.absolute(np.subtract(c,s)) )

class geneticCryptanalysis:
    def __init__(self, 
                public_key: list[int], 
                ciphertext: list[int], 
                block_count: int = 2,
                pop: int = 100,
                p_m: float = 0.1,
                p_c: float = 0.1,
                trn_size: int = 5
                ) -> None:

        self.pk = public_key

        self.c = []
        self.blocksize = int( round( len(ciphertext) / block_count ) )
        for i in range(0,block_count):
            if i == block_count-1:
                self.c.append( ciphertext[self.blocksize*i:] )
            else:
                self.c.append( ciphertext[self.blocksize*i:self.blocksize*(i+1)] )
        
        if self.c[-1] == []:
            self.c.remove([])

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
            out_f = open("block" +str(i)+".csv", "w")
            
            # tracks if solution has been found by evaluating the fitness of the best invdividual of the last population
            best_fitness = 1
            while best_fitness != '0':
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
                
                best = probe.best_of_gen(offspring) 
                
                best_genome, best_fitness = str(best).split('] ')
                if best_fitness == '0':
                    self.solution += best_genome
                    print(f"Solution for block {i} found in {generation_counter.generation()} generations")

                parents = offspring

                # increment to the next generation
                generation_counter()  
            out_f.close() 
        
        end = time.time()
        print(f'Time to complete: {end-start}')
        return None

    # pull solutions from csv files and combine to solve the cipher
    def solveCipher(self) -> str:

        #binary = self.solution
        binary = ''
        for csv in ['block0.csv','block1.csv','block2.csv','block3.csv']:
            # open as a dataframe
            df = pd.read_csv(csv)
            
            # drop plaintext rows
            df = df[df['step'] != 'step']

            # convert datatypes from strings
            df['fitness'] = df['fitness'].astype(int)
            
            # get where fitness is the solution
            df = df.loc[df['fitness'] == 0]
          
            df = df.reset_index()

            # get the genome
            try:
                binary += str( df.at[0, 'genome'] )
            except KeyError:
                return f"Solution not found in {csv[:-4]}, sorry!"
        
        for char in [' ', '[', ']', '\n']:
            binary = binary.replace(char, '')
            
            #plaintext = int(binary, 2)
        return binary 
        #return plaintext.to_bytes((plaintext.bit_length() + 7) // 8, 'big').decode()

    # for troubleshooting purposes
    def gcd(self, a, b):
        while b != 0:
            a, b = b, a % b
        return a

    # for troubleshooting purposes
    def xnor (self, s1,s2):
        return ''.join(str(~(ord(a) ^ ord(b))+2) for a,b in zip(s1,s2))


if __name__ == '__main__':

    gc = geneticCryptanalysis(   
                        public_key = [53,106,92,50,100,80,40,80,40,80,40,40],
                        ciphertext = [326,396,498,341,476,541,498,186],
                        block_count = 4,
                        pop = 50,
                        p_m = 0.1,
                        p_c = 0.8,
                        trn_size = 5
                        )

    #gc.geneticDecrypt()
    #print( gc.solveCipher() )


    #############################
    # FOR TESTING, REMOVE LATER #
    #############################

    '''
    # message = 'Hello World!'
    binary = '010010000110010101101100011011000110111100100000010101110110111101110010011011000110010000100001'
    genome = '010010100100010100111011'
    #print(binary[:24])

    pk = [53,106,92,50,100,80,40,80,40,80,40,40]
        #  0  1   0  0  1   0  0  0  0  1  1  0 = 326 true best fitness (actual plaintext)
        #  0  1   0  0  1   0  1  0  0  1  0  0 = 326 false best fitness
    pk_l = len(pk)
    ciphertext = [326,396]#,498,341]#,476,541,498,186]

    # n = length of public/private key
    # k = length of ciphertext, also # of blocks
    # M = plaintext, split into k blocks m, each the length of n

    # the fitness of for each block i:
    # absolute value of:
    #   the ciphertext for i 
    #   MINUS
    #   (sum j:0 -> n) of pk at j * bit at j


    # calculate fitness and return

    scores = []
    for i, c in enumerate(ciphertext):
        g = str( genome[i*pk_l : (i+1)*pk_l] )
        print(g)
        score = 0
        for j, p in enumerate(pk):
            score += p*int(g[j])
        scores.append(score)
    
    c = np.array(ciphertext[:4])
    s = np.array(scores)

    print(c)
    print(s)

    sub = np.subtract(c,s)
    abs_sum = np.sum( np.absolute(sub) )
    
    print("=============")
    print("FITNESS SCORE")
    print(abs_sum)
    '''