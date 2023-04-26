# Authored by Sam Dahms
#
# invoke with 'python3 -m cryptanalysis.main'

from cryptanalysis.genetic_cryptanalysis_MH import geneticCryptanalysis
from merkle_hellman.merklehellman import MerkleHellman

import argparse
import random
import string
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Cryptanalysis of the Merkle-Hellman Knapsack Cipher utilizing a Genetic Algorithm")

    parser.add_argument("--pub_key_l",  default=8,        help="public key length",        type=int)
    parser.add_argument("--num_blocks", default=2,        help="how many parallel splits", type=int)
    parser.add_argument("--int_max",    default=1000000,  help="maxsize of integers",      type=int)
    parser.add_argument("--ss_model",   default='simple', help="subset sum model, can be simple/wong/none", type=str)

    parser.add_argument("--pop",        default=64,       help="population size",          type=int)
    parser.add_argument("--p_m",        default=0.1,      help="probability of mutation",  type=float)
    parser.add_argument("--p_c",        default=0.7,      help="probability of crossover", type=float)
    parser.add_argument("--trn_size",   default=4,        help="tournament size",          type=int)
    parser.add_argument("--max_iter",   default=2500,     help="max generations",           type=int)

    parser.add_argument("--csv_output", required=True,    help="csv output file name",     type=str)
    parser.add_argument("--verbose",    default=False,    help="verbosity",                type=str)
    
    args = parser.parse_args()    

    pkl = args.pub_key_l
    num_blocks = args.num_blocks

    plaintext_binary = ''
    plaintext_decimal = []
    for _ in range(num_blocks):
        x = random.getrandbits(pkl)
        plaintext_decimal.append(x)
        plaintext_binary += str( bin(x)[2:] ).zfill(int(pkl))

    mh = MerkleHellman(pkl, args.int_max, args.ss_model)
    cipher = mh.encrypt(plaintext_decimal)

    v = args.verbose

    gc = geneticCryptanalysis( 
                        plaintext = plaintext_binary,
                        public_key = mh.pub_key,
                        ciphertext = cipher,
                        pop = args.pop,
                        p_m = args.p_m,
                        p_c = args.p_c,
                        trn_size = args.trn_size,
                        max_iter = args.max_iter,
                        csv_output = args.csv_output,
                        verbose = v
                        )

    print( f"Cryptanalysis Solution :: {gc.solution}" )

    if v:
        print( f"Original Binary String :: {plaintext_binary}")
        

        if pkl == 8:
            decimal_decrypted = mh.decrypt(cipher)
            binary_decrypted = ''.join(bin(x)[2:].zfill(8) for x in decimal_decrypted)
            print( f"MH Decrypted Plaintext :: {binary_decrypted}")

    
    

    