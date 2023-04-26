# Authored by Sam Dahms
#
# invoke with 'python3 -m cryptanalysis.main'

from cryptanalysis.genetic_cryptanalysis_MH import geneticCryptanalysis
from merkle_hellman.merklehellman import MerkleHellman

import argparse
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Cryptanalysis of the Merkle-Hellman Knapsack Cipher utilizing a Genetic Algorithm")
    
    parser.add_argument("--plaintext",  default='ab',     help="message to crack, MAX LENGTH OF 5 (for now)", type=str)

    parser.add_argument("--pub_key_l",  default=16,       help="public key length",        type=int)
    parser.add_argument("--int_max",    default=1000000,  help="maxsize of integers",      type=int)
    parser.add_argument("--ss_model",   default='simple', help="subset sum model, can be simple/wong/none", type=str)

    parser.add_argument("--pop",        default=50,       help="population size",          type=int)
    parser.add_argument("--p_m",        default=0.01,     help="probability of mutation",  type=float)
    parser.add_argument("--p_c",        default=0.3,      help="probability of crossover", type=float)
    parser.add_argument("--trn_size",   default=2,        help="tournament size",          type=int)

    parser.add_argument("--csv_output", required=False,   help="csv output file name",     type=str)
    parser.add_argument("--verbose",    default=False,    help="verbosity",                type=str)
    
    args = parser.parse_args()    
    
    pt = args.plaintext
    if len(pt) > 6:
        print("--plaintext argument must be 5 characters or less (for now), sorry!")
        sys.exit(1)
    
    plaintext = []
    for ch in pt:
        plaintext.append(ch)

    plaintext_binary = ''.join('{0:08b}'.format(ord(x), 'b') for x in plaintext)
    plaintext_decimal = []
    for x in plaintext:
        plaintext_decimal.append(ord(x))

    pub_key_l = args.pub_key_l
    int_max = args.int_max
    ss_model = args.ss_model

    mh = MerkleHellman(pub_key_l, int_max, ss_model)
    cipher = mh.encrypt(plaintext_decimal)
    
    pop = args.pop
    p_m = args.p_m
    p_c = args.p_c
    trn_size = args.trn_size

    csv_out = args.csv_output
    v = args.verbose
    v = True

    gc = geneticCryptanalysis( 
                        plaintext = plaintext_binary,
                        public_key = mh.pub_key,
                        ciphertext = cipher,
                        pop = pop,
                        p_m = p_m,
                        p_c = p_c,
                        trn_size = trn_size,
                        csv_output = csv_out,
                        verbose = v
                        )

    gc.geneticDecrypt()

    decimal_decrypted = mh.decrypt(cipher)
    binary_decrypted = ''.join(bin(x)[2:].zfill(8) for x in decimal_decrypted)

    print( f"Cryptanalysis Solution :: {gc.solution}" )
    print( f"MH Decrypted Plaintext :: {binary_decrypted}")