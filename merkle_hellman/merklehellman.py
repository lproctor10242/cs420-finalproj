# imports
import random
import math
import os
import numpy as np
from subset_sum.subsetsumGA import SubsetSumGADriver
import sys

# class for merkle hellman cryptosystem implementation
class MerkleHellman():

    # class init
    def __init__(self, pt_len: int):
        self.l = pt_len
        self.weights = self.genPublicWeights()
        self.q = self.genQ()
        self.r = self.genR()
        self.pub_key = self.genPublicKey()
        self.private_key = self.genPrivateKey()

    # function to generate public weights
    def genPublicWeights(self):
        weights = []
        for i in range(self.l):
            weights.append(sum(weights) + random.randint(0, sys.maxsize))
        return weights

    # get random q
    def genQ(self):
        if self.weights != []:
            return sum(self.weights) + random.randint(0, sys.maxsize)
        else:
            return None

    # get r such that q,r are coprime (i.e. gcd(q,r) = 1)
    def genR(self):
        if self.q:
            while True:
                r = random.randint(0, sys.maxsize)
                if math.gcd(r, self.q) == 1:
                    return r
        else:
            return None

    # generate the public key
    def genPublicKey(self):
        if self.weights and self.q and self.r:
            pub_key = []
            for i in range(self.l):
                pub_key.append((self.r * self.weights[i]) % self.q)
            return pub_key
        else:
            return None

    # generate private key
    def genPrivateKey(self):
        if self.weights and self.q and self.r:
            return (self.weights, self.q, self.r)
        else:
            return None

    # encrypt function
    def encrypt(self, plaintext: int):
        if self.pub_key:
            cipher = 0
            for i in range(self.l):
                cipher += ((plaintext >> i) & 1) * self.pub_key[i]
            return cipher
        else:
            return None

    # decrypt function
    def decrypt(self, cipher: int):

        # get r^-1 and form the subset sum target (c_new)
        r_inv = pow(self.r, -1, self.q)
        c_new = (cipher*r_inv) % self.q

        # run GA to solve subset sum problem
        ss_solution = SubsetSumGADriver(200, float(1/self.l), 0.7, int(0.2*self.l), self.weights, c_new, "output.csv").solution

        # decrypt cipher
        return int("".join(str(i) for i in ss_solution)[::-1], base=2)
        

if __name__ == "__main__":

    mh = MerkleHellman(8)
    plaintext = ord("a")
    print("Plaintext:", plaintext)
    cipher = mh.encrypt(plaintext)
    print("Cipher:", cipher)
    plaintext = mh.decrypt(cipher)
    print("Decrypted Cipher:", plaintext)

    



    
