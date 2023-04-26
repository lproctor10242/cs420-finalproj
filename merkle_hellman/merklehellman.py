# imports
import random
import math
from subset_sum.subsetsumGA import SubsetSumGASimple, SubsetSumGAWong
import sys


# class for merkle hellman cryptosystem implementation
class MerkleHellman:
    # class init
    def __init__(self, pk_len: int, int_max, ga_model: str):
        self.pk_len = pk_len
        self.int_max = int_max
        self.ga_model = ga_model
        self.weights = self.genPublicWeights()
        self.q = self.genQ()
        self.r = self.genR()
        self.pub_key = self.genPublicKey()
        self.private_key = self.genPrivateKey()

    # function to generate public weights
    def genPublicWeights(self):
        weights = []
        for i in range(self.pk_len):
            weights.append(sum(weights) + random.randint(0, self.int_max))
        return weights

    # get random q
    def genQ(self):
        if self.weights != []:
            return sum(self.weights) + random.randint(0, self.int_max)
        else:
            return None

    # get r such that q,r are coprime (i.e. gcd(q,r) = 1)
    def genR(self):
        if self.q:
            while True:
                r = random.randint(0, self.q-1)
                if math.gcd(r, self.q) == 1:
                    return r
        else:
            return None

    # generate the public key
    def genPublicKey(self):
        if self.weights and self.q and self.r:
            pub_key = []
            for i in range(self.pk_len):
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
    def encrypt(self, plaintext: list):
        if self.pub_key:
            ciphers = []
            for i in plaintext:
                block_cipher = 0
                for j in range(self.pk_len):
                    block_cipher += ((i >> j) & 1) * self.pub_key[j]
                ciphers.append(block_cipher)
            return ciphers
        else:
            return None

    # decrypt function
    def decrypt(self, ciphers: list):

        # create plaintext array
        plaintext = []

        # decrypt cipher blocks
        for i in ciphers:
            # get r^-1 and form the subset sum target (c_new)
            r_inv = pow(self.r, -1, self.q)
            c_new = (i * r_inv) % self.q

            # run GA to solve subset sum problem
            if self.ga_model == "simple":
                ss_solution = SubsetSumGASimple(
                    200,
                    float(1 / self.pk_len),
                    0.7,
                    int(0.2 * self.pk_len),
                    self.weights,
                    c_new,
                    "output.csv",
                ).solution
            elif self.ga_model == "wong":
                ss_solution = SubsetSumGAWong(
                    200,
                    0.6,
                    0.999,
                    self.weights,
                    c_new,
                ).solution

            # append decrypted block to plaintext array
            plaintext.append(int("".join(str(j) for j in ss_solution)[::-1], base=2))

        # return decrypted message blocks
        return plaintext


if __name__ == "__main__":
    mh = MerkleHellman(8, 1000, "simple")
    plaintext = [ord("a"), ord("b")]
    print("Plaintext:", plaintext)
    print("Public Key:", mh.pub_key)
    cipher = mh.encrypt(plaintext)
    print("Cipher:", cipher)
    plaintext = mh.decrypt(cipher)
    print("Decrypted Cipher:", plaintext)
