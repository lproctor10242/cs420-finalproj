from merkle_hellman.merklehellman import MerkleHellman
import random
import time


if __name__ == "__main__":

    # open output CSV
    fd = open("./merkle_hellman/data/runtimes.csv", "a")
    fd.write("ga_model, pk_len, plaintext, cipher, decrypted_cipher, encrypt_time, decrypt_time\n")

    # get runtime data for different combinations (10 for each combination)
    for model in ["none", "simple"]:
        for i in [6]: # [1, 2, 4, 8, 10, 12, 14, 16]
            for j in range(10):

                # initialize MH class and plaintext
                mh = MerkleHellman(i, 1000, model)
                plaintext = random.getrandbits(i)

                # get time to encrypt
                start = time.time()
                cipher = mh.encrypt([plaintext])
                end = time.time()
                encrypt_time = end - start

                # get time to decrypt
                start = time.time()
                decrypted_cipher = mh.decrypt(cipher)
                end = time.time()
                decrypt_time = end - start

                # write results to output file
                fd.write(",".join([model, str(i), str(plaintext), str(cipher[0]), str(decrypted_cipher[0]), str(encrypt_time), str(decrypt_time)]) + "\n")
                print("Finished:", model, str(i), str(j))

    # close output CSV
    fd.close()


