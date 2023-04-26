import os

def main_call() -> None:
    for pkl in [4,8,12,16]:
        for bc in [2,3,4,5]:
            for n in [32,64,128,256]:
                for pc in [0.001, 0.01, 0.1, 0.33]:
                    for i in range(25):
                        csv_out = f'{i+1}-key_len{pkl}-block_count{bc}-population{n}-crossover_chance{pc}'
                        os.system('python3 -m cryptanalysis.main' + 
                                " --pub_key_l " + str(pkl) +
                                " --num_blocks " + str(bc) +
                                " --pop " + str(n) +
                                " --p_m " + str(pc) +
                                " --csv_output " + csv_out + 
                                " --verbose False"
                                )

                        print ( f'Number #{i+1}, key_len{pkl}, block_count{bc}, population{n}, crossover_chance{pc} COMPLETE!' )
if __name__ == '__main__':
    main_call()
    