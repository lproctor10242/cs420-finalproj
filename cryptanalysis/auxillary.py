import os

def main_call() -> None:
    for pkl in [8,16,24]:
        for bc in [2,3,4,5]:
            for n in [32,64,128,256]:
                for pc in [0.01, 0.1, 0.5, 0.9]:
                    for i in range(25):
                        csv_out = f'{i+1}-k{pkl}-bc{bc}-n{n}-pc{pc}'
                        print('python3 -m cryptanalysis.main' + 
                                " --pub_key_l " + str(pkl) +
                                " --num_blocks " + bc +
                                " --pop " + str(n) +
                                " --p_c " + str(pc) +
                                " --csv_output " + csv_out + 
                                " --verbose True"
                                )

if __name__ == '__main__':
    main_call()
    