import glob
import os
import numpy as np
import pandas as pd
import time

def gather_data() -> None:
    count = 0
    for pkl in [4,8,12,16]:
        for bc in [2,3,4,5]:
            for n in [32,64,128,256]:
                #for pm in [0.001, 0.01, 0.1, 0.33]:
                for i in range(25):
                    csv_out = f'{i+1}-key_len{pkl}-block_count{bc}-population{n}' #-mut_chance{pm}'
                    results = [i.strip('\n') for i in os.popen('python3 -m cryptanalysis.main' + 
                                                                " --pub_key_l " + str(pkl) +
                                                                " --num_blocks " + str(bc) +
                                                                " --pop " + str(n) +
                                                                #" --p_m " + str(pm) +
                                                                " --csv_output " + csv_out
                                                                ) ]
                    if '2' in results[0]:
                        # TODO: track time to complete for each here
                        pass

                    print ( f'Number #{i+1}, key_len{pkl}, block_count{bc}, population{n} COMPLETE!' )
                    count += 1
                    total = len(pkl) * len(bc) * len(n) * 25
                    print ( f'Progress :: {count}/{total} :: {(count/total)*100}%')

def normalize(target: float, max: int, min: int) -> float:
    return abs( (target - min) / (max - min) )

def compile_csvs():
    dir_name = 'cryptanalysis_csvs/'

    # Get list of all files in a given directory sorted by name
    list_of_files = sorted( filter( os.path.isfile,
                            glob.glob(dir_name + '*') ) )

    datum = []
    count = 0

    start = time.time()
    for file_path in list_of_files:

        current = time.time()
        # get important vaiables out of the file name
        garbage, goodies = file_path.split('csvs/')

        # '1-key_len4-block_count2-population32-b0.csv
        num, pkl, bc, pop, block = goodies.split('-')
        pkl = pkl.replace('key_len', '')
        bc = bc.replace('block_count', '')
        pop = pop.replace('population', '')
        block = block.replace('.csv', '')

        print(file_path)

        # open up the file as df
        df = pd.read_csv(file_path)

        # drop plaintext rows
        df = df[df['step'] != 'step']

        # convert datatypes from strings
        df['step'] = df['step'].astype('int')
        df['fitness'] = df['fitness'].astype('float')
        
        # get fitness values
        fitBest = normalize( df['fitness'].min(), df['fitness'].min(), df['fitness'].max() )
        fitAvg  = normalize( df['fitness'].mean(), df['fitness'].min(), df['fitness'].max() )

        print(fitBest, df['fitness'].min())
        print(fitAvg, df['fitness'].mean())

        # track number of solutions found
        if fitBest != df['fitness'].min():
                solFound = False
                numSolFound = 0
        else:
            solFound = True
            numSolFound = len(df[df['fitness'] == 0.0])

        # get # of generations to converge
        conv_time = df['step'].max()
            
        datum.append([num, pkl, bc, pop, block, conv_time, fitBest, fitAvg, solFound, numSolFound])
        print(datum)
        break
        
        count += 1
        print(f"csv {count} of {len(list_of_files)} finished in {time.time()-current}")

    cols = ['i', 
            'pub_key_len',
            'block_count', 
            'population',
            'block_num',
            'convergence_time',
            'best_fitness',
            'average_fitness',
            'solution_found',
            'num_solutions_found',
            ]

    final_df = pd.DataFrame(data=datum, columns=cols)
    final_df.to_csv('cryptanalysis_csvs/0-compiled.csv', index=False)

    print(f"finished in {(time.time()-start)/60} minutes")

    
if __name__ == '__main__':
    compile_csvs()