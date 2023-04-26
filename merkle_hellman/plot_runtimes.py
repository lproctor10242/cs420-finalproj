# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # dict of parameters and value ranges to iterate over
    param_ranges = {
        "ga_model": [["none", "simple"], ["Standard", "Genetic Algorithm"]],
        "pk_len": [[2, 4, 6, 8, 10, 12, 14, 16], "Public Key Length"]
    }

    # read file contents
    mh_df = pd.read_csv("./merkle_hellman/data/runtimes.csv")

    # generate runtime comparison graphs
    for model in range(len(param_ranges["ga_model"][0])):
        plot_arr = []
        for i in param_ranges["pk_len"][0]:
            plot_arr.append(mh_df[(mh_df["ga_model"] == param_ranges["ga_model"][0][model]) & (mh_df["pk_len"] == i)]["decrypt_time"])
        mean_vals = [sum(i)/len(i.index) for i in plot_arr]
        plt.boxplot(plot_arr, widths=1, positions=param_ranges["pk_len"][0], labels=param_ranges["pk_len"][0])
        plt.plot(param_ranges["pk_len"][0], mean_vals, label=param_ranges["ga_model"][1][model])
        plt.title("Merkle-Hellman Performance for " + param_ranges["ga_model"][1][model] + " Implementation")
        plt.xlabel(param_ranges["pk_len"][1])
        plt.yticks([-50, 0, 100, 200, 300, 400, 500, 600, 700, 800], labels=["", 0, 100, 200, 300, 400, 500, 600, 700, 800])
        plt.ylabel("MH Decrypt Runtime (s)")
        plt.savefig("./merkle_hellman/figures/" + param_ranges["ga_model"][0][model] + "runtime.png")
        plt.clf()