import cec2017 as c
import numpy as np
import matplotlib.pyplot as plt
import csv
from get_F import Get_F
from RSA import RSA
from Function_plot import func_plot
import os

def run_and_collect(F_name, Solution_no, Max_iter, runs=50):
    F_obj, LB, UB, Dim = c.Get_F(F_name)
    fitness_values = []
    convergence_curves = []

    for run in range(runs):
        Best_F, Best_P, Convergence_curve = RSA(Solution_no, Max_iter, LB, UB, Dim, F_obj)
        fitness_values.append(Best_F)
        convergence_curves.append(Convergence_curve)

    median_val = np.median(fitness_values)
    std_val = np.std(fitness_values)

    # Plot convergence graph for the last run
    plt.figure(figsize=(10, 5))
    plt.plot(convergence_curves[-1], 'b-', linewidth=2)
    plt.title(f"Convergence Curve - {F_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Value")
    plt.grid(True)
    plt.tight_layout()

    # # Save plot to file
    # if not os.path.exists("convergence_plots"):
    #     os.makedirs("convergence_plots")
    # plt.savefig(f"convergence_plots/convergence_{F_name}.png")
    # plt.close()

    return F_name, median_val, std_val

def main():
    Solution_no = 50
    Max_iter = 1200
    
    results = [("Function", "Median", "StdDev")]

    for i in range(1,2):
        F_name = f"F{i}"
        print(f"\nRunning {F_name}...")
        F, median_val, std_val = run_and_collect(F_name, Solution_no, Max_iter)
        results.append((F, f"{median_val:.6e}", f"{std_val:.6e}"))

    # Write results to CSV
    with open("cec2017_rsa_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(results)
        

    print("\n✅ All CEC2017 functions processed. Results saved to 'cec2017_rsa_results.csv'.")

if __name__ == "__main__":
    main()
