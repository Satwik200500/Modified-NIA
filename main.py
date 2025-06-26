import numpy as np
import matplotlib.pyplot as plt
from get_F import Get_F
from RSA import RSA
from Function_plot import func_plot
import cec2017 as c

def main():
    # RSA parameters
    Solution_no = 50          # Number of search agents (population size)
    Max_iter = 1200           # Maximum number of iterations
    F_name = 'F8'             # Name of benchmark function (e.g., 'F1', 'F2', ..., 'F23')

    # Load function details (bounds, dimension, and function handle)
    F_obj, LB, UB, Dim = c.Get_F(F_name)

    # Call the Reptile Search Algorithm
    Best_F, Best_P, Convergence_curve = RSA(Solution_no, Max_iter, LB, UB, Dim, F_obj)

    # Display results
    print(f"\\nThe best solution obtained by RSA is:\\n{Best_P}")
    print(f"\\nThe best optimal value of the objective function found by RSA is: {Best_F}")

    # Plot convergence curve
    plt.figure(figsize=(10, 5))
    plt.plot(Convergence_curve, 'b-', linewidth=2)
    plt.title(f"Convergence Curve - {F_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional 2D function surface plot (only for 2D functions)
    if Dim == 2:
        func_plot(F_name)

if __name__ == "__main__":
    main()
