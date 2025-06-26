import cec2017 as c
import numpy as np
import csv
from get_F import Get_F
from scipy.stats import rankdata

# Hybrid RSA+PSO within-loop version
def RSA_PSO_Hybrid(Solution_no, Max_iter, LB, UB, Dim, F_obj):
    X = np.random.uniform(LB, UB, (Solution_no, Dim))
    V = np.zeros((Solution_no, Dim))
    Pbest = X.copy()
    Pbest_fit = np.array([F_obj(x) for x in X])
    Gbest_index = np.argmin(Pbest_fit)
    Gbest = Pbest[Gbest_index].copy()
    Gbest_fit = Pbest_fit[Gbest_index]

    Convergence_curve = []

    w = 0.9
    c1 = 1.5
    c2 = 1.5

    for t in range(Max_iter):
        for i in range(Solution_no):
            rand = np.random.rand(Dim)
            elite_move = Gbest + rand * (X[i] - Gbest)

            r1 = np.random.rand(Dim)
            r2 = np.random.rand(Dim)
            V[i] = w * V[i] + c1 * r1 * (Pbest[i] - X[i]) + c2 * r2 * (Gbest - X[i])
            pso_move = X[i] + V[i]

            new_pos = 0.5 * elite_move + 0.5 * pso_move
            new_pos = np.clip(new_pos, LB, UB)

            fit = F_obj(new_pos)
            if fit < Pbest_fit[i]:
                Pbest[i] = new_pos
                Pbest_fit[i] = fit
                if fit < Gbest_fit:
                    Gbest = new_pos
                    Gbest_fit = fit

            X[i] = new_pos

        Convergence_curve.append(Gbest_fit)
        w = 0.9 - t * ((0.9 - 0.4) / Max_iter)

    return Gbest_fit, Gbest, Convergence_curve


def run_algorithm(Solution_no, Max_iter, F_name, max_runs=10):
    all_fitness = []

    F_obj, LB, UB, Dim = c.Get_F(F_name)

    for run in range(max_runs):
        Best_F, _, _ = RSA_PSO_Hybrid(Solution_no, Max_iter, LB, UB, Dim, F_obj)
        all_fitness.append(Best_F)

    fitness_mean = np.mean(all_fitness)
    fitness_std = np.std(all_fitness)
    return fitness_mean, fitness_std


def main():
    Solution_no = 50
    Max_iter = 1200
    max_runs = 10

    results = []

    for i in range(2, 3):
        F_name = f"F{i}"
        print(f"\n Running {F_name}...")
        mean_fit, std_fit = run_algorithm(Solution_no, Max_iter, F_name, max_runs)
        print(f" Done: Mean = {mean_fit:.6e}, Std = {std_fit:.6e}")
        results.append([F_name, mean_fit, std_fit])

    # Save to CSV
    with open("CEC2014_RSA_PSO_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Function", "Mean Fitness", "Std Deviation"])
        writer.writerows(results)

    print("\n✅ All results saved to 'CEC2014_RSA_PSO_results.csv'.")


if __name__ == "__main__":
    main()