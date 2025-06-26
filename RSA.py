import numpy as np

# Function to initialize the positions of solutions
def initialization(N, Dim, UB, LB):
    return np.random.uniform(LB, UB, (N, Dim))

# Main RSA function with soft reinjection and elitism
def RSA(N, T, LB, UB, Dim, F_obj):
    Best_P = np.zeros(Dim)
    Best_F = float('inf')
    X = initialization(N, Dim, UB, LB)
    Xnew = np.zeros((N, Dim))
    Conv = np.zeros(T)

    t = 1
    Alpha = 0.1
    Beta = 0.005
    Ffun = np.zeros(N)
    Ffun_new = np.zeros(N)
    last_improvement_iter = 0

    for i in range(N):
        Ffun[i] = F_obj(X[i, :])
        if Ffun[i] < Best_F:
            Best_F = Ffun[i]
            Best_P = X[i, :]

    while t <= T:
        ES = 2 * np.random.randint(-1, 2) * (1 - (t / T))

        for i in range(1, N):
            for j in range(Dim):
                R = (Best_P[j] - X[np.random.randint(N), j]) / (Best_P[j] + np.finfo(float).eps)
                P = Alpha + (X[i, j] - np.mean(X[i, :])) / (Best_P[j] * (UB - LB) + np.finfo(float).eps)
                Eta = Best_P[j] * P

                if t < T / 4:
                    Xnew[i, j] = Best_P[j] - Eta * Beta - R * np.random.rand()
                elif t < 2 * T / 4:
                    Xnew[i, j] = Best_P[j] * X[np.random.randint(N), j] * ES * np.random.rand()
                elif t < 3 * T / 4:
                    Xnew[i, j] = Best_P[j] * P * np.random.rand()
                else:
                    Xnew[i, j] = Best_P[j] - Eta * np.finfo(float).eps - R * np.random.rand()

            # Boundary check
            Flag_UB = Xnew[i, :] > UB
            Flag_LB = Xnew[i, :] < LB
            Xnew[i, :] = (Xnew[i, :] * (~(Flag_UB + Flag_LB))) + UB * Flag_UB + LB * Flag_LB

            Ffun_new[i] = F_obj(Xnew[i, :])
            if Ffun_new[i] < Ffun[i]:
                X[i, :] = Xnew[i, :]
                Ffun[i] = Ffun_new[i]

            if Ffun[i] < Best_F:
                Best_F = Ffun[i]
                Best_P = X[i, :]
                last_improvement_iter = t

        # Elitism
        X[0, :] = Best_P
        Ffun[0] = Best_F

        # Soft mutation every 100 iterations
        if t % 100 == 0:
            mut_idx = np.random.choice(N, size=N // 10, replace=False)
            X[mut_idx, :] += np.random.normal(0, 20, size=(len(mut_idx), Dim))

        # Replace worst agents every 200 iterations
        if t % 200 == 0:
            worst_indices = np.argsort(Ffun)[-N // 5:]
            X[worst_indices, :] = initialization(len(worst_indices), Dim, UB, LB)
            for idx in worst_indices:
                Ffun[idx] = F_obj(X[idx, :])

        # Soft stagnation reinjection if no improvement for 500 steps
        if t - last_improvement_iter > 500:
            print("Stagnation detected. Soft reinjection of diversity...")
            reinject_idx = np.argsort(Ffun)[-N // 2:]
            X[reinject_idx, :] += np.random.normal(0, 30, size=(len(reinject_idx), Dim))
            Ffun[reinject_idx] = np.array([F_obj(X[i]) for i in reinject_idx])
            Best_F = np.min(Ffun)
            Best_P = X[np.argmin(Ffun), :]
            last_improvement_iter = t

        Conv[t - 1] = Best_F
        if t % 200 == 0:
            print(f"At iteration {t}, the best solution fitness is {Best_F}")

        t += 1

    return Best_F, Best_P, Conv




