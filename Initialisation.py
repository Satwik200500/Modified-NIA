import numpy as np

def initialization(N, Dim, UB, LB):
    X = np.zeros((N, Dim))
    B_no = len(UB)  # Number of boundaries

    if B_no == 1:
        X = np.random.rand(N, Dim) * (UB[0] - LB[0]) + LB[0]
    else:
        for i in range(Dim):
            Ub_i = UB[i]
            Lb_i = LB[i]
            X[:, i] = np.random.rand(N) * (Ub_i - Lb_i) + Lb_i

    return X

def main():
    N, Dim = 5, 3
    UB = [10, 20, 30]
    LB = [1, 2, 3]

    result = initialization(N, Dim, UB, LB)

    # Print the result
    for row in result:
        print(" ".join(map(str, row)))

if __name__ == "__main__":
    main()
