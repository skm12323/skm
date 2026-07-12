import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt
import os

K = tc.set_backend("jax")

c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)

def compute(N):
    s = np.array(c.sample(batch=N, format="sample_int"))

    compute_expectation = 0
    for i in range(N):
        if s[i] == 3 or s[i] == 0:
            compute_expectation += 1/N
        else:
            compute_expectation -= 1/N
    return compute_expectation
'''
def compute_1(N):
    s = np.array(c.sample(batch=N, format="sample_int"))
    signs = np.where((s == 0) | (s == 3), 1, -1)
    return np.mean(signs)
'''
def diff(N):
    return np.abs(compute(N) - 1)

if __name__ == "__main__":
#    Ns = [10, 100, 1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    Ns1 = np.logspace(0, 5, 30, dtype=int)
    Ns2 = [40000, 50000, 60000, 70000, 75000, 80000, 85000, 90000]
    Ns = np.unique(np.concatenate((Ns1, Ns2)))
    plt.figure(figsize=(9, 6))
    for N in Ns:
        plt.scatter(N, diff(N), label=f"N={N}")
    plt.xscale("log")
    plt.xlabel("Number of samples (N)")
    plt.ylabel("Difference between computed and theoretical expectation")
    plt.title("Difference vs Number of Samples")
    plt.legend(ncol=3, fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "expectations_diff_Bell.png",
    )
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.show()