import os
import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

K = tc.set_backend("jax")

def expectation_tc(axis_1, axis_2, theta):
    pauli_matrices = {
        'x': K.stack([K.stack([0, 1]), K.stack([1, 0])]),
        'y': K.stack([K.stack([0, -1j]), K.stack([1j, 0])]),
        'z': K.stack([K.stack([1, 0]), K.stack([0, -1])])
    }
    if axis_1 not in pauli_matrices or axis_2 not in pauli_matrices:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
    
    P = pauli_matrices[axis_1]
    Q = pauli_matrices[axis_2]

    exp_val = K.expm(1j * theta * P/2)
    v_0 = K.stack([1, 0])
    v = K.tensordot(exp_val, v_0, axes=1)
    expectation_value = K.tensordot(K.conj(v), K.tensordot(Q, v, axes=1), axes=1)
    return expectation_value
    
if __name__ == "__main__":
    thetas = np.linspace(0, 2 * np.pi, 400)

    combos = [
        ("x", "x"), ("x", "y"), ("x", "z"),
        ("y", "x"), ("y", "y"), ("y", "z"),
        ("z", "x"), ("z", "y"), ("z", "z"),
    ]

    plt.figure(figsize=(9, 6))
    for a, b in combos:
        vals = [np.real(expectation_tc(a, b, t)) for t in thetas]
        plt.plot(thetas, vals, label=f"rotate {a} -> expect {b}")

    plt.xlabel(r"$\theta$")
    plt.ylabel("Expectation value")
    plt.title("Pauli expectation vs theta for various axis combinations")
    plt.legend(ncol=3, fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pauli_expectations_tc.png",
    )
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.show()
