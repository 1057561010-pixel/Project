import numpy as np
import matplotlib.pyplot as plt

hc_eV_um = 1.239841984

def poly6(x, coeffs):
    return sum(coeffs[i] * x ** i for i in range(len(coeffs)))

def n_AlGaAs_Gehrsitz_full(lambda_um, x, T=300):

    A_coeff = [10.906, -2.274, 1.262, -0.333, 0.169, -0.025]
    C0_coeff = [0.004481, 0.01308, -0.0402, 0.0816, -0.0577, 0.0140]
    E0_coeff = [3.497, 1.01, 0.532, 1.53, -2.13, 0.743]
    C1_coeff = [0.1202, -0.0310, 0.0455, -0.0802, 0.0745, -0.0245]
    E1_coeff = [4.646, 1.783, -1.779, 2.905, -3.13, 1.099]


    A = poly6(x, A_coeff)
    C0 = poly6(x, C0_coeff)
    E0 = poly6(x, E0_coeff)
    C1 = poly6(x, C1_coeff)
    E1 = poly6(x, E1_coeff)

    E = hc_eV_um / lambda_um

    n2 = A + C0 / (E0 ** 2 - E ** 2) + C1 / (E1 ** 2 - E ** 2)

    return np.sqrt(np.real(n2))


if __name__ == "__main__":
    wls = np.linspace(1.0, 1.8, 300)
    for x in [0.0, 0.3, 0.5, 0.8, 1.0]:
        n_vals = [n_AlGaAs_Gehrsitz_full(lam, x) for lam in wls]
        plt.plot(wls, n_vals, label=f"x={x}")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Refractive index n")
    plt.title("Refractive index of AlₓGa₁₋ₓAs")
    plt.legend()
    plt.grid(True)
    plt.show()
