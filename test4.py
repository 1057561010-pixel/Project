import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev
import scipy.constants as sc
from EMpy.modesolvers import FD

try:
    import nk

c_um_ps  = 299.792458
c_nm_ps  = sc.c * 1e-9 * 1e12

core_w, core_h = 0.40, 0.22
nx, ny = 241, 241
x = np.linspace(-1.5*core_w, 1.5*core_w, nx)
y = np.linspace(-1.5*core_h, 1.5*core_h, ny)
boundary = '0000'

x_Al = 0.30
USE_SIO2_SELLMEIER = False

def n_core_AlGaAs(lam_um: float, x_al: float) -> float:
    val = nk.AlGaAs_interp(x_al, lam_um, k=True)
    return float(np.real(val).item())

def n_SiO2(lam_um: float) -> float:
    L2 = lam_um**2
    B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
    C1, C2, C3 = 0.0684043**2, 0.1162414**2, 9.896161**2
    return np.sqrt(1 + B1*L2/(L2 - C1) + B2*L2/(L2 - C2) + B3*L2/(L2 - C3))

def n_clad_of(lam_um: float) -> float:
    return n_SiO2(lam_um) if USE_SIO2_SELLMEIER else 1.44

wls_um = np.linspace(1.45, 1.65, 11)
neigs  = 4
tol    = 1e-10

neff_list = []

for wl in wls_um:
    def epsfunc(xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
        n_core = n_core_AlGaAs(wl, x_Al)
        n_clad = n_clad_of(wl)
        X, Y = np.meshgrid(xv, yv, indexing="ij")
        core_mask = (np.abs(X) <= core_w/2) & (np.abs(Y) <= core_h/2)
        eps = np.full((xv.size, yv.size), n_clad**2, dtype=float)
        eps[core_mask] = n_core**2
        return eps

    solver = FD.VFDModeSolver(wl=wl, x=x, y=y, epsfunc=epsfunc, boundary=boundary)
    solver.solve(neigs=neigs, tol=tol)

    if not hasattr(solver, "modes") or len(solver.modes) == 0:
        raise RuntimeError("No modes found. Check window size/grid/boundary.")

    neff0 = float(np.real(solver.modes[0].neff))
    neff_list.append(neff0)
    print(f"λ = {wl:.3f} µm  →  n_eff(TE0) = {neff0:.6f}")

neff = np.array(neff_list)

def analyze_dispersion_cheb(wls_um: np.ndarray, neff: np.ndarray, deg: int = 4):
    x_nm = np.asarray(wls_um, float) * 1e3
    y    = np.asarray(neff,   float)

    p = Chebyshev.fit(x_nm, y, deg=deg)

    xx_nm, yy = p.linspace(400)
    d1 = p.deriv(1)(xx_nm)
    d2 = p.deriv(2)(xx_nm)

    ng = yy - xx_nm * d1

    D_per_m = -(xx_nm / sc.c) * d2 * 1e12
    D_km    = D_per_m * 1e3

    beta2_ps2_km = -(xx_nm**2 / (2*np.pi*c_nm_ps)) * D_km

    zdw_um = None
    s = np.sign(D_km)
    idx = np.where(s[:-1]*s[1:] < 0)[0]
    if idx.size:
        i = idx[0]
        x0, x1 = xx_nm[i], xx_nm[i+1]
        y0, y1 = D_km[i], D_km[i+1]
        xz = x0 - y0*(x1 - x0)/(y1 - y0)
        zdw_um = xz / 1e3

    return {
        "xx_um": xx_nm/1e3,
        "nfit": yy,
        "ng": ng,
        "D_ps_nm_km": D_km,
        "beta2_ps2_km": beta2_ps2_km,
        "ZDW_um": zdw_um,
        "cheb_poly": p
    }

res = analyze_dispersion_cheb(wls_um, neff, deg=4)

print("\n=== Summary (Chebyshev) ===")
print(f"x_Al={x_Al:.2f}, core_w={core_w:.3f} µm, core_h={core_h:.3f} µm, "
      f"clad={'SiO2(Sellmeier)' if USE_SIO2_SELLMEIER else 'const 1.44'}")
if res["ZDW_um"] is not None:
    print(f"Estimated ZDW ≈ {res['ZDW_um']:.3f} µm")
else:
    print("No zero-dispersion crossing in scanned range.")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(wls_um, neff, 'o', label='n_eff data')
plt.plot(res["xx_um"], res["nfit"], '-', label='Chebyshev fit')
plt.xlabel('Wavelength (µm)'); plt.ylabel('n_eff')
plt.title('Effective index and Wavelength')
plt.grid(True); plt.legend()

plt.subplot(1,2,2)
plt.plot(res["xx_um"], res["D_ps_nm_km"], '-', label='D  [ps/(nm·km)]')
plt.axhline(0, color='gray', ls=':')
if res["ZDW_um"] is not None:
    plt.axvline(res["ZDW_um"], ls='--', label=f'ZDW ≈ {res["ZDW_um"]:.3f} µm')
plt.xlabel('Wavelength (µm)'); plt.ylabel('Dispersion D [ps/(nm·km)]')
plt.title('Material and Waveguide Dispersion')
plt.grid(True); plt.legend()

plt.tight_layout(); plt.show()
