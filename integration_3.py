print(">>> FINAL FGM TABLE INTEGRATION SCRIPT <<<")

import os, glob
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.interpolate import griddata
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================

CSV_DIR = r"C:\Users\Admin\Downloads\NEW\02_processed_flamelet_results"
OUT_DIR = "FGM_tables"
os.makedirs(OUT_DIR, exist_ok=True)

NZ, NC = 51, 51
NvarZ, NvarC = 5, 5

Z_grid = np.linspace(0.0, 1.0, NZ)
C_grid = np.linspace(0.0, 1.0, NC)

zetaZ_grid = np.linspace(0.0, 1.0, NvarZ)
zetaC_grid = np.linspace(0.0, 1.0, NvarC)

R_univ = 8314.462618
EPS_PV = 1e-12

# ============================================================
# LOAD CSV FILES
# ============================================================

files = sorted(glob.glob(os.path.join(CSV_DIR, "post_strain_loop_*.csv")))
if not files:
    raise RuntimeError("No CSV files found")

print(f"Loading {len(files)} flamelet CSV files...")
data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# ============================================================
# EXTRACT DATA
# ============================================================

Z  = data["Z"].values
PV = data["PV"].values
T  = data["T[K]"].values

rho   = data["rho[kg/m3]"].values
cp    = data["cp[J/kgK]"].values
mu    = data["mu[Pa s]"].values
alpha = data["alpha[kg/ms]"].values
Mbar  = data["Mbar[kg/kmol]"].values

Y = {
    "CH4"  : data["Y_CH4"].values,
    "O2"   : data["Y_O2"].values,
    "N2"   : data["Y_N2"].values,
    "CO2"  : data["Y_CO2"].values,
    "H2O"  : data["Y_H2O"].values,
    "OH"   : data["Y_OH"].values,
    "CH2O" : data["Y_CH2O"].values,
}

SourcePV = (
    data["wCO2[kg/m3s]"].values +
    data["wH2O[kg/m3s]"].values +
    data["wCO[kg/m3s]"].values +
    data["wH2[kg/m3s]"].values
)

with np.errstate(divide="ignore", invalid="ignore"):
    psi = np.where(T > 10.0, 1.0 / ((R_univ / Mbar) * T), 0.0)

# ============================================================
# Yc ENVELOPE (ROBUST BINNING)
# ============================================================

Yc_u = np.zeros(NZ)
Yc_b = np.zeros(NZ)

idx = np.argsort(Z)
Zs, PVs = Z[idx], PV[idx]
dZ = Z_grid[1] - Z_grid[0]

for i, Zm in enumerate(Z_grid):
    mask = (Zs >= Zm - 0.5*dZ) & (Zs <= Zm + 0.5*dZ)
    if np.any(mask):
        Yc_u[i] = PVs[mask].min()
        Yc_b[i] = PVs[mask].max()
    else:
        j = np.argmin(np.abs(Zs - Zm))
        Yc_u[i] = PVs[j]
        Yc_b[i] = PVs[j]

print("Yc envelope constructed successfully")

# ============================================================
# Z–PV → Z–C
# ============================================================

def compute_C(Z, PV):
    Yu = np.interp(Z, Z_grid, Yc_u)
    Yb = np.interp(Z, Z_grid, Yc_b)
    return (PV - Yu) / (Yb - Yu + EPS_PV)

C = compute_C(Z, PV)

# ============================================================
# SAFE BETA PDF (NO OVERFLOW)
# ============================================================

def beta_pdf(mean, var, grid):
    pdf = np.zeros_like(grid)

    if var <= 1e-14 or mean <= 1e-12 or mean >= 1.0 - 1e-12:
        pdf[np.argmin(np.abs(grid - mean))] = 1.0
        return pdf

    denom = mean * (1.0 - mean)
    var = min(var, 0.999 * denom)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        a = mean * (denom / var - 1.0)
        b = (1.0 - mean) * (denom / var - 1.0)
        pdf = beta.pdf(grid, a, b)

    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    s = pdf.sum()

    if s > 0.0:
        pdf /= s
    else:
        pdf[np.argmin(np.abs(grid - mean))] = 1.0

    return pdf

# ============================================================
# ------------------ 2D TABLES -------------------------------
# ============================================================

print("Building 2D tables...")

PVmin = np.zeros((NvarZ, NZ))
PVmax = np.zeros((NvarZ, NZ))
Yu2I  = np.zeros((NvarZ, NZ))
Yb2I  = np.zeros((NvarZ, NZ))
YuYbI = np.zeros((NvarZ, NZ))

for iz, Zm in enumerate(tqdm(Z_grid, desc="Z-grid (2D)")):
    for ivz, zetaZ in enumerate(zetaZ_grid):
        PZ = beta_pdf(Zm, zetaZ * Zm * (1 - Zm), Z_grid)
        PVmin[ivz, iz] = np.sum(Yc_u * PZ)
        PVmax[ivz, iz] = np.sum(Yc_b * PZ)
        Yu2I[ivz, iz]  = np.sum(Yc_u**2 * PZ)
        Yb2I[ivz, iz]  = np.sum(Yc_b**2 * PZ)
        YuYbI[ivz, iz] = np.sum(Yc_u * Yc_b * PZ)

mask = PVmax <= PVmin
PVmax[mask] = PVmin[mask] + EPS_PV

# ============================================================
# Z–C INTERPOLATION (HYBRID NaN POLICY)
# ============================================================

print("Interpolating flamelet fields onto Z–C grid...")

points = np.column_stack((Z, C))
Z_mesh, C_mesh = np.meshgrid(Z_grid, C_grid, indexing="ij")

phi_ZC = {}

for name, raw in {
    "T": T, "rho": rho, "mu": mu, "alpha": alpha,
    "Cps": cp, "psi": psi, **Y
}.items():
    lin = griddata(points, raw, (Z_mesh, C_mesh), method="linear", fill_value=np.nan)
    nn  = griddata(points, raw, (Z_mesh, C_mesh), method="nearest")
    phi_ZC[name] = np.where(np.isnan(lin), nn, lin)

lin = griddata(points, SourcePV, (Z_mesh, C_mesh), method="linear", fill_value=np.nan)
phi_ZC["SourcePV"] = np.nan_to_num(lin, nan=0.0)

# ============================================================
# ------------------ 4D TABLES -------------------------------
# ============================================================

tables = {k: np.zeros((NvarC, NC, NvarZ, NZ)) for k in phi_ZC}
tables.update({
    "YuWI": np.zeros((NvarC, NC, NvarZ, NZ)),
    "YbWI": np.zeros((NvarC, NC, NvarZ, NZ)),
    "YWI" : np.zeros((NvarC, NC, NvarZ, NZ)),
})

print("Building 4D tables...")

for iz, Zm in enumerate(tqdm(Z_grid, desc="Z-grid (4D)")):
    for ivz, zetaZ in enumerate(zetaZ_grid):
        PZ = beta_pdf(Zm, zetaZ * Zm * (1 - Zm), Z_grid)
        Yu_bar = np.sum(Yc_u * PZ)
        Yb_bar = np.sum(Yc_b * PZ)

        for ic, Cm in enumerate(C_grid):
            for ivc, zetaC in enumerate(zetaC_grid):
                PC = beta_pdf(Cm, zetaC * Cm * (1 - Cm), C_grid)
                W = PZ[:, None] * PC[None, :]

                tables["YuWI"][ivc, ic, ivz, iz] = Yu_bar
                tables["YbWI"][ivc, ic, ivz, iz] = Yb_bar
                tables["YWI"][ivc, ic, ivz, iz]  = Cm * Yb_bar + (1 - Cm) * Yu_bar

                for name in phi_ZC:
                    tables[name][ivc, ic, ivz, iz] = np.sum(phi_ZC[name] * W)

# ============================================================
# SAVE TABLES
# ============================================================

print("Saving tables...")

np.save(f"{OUT_DIR}/PVmin_table.npy", PVmin)
np.save(f"{OUT_DIR}/PVmax_table.npy", PVmax)
np.save(f"{OUT_DIR}/Yu2I_table.npy",  Yu2I)
np.save(f"{OUT_DIR}/Yb2I_table.npy",  Yb2I)
np.save(f"{OUT_DIR}/YuYbI_table.npy", YuYbI)

for name, arr in tables.items():
    np.save(f"{OUT_DIR}/{name}_table.npy", arr)

print("✔ ALL TABLES GENERATED — NUMERICALLY ROBUST & SOLVER-SAFE")
