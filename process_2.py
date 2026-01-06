import cantera as ct
import numpy as np
import glob
import os

# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def derivative(x, y):
    dydx = np.zeros_like(y)
    dx = np.diff(x)
    dy = np.diff(y)
    dydx[:-1] = dy / dx
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dydx


def computeStrainRates(f, Z):
    strainRates = derivative(f.grid, f.velocity)
    gradZ = derivative(f.grid, Z)
    return strainRates, gradZ


def computeBilger(composition):
    gas = ct.Solution("gri30.yaml")
    gas.X = composition
    return (
        2.0 * gas.elemental_mass_fraction("C") / gas.atomic_weight("C")
        + 0.5 * gas.elemental_mass_fraction("H") / gas.atomic_weight("H")
        - 1.0 * gas.elemental_mass_fraction("O") / gas.atomic_weight("O")
    )

# --------------------------------------------------
# Directories
# --------------------------------------------------

raw_directory = "01_flamelet_results/"
out_directory = "02_processed_flamelet_results/"
os.makedirs(out_directory, exist_ok=True)

# --------------------------------------------------
# Loop over flamelets
# --------------------------------------------------

for name in glob.glob(raw_directory + "*.yaml"):

    print("Processing:", name)

    gas = ct.Solution("gri30.yaml")
    f = ct.CounterflowDiffusionFlame(gas, width=1.0)
    f.restore(name, name="diff1D", loglevel=0)

    fuel = "CH4:1"
    air = "O2:0.21, N2:0.79"

    b_fuel = computeBilger(fuel)
    b_air = computeBilger(air)

    # --------------------------------------------------
    # Mixture fraction Z
    # --------------------------------------------------

    b = np.array([computeBilger(f.X[:, i]) for i in range(f.X.shape[1])])
    Z = (b - b_air) / (b_fuel - b_air)

    # --------------------------------------------------
    # Unscaled progress variable PV = Yc
    # --------------------------------------------------

    idx = gas.species_index
    MW = gas.molecular_weights

    Yc = (
        4.0 * f.Y[idx("H2O")] / MW[idx("H2O")]
        + 2.0 * f.Y[idx("CO2")] / MW[idx("CO2")]
        + 0.5 * f.Y[idx("H2")] / MW[idx("H2")]
        + 1.0 * f.Y[idx("CO")] / MW[idx("CO")]
    )

    # --------------------------------------------------
    # Strain rate and scalar dissipation rate
    # --------------------------------------------------

    strainRates, gradZ = computeStrainRates(f, Z)
    chi = 2.0 * f.velocity * Z * gradZ

    # --------------------------------------------------
    # Thermophysical properties
    # --------------------------------------------------

    alpha = f.thermal_conductivity / f.cp

    gas.basis = "molar"
    rho_molar = f.density
    gas.basis = "mass"
    rho_mass = f.density
    Mbar = rho_mass / rho_molar

    # --------------------------------------------------
    # Assemble data matrix
    # --------------------------------------------------

    data = np.column_stack([
        f.grid,
        f.velocity,
        f.T,
        Z,
        Yc,
        f.Y[idx("CO2")],
        f.Y[idx("H2O")],
        f.Y[idx("CO")],
        f.Y[idx("H2")],
        f.elemental_mass_fraction("C"),
        f.elemental_mass_fraction("H"),
        f.elemental_mass_fraction("O"),
        f.net_production_rates[idx("CO2")] * MW[idx("CO2")],
        f.net_production_rates[idx("H2O")] * MW[idx("H2O")],
        f.net_production_rates[idx("CO")] * MW[idx("CO")],
        f.net_production_rates[idx("H2")] * MW[idx("H2")],
        f.density,
        f.cp,
        alpha,
        f.viscosity,
        Mbar,
        chi,
        f.heat_release_rate,
        f.h,
        f.thermal_conductivity
    ])

    # Append all species mass fractions
    for k in range(gas.n_species):
        data = np.column_stack((data, f.Y[k]))

    # --------------------------------------------------
    # Clean single-line CSV header (NO blank lines)
    # --------------------------------------------------

    header = (
        "z[m],u[m/s],T[K],Z,PV,"
        "Y_CO2,Y_H2O,Y_CO,Y_H2,"
        "Y_C,Y_H,Y_O,"
        "wCO2[kg/m3s],wH2O[kg/m3s],wCO[kg/m3s],wH2[kg/m3s],"
        "rho[kg/m3],cp[J/kgK],alpha[kg/ms],mu[Pa s],Mbar[kg/kmol],"
        "chi[1/s],HRR[W/m3],h[J/kg],lambda[W/mK],"
        + ",".join([f"Y_{sp}" for sp in gas.species_names])
    )

    # --------------------------------------------------
    # Write CSV
    # --------------------------------------------------

    out_file = os.path.join(
        out_directory,
        "post_" + os.path.basename(name).replace(".yaml", ".csv")
    )

    np.savetxt(
        out_file,
        data,
        fmt="%.7e",
        delimiter=",",
        header=header,
        comments=""
    )

    print("Written:", out_file)
