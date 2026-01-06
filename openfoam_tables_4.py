#!/usr/bin/env python3
import numpy as np
import os

# Input and output folders
input_folder = "FGM_tables"        # where .npy files are stored
output_folder = "tablesFoam"       # OpenFOAM formatted tables
os.makedirs(output_folder, exist_ok=True)


def foam_header(table_name):
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield          | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration      | Version:  7                                     |
|   \\  /    A nd            | Web:      http://www.OpenFOAM.com               |
|    \\/     M anipulation   |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{   version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      LDMtable;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

{table_name}
"""


def write_openfoam_table(file_path, table_name, arr):
    with open(file_path, "w") as f:
        f.write(foam_header(table_name))

        # ---------------- 4D TABLE ----------------
        # Expected order: (varC, C, varZ, Z)
        if arr.ndim == 4:
            n_varC, n_C, n_varZ, n_Z = arr.shape

            f.write(f"{n_varC}\n(\n")
            for ivarC in range(n_varC):
                f.write(f"{n_C}\n(\n")
                for iC in range(n_C):
                    f.write(f"{n_varZ}\n(\n")
                    for ivarZ in range(n_varZ):
                        f.write(f"{n_Z}\n(\n")
                        for iZ in range(n_Z):
                            f.write(f"{arr[ivarC, iC, ivarZ, iZ]:.8e}\n")
                        f.write(")\n")
                    f.write(")\n")
                f.write(")\n")
            f.write(")\n;\n")   # <-- semicolon added here

        # ---------------- 2D TABLE ----------------
        # Expected order: (varZ, Z)
        elif arr.ndim == 2:
            n_varZ, n_Z = arr.shape

            f.write(f"{n_varZ}\n(\n")
            for ivarZ in range(n_varZ):
                f.write(f"{n_Z}\n(\n")
                for iZ in range(n_Z):
                    f.write(f"{arr[ivarZ, iZ]:.8e}\n")
                f.write(")\n")
            f.write(")\n;\n")   # <-- semicolon added here

        else:
            raise ValueError(
                f"Unsupported table dimension {arr.ndim} for {table_name}"
            )


# -------------------------------------------------
# Convert all tables
# -------------------------------------------------
for fname in os.listdir(input_folder):
    if fname.endswith(".npy"):
        table_name = fname.replace(".npy", "")
        arr = np.load(os.path.join(input_folder, fname))

        out_path = os.path.join(output_folder, table_name)
        write_openfoam_table(out_path, table_name, arr)

        print(f"✅ Written {arr.ndim}D table → {out_path}")
