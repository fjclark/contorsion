# Contorsion

Convenience tool for running torsion drives and benchmarking force fields using QCEngine.

# Acknowledgements

Lots of code stolen from Josh Horton's examples, for example [here](https://gist.github.com/jthorton/2e556e6080ca616f210a7284dc398de8).

## Installation

Install [pixi](https://pixi.sh/latest/), then run

```bash
pixi install
```

## Features

- Run reference QM torsiondrives for all rotatable bonds in a molecule
- Benchmark force fields against reference data
- Generate energy and RMSD plots
- Modular, type-hinted API with clean separation of concerns

## CLI Usage

### 1. Run Reference Scans

Run reference QM torsiondrives for all rotatable bonds:

```bash
contorsion run-reference-scans \
    "CCO" \
    --method small \
    --program mace \
    --output-dir reference_scans \
    --grid-spacing 15 \
    --ncores 4 \
```

### 2. Run Benchmark Scans

Run force field benchmarks using reference geometries:

```bash
contorsion run-benchmark-scans \
    reference_scans \
    --method openff_unconstrained-2.2.1.offxml \
    --program openmm --basis smirnoff \
    --output-dir benchmark_scans \
    --ncores 1 \
    --memory 1.0
```

### 3. Generate Plots

Create analysis plots comparing reference and benchmark:

```bash
contorsion plot \ 
    reference_scans \
    benchmark_scans \
    --output-dir analysis \
    --reference-label "Mace-Small" \
    --benchmark-label "Sage"
```

## Python API

```python
from pathlib import Path
from qcelemental.models.common_models import Model
from contorsion.drive_torsions import (
    run_reference_torsiondrives,
    run_benchmark_torsiondrives,
    create_analysis_plots,
)

# Run reference scans
model = Model(method="B3LYP-D3BJ", basis="DZVP")
results = run_reference_torsiondrives(
    smiles="CCO",
    model=model,
    output_dir=Path("reference_scans"),
    grid_spacing=15,
    ncores=4,
    memory=8.0,
)

# Run benchmarks
ff_model = Model(method="UFF", basis=None)
benchmark_results = run_benchmark_torsiondrives(
    reference_dir=Path("reference_scans"),
    model=ff_model,
    output_dir=Path("benchmark_scans"),
    program="rdkit",
    ncores=1,
    memory=1.0,
)

# Create plots
create_analysis_plots(
    reference_dir=Path("reference_scans"),
    benchmark_dir=Path("benchmark_scans"),
    output_dir=Path("analysis"),
    reference_label="B3LYP-D3BJ/DZVP",
    benchmark_label="UFF",
)
```

## Output Structure

### Reference Scans
```
reference_scans/
├── molecule_info.json      # Molecule metadata and dihedral definitions
├── molecule.png            # 2D structure with rotatable bonds highlighted
├── bond_0_reference.json   # TorsionDriveResult for first rotatable bond
├── bond_1_reference.json   # TorsionDriveResult for second rotatable bond
└── ...
```

### Benchmark Scans
```
benchmark_scans/
├── bond_0_benchmark.json   # Benchmark result for first bond
├── bond_1_benchmark.json   # Benchmark result for second bond
└── ...
```

### Analysis Output
```
analysis/
├── molecule.png           # Copy of molecule structure
├── summary.json          # Summary metrics (RMSE, max RMSD) for all bonds
├── bond_0/
│   ├── energy.pdf        # Energy profile comparison
│   ├── energy.png
│   ├── rmsd.pdf          # RMSD profile
│   └── rmsd.png
├── bond_1/
│   └── ...
└── ...
```

## Core Functions

### Utility Functions

- `find_rotatable_bonds_with_dihedrals(molecule)`: Find all rotatable bonds and return dihedral indices
- `build_torsiondrive_input(...)`: Build TorsionDriveInput for a specific dihedral
- `run_single_torsiondrive(...)`: Execute a single torsiondrive calculation
- `run_constrained_optimization(...)`: Run constrained geometry optimization
- `calculate_energy_rmsd(...)`: Calculate relative energies and RMSDs between scans
- `calculate_rmse(...)`: Calculate root mean square error

### Plotting Functions

- `plot_energy_profile(...)`: Create energy profile plot
- `plot_rmsd_profile(...)`: Create RMSD profile plot
- `draw_molecule_with_bonds(...)`: Draw 2D molecule with highlighted bonds

### High-Level Workflow Functions

- `run_reference_torsiondrives(...)`: Run reference scans for all rotatable bonds
- `run_benchmark_torsiondrives(...)`: Run benchmark scans using reference geometries
- `create_analysis_plots(...)`: Create all analysis plots and summary
