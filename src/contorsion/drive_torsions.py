"""Functionality for driving torsions with QCEngine and analyzing results."""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging

import numpy as np
from numpy.typing import NDArray
import typer
from rich.console import Console
from rich.progress import track
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from qcelemental.models import DriverEnum, FailedOperation
from qcelemental.models import Molecule as QCMolecule
from qcelemental.models.common_models import Model
from qcelemental.models.procedures import (
    OptimizationInput,
    OptimizationSpecification,
    QCInputSpecification,
    TDKeywords,
    TorsionDriveInput,
    TorsionDriveResult,
)
from qcengine.units import ureg
import qcengine

from openff.toolkit.topology import Molecule, Atom, Bond
from openff.units import unit
from rdkit.Chem import Draw

# Set up logging
logging.disable(level=logging.CRITICAL)

# Create CLI app and console
app = typer.Typer(help="Drive torsions and benchmark force fields")
console = Console()


# ============================================================================
# Core utility functions
# ============================================================================


def find_rotatable_bonds_with_dihedrals(
    molecule: Molecule,
) -> List[Tuple[int, int, int, int]]:
    """
    Find all rotatable bonds and return dihedral atom indices for each.

    Parameters
    ----------
    molecule : Molecule
        OpenFF Molecule to analyze

    Returns
    -------
    List[Tuple[int, int, int, int]]
        List of dihedral indices (atom1, atom2, atom3, atom4) for each
        rotatable bond
    """
    dihedrals = []
    rotatable_bonds = molecule.find_rotatable_bonds()

    for bond in rotatable_bonds:
        atom2: Atom = bond.atom1
        atom3: Atom = bond.atom2

        # Find atoms bonded to atom2 (excluding atom3)
        atom2_neighbors = [
            atom.molecule_particle_index for atom in atom2.bonded_atoms if atom != atom3
        ]
        # Find atoms bonded to atom3 (excluding atom2)
        atom3_neighbors = [
            atom.molecule_particle_index for atom in atom3.bonded_atoms if atom != atom2
        ]

        if atom2_neighbors and atom3_neighbors:
            atom1 = atom2_neighbors[0]
            atom4 = atom3_neighbors[0]
            dihedrals.append(
                (
                    atom1,
                    atom2.molecule_particle_index,
                    atom3.molecule_particle_index,
                    atom4,
                )
            )

    return dihedrals


def build_torsiondrive_input(
    molecule: Molecule,
    dihedral: Tuple[int, int, int, int],
    model: Model,
    program: str,
    grid_spacing: int = 15,
    optimization_keywords: Optional[Dict[str, Any]] = None,
) -> TorsionDriveInput:
    """
    Build a TorsionDriveInput for a specific dihedral.

    Parameters
    ----------
    molecule : Molecule
        OpenFF Molecule with at least one conformer
    dihedral : Tuple[int, int, int, int]
        Dihedral atom indices (atom1, atom2, atom3, atom4)
    model : Model
        QCElemental Model specification
    program : str
        QCEngine program name
    grid_spacing : int, default=15
        Grid spacing in degrees for torsion scan
    optimization_keywords : Optional[Dict[str, Any]], default=None
        Keywords for geometric optimizer. If None, uses sensible defaults

    Returns
    -------
    TorsionDriveInput
        Input specification for torsiondrive
    """
    if optimization_keywords is None:
        optimization_keywords = {
            "coordsys": "dlc",
            "maxiter": 600,
            "enforce": 0.1,
            "reset": True,
            "qccnv": True,
            "epsilon": 0.0,
            "program": program,
        }

    keywords = TDKeywords(dihedrals=[dihedral], grid_spacing=[grid_spacing])
    input_spec = QCInputSpecification(
        driver=DriverEnum.gradient,
        model=model,
    )
    optimization_spec = OptimizationSpecification(
        procedure="geometric",
        keywords=optimization_keywords,
    )

    return TorsionDriveInput(
        keywords=keywords,
        input_specification=input_spec,
        initial_molecule=[molecule.to_qcschema()],
        optimization_spec=optimization_spec,
    )


def run_single_torsiondrive(
    td_input: TorsionDriveInput,
    ncores: int = 1,
    memory: float = 4.0,
) -> TorsionDriveResult:
    """
    Execute a single torsiondrive calculation.

    Parameters
    ----------
    td_input : TorsionDriveInput
        Input specification for the torsiondrive
    ncores : int, default=1
        Number of CPU cores to use
    memory : float, default=4.0
        Memory in GB

    Returns
    -------
    TorsionDriveResult
        Result from the torsiondrive calculation

    Raises
    ------
    RuntimeError
        If the torsiondrive calculation fails
    """
    result = qcengine.compute_procedure(
        input_data=td_input,
        procedure="torsiondrive",
        raise_error=True,
        local_options={"ncores": ncores, "memory": memory},
    )
    return result


def run_constrained_optimization(
    molecule: QCMolecule,
    dihedral: Tuple[int, int, int, int],
    model: Model,
    optimization_keywords: Dict[str, Any],
    ncores: int = 1,
    memory: float = 1.0,
) -> Tuple[QCMolecule, float, bool]:
    """
    Run a constrained geometry optimization with frozen dihedral.

    Parameters
    ----------
    molecule : QCMolecule
        Starting geometry
    dihedral : Tuple[int, int, int, int]
        Dihedral indices to freeze
    model : Model
        Force field or QM method
    optimization_keywords : Dict[str, Any]
        Geometric optimization settings
    ncores : int, default=1
        Number of cores
    memory : float, default=1.0
        Memory in GB

    Returns
    -------
    Tuple[QCMolecule, float, bool]
        Final molecule, final energy (Hartree), and success flag
    """
    qc_spec = QCInputSpecification(
        driver="gradient",
        model=model,
    )

    opt_keywords = optimization_keywords.copy()
    opt_keywords["constraints"] = {
        "freeze": [{"type": "dihedral", "indices": list(dihedral)}]
    }

    opt_task = OptimizationInput(
        keywords=opt_keywords,
        input_specification=qc_spec,
        initial_molecule=molecule,
    )

    result = qcengine.compute_procedure(
        input_data=opt_task,
        procedure="geometric",
        task_config={"ncores": ncores, "memory": memory},
    )

    if isinstance(result, FailedOperation):
        # Try to extract last geometry if available
        if hasattr(result, "input_data") and "final_molecule" in result.input_data:
            final_mol = QCMolecule.from_data(result.input_data["final_molecule"])
            energy = result.input_data["energies"][-1]
            return final_mol, energy, False
        else:
            raise RuntimeError(f"Optimization failed: {result.error}")
    else:
        return result.final_molecule, result.energies[-1], True


def calculate_energy_rmsd(
    reference: TorsionDriveResult,
    target: TorsionDriveResult,
    angle_range: range = range(-165, 180, 15),
) -> Dict[str, NDArray[np.floating]]:
    """
    Calculate relative energies and RMSDs between reference and target scans.

    Parameters
    ----------
    reference : TorsionDriveResult
        Reference (e.g., QM) scan
    target : TorsionDriveResult
        Target (e.g., force field) scan
    angle_range : range, default=range(-165, 180, 15)
        Angles to analyze

    Returns
    -------
    Dict[str, NDArray[np.floating]]
        Dictionary with keys:
        - "angles": angles in degrees
        - "ref_energies": reference energies in kJ/mol
        - "target_energies": target energies in kJ/mol
        - "rmsds": RMSD values in Angstrom
    """
    angles = []
    ref_energies_list = []
    target_energies_list = []
    rmsds = []

    for angle in angle_range:
        angle_key = str(angle)
        if angle_key not in reference.final_energies:
            continue
        if angle_key not in target.final_energies:
            continue

        angles.append(angle)
        ref_energies_list.append(reference.final_energies[angle_key])
        target_energies_list.append(target.final_energies[angle_key])

        ref_mol = reference.final_molecules[angle_key]
        target_mol = target.final_molecules[angle_key]
        _, align_data = target_mol.align(ref_mol=ref_mol, atoms_map=True)
        rmsds.append(align_data["rmsd"])

    # Convert to numpy arrays and normalize energies
    ref_energies = np.array(ref_energies_list)
    ref_energies -= ref_energies.min()
    ref_energies *= ureg.conversion_factor("hartree", "kilojoule/mol")

    target_energies = np.array(target_energies_list)
    target_energies -= target_energies.min()
    target_energies *= ureg.conversion_factor("hartree", "kilojoule/mol")

    return {
        "angles": np.array(angles),
        "ref_energies": ref_energies,
        "target_energies": target_energies,
        "rmsds": np.array(rmsds),
    }


def calculate_rmse(
    reference: NDArray[np.floating],
    target: NDArray[np.floating],
) -> float:
    """
    Calculate root mean square error.

    Parameters
    ----------
    reference : NDArray[np.floating]
        Reference values
    target : NDArray[np.floating]
        Target values

    Returns
    -------
    float
        RMSE
    """
    return float(np.sqrt(np.mean((reference - target) ** 2)))


# ============================================================================
# Plotting functions
# ============================================================================


def plot_energy_profile(
    data: Dict[str, NDArray[np.floating]],
    reference_label: str = "Reference",
    target_label: str = "Target",
) -> Figure:
    """
    Create energy profile plot.

    Parameters
    ----------
    data : Dict[str, NDArray[np.floating]]
        Data from calculate_energy_rmsd
    reference_label : str, default="Reference"
        Label for reference data
    target_label : str, default="Target"
        Label for target data

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        data["angles"],
        data["ref_energies"],
        marker="o",
        label=reference_label,
        linewidth=2,
    )
    ax.plot(
        data["angles"],
        data["target_energies"],
        marker="x",
        label=target_label,
        linewidth=2,
    )
    ax.set_xlabel("Dihedral Angle (°)", fontsize=12)
    ax.set_ylabel("Relative Energy (kJ/mol)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_rmsd_profile(
    data: Dict[str, NDArray[np.floating]],
    target_label: str = "Target",
) -> Figure:
    """
    Create RMSD profile plot.

    Parameters
    ----------
    data : Dict[str, NDArray[np.floating]]
        Data from calculate_energy_rmsd
    target_label : str, default="Target"
        Label for the data

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        data["angles"],
        data["rmsds"],
        marker="o",
        label=target_label,
        linewidth=2,
    )
    ax.set_xlabel("Dihedral Angle (°)", fontsize=12)
    ax.set_ylabel("RMSD (Å)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def draw_molecule_with_bonds(
    molecule: Molecule,
    dihedrals: List[Tuple[int, int, int, int]],
    output_path: Path,
) -> None:
    """
    Draw 2D molecule structure with rotatable bonds highlighted.

    Parameters
    ----------
    molecule : Molecule
        OpenFF Molecule
    dihedrals : List[Tuple[int, int, int, int]]
        List of dihedral atom indices to highlight
    output_path : Path
        Path to save the image
    """
    rdkit_mol = molecule.to_rdkit()
    Draw.rdDepictor.Compute2DCoords(rdkit_mol)

    # Flatten dihedrals to get all highlighted atoms
    highlight_atoms = list(set(atom for dihedral in dihedrals for atom in dihedral))

    Draw.MolToFile(
        rdkit_mol,
        str(output_path),
        highlightAtoms=highlight_atoms,
        size=(800, 800),
    )


# ============================================================================
# High-level workflow functions
# ============================================================================


def run_reference_torsiondrives(
    smiles: str,
    model: Model,
    output_dir: Path,
    program: str = "openmm",
    grid_spacing: int = 15,
    n_conformers: int = 1,
    ncores: int = 1,
    memory: float = 4.0,
) -> Dict[int, TorsionDriveResult]:
    """
    Run reference torsiondrives for all rotatable bonds in a molecule.

    Parameters
    ----------
    smiles : str
        SMILES string of molecule
    model : Model
        QCElemental Model (e.g., QM method)
    program : str, default="openmm"
        QCEngine program name
    output_dir : Path
        Directory to save results
    grid_spacing : int, default=15
        Grid spacing in degrees
    n_conformers : int, default=1
        Number of conformers to generate
    ncores : int, default=1
        Number of CPU cores
    memory : float, default=4.0
        Memory in GB

    Returns
    -------
    Dict[int, TorsionDriveResult]
        Results indexed by bond number
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build molecule
    console.print(f"[bold blue]Building molecule from SMILES:[/bold blue] {smiles}")
    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=n_conformers)

    # Find rotatable bonds
    dihedrals = find_rotatable_bonds_with_dihedrals(molecule)
    console.print(f"[green]Found {len(dihedrals)} rotatable bonds[/green]")

    if not dihedrals:
        console.print("[yellow]Warning: No rotatable bonds found[/yellow]")
        return {}

    # Save molecule info
    mol_data = {
        "smiles": smiles,
        "n_rotatable_bonds": len(dihedrals),
        "dihedrals": dihedrals,
    }
    with open(output_dir / "molecule_info.json", "w") as f:
        json.dump(mol_data, f, indent=2)

    # Draw molecule
    draw_molecule_with_bonds(
        molecule,
        dihedrals,
        output_dir / "molecule.png",
    )

    # Run torsiondrives
    results = {}
    for i, dihedral in enumerate(dihedrals):
        console.print(
            f"[cyan]Running torsiondrive for bond {i + 1}/{len(dihedrals)}: "
            f"{dihedral}[/cyan]"
        )

        td_input = build_torsiondrive_input(
            molecule,
            dihedral,
            model,
            program,
            grid_spacing=grid_spacing,
        )

        result = run_single_torsiondrive(td_input, ncores=ncores, memory=memory)

        # Save result
        result_file = output_dir / f"bond_{i}_reference.json"
        with open(result_file, "w") as f:
            f.write(result.json())

        results[i] = result
        console.print(f"[green]✓ Bond {i + 1} complete[/green]")

    return results


def run_benchmark_torsiondrives(
    reference_dir: Path,
    model: Model,
    output_dir: Path,
    program: str = "rdkit",
    ncores: int = 1,
    memory: float = 1.0,
) -> Dict[int, TorsionDriveResult]:
    """
    Run benchmark torsiondrives using reference geometries.

    Parameters
    ----------
    reference_dir : Path
        Directory containing reference torsiondrive results
    model : Model
        Force field model to benchmark
    output_dir : Path
        Directory to save benchmark results
    program : str, default="rdkit"
        QCEngine program name
    ncores : int, default=1
        Number of CPU cores
    memory : float, default=1.0
        Memory in GB

    Returns
    -------
    Dict[int, TorsionDriveResult]
        Benchmark results indexed by bond number
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load molecule info
    with open(reference_dir / "molecule_info.json") as f:
        mol_info = json.load(f)

    dihedrals = [tuple(d) for d in mol_info["dihedrals"]]
    console.print(
        f"[bold blue]Running benchmark for {len(dihedrals)} bonds[/bold blue]"
    )

    results = {}
    for i in range(len(dihedrals)):
        reference_file = reference_dir / f"bond_{i}_reference.json"
        if not reference_file.exists():
            console.print(
                f"[yellow]Warning: Reference file for bond {i} not found[/yellow]"
            )
            continue

        console.print(f"[cyan]Processing bond {i + 1}/{len(dihedrals)}[/cyan]")

        # Load reference
        reference = TorsionDriveResult.parse_file(str(reference_file))
        dihedral = dihedrals[i]

        # Build benchmark result
        benchmark_result = TorsionDriveResult(
            keywords=reference.keywords,
            extras=reference.extras,
            input_specification=QCInputSpecification(
                driver="gradient",
                model=model,
            ),
            initial_molecule=list(reference.final_molecules.values()),
            optimization_spec=reference.optimization_spec,
            final_energies={},
            final_molecules={},
            optimization_history={},
            provenance={
                "creator": "geometric_custom",
                "routine": "custom",
                "version": 1,
            },
            success=True,
        )

        #     target_result = TorsionDriveResult(
        #     keywords=qm_input_scan.keywords,
        #     extras=qm_input_scan.extras,
        #     input_specification=qc_spec,
        #     initial_molecule=list(qm_inputs.values()),
        #     optimization_spec=qm_input_scan.optimization_spec,
        #     final_energies={},
        #     final_molecules={},
        #     optimization_history={},
        #     provenance={"creator": "geometric_custom", "routine": "custom", "version": 1},
        #     success=True,
        # )

        # Get optimization keywords
        opt_keywords = reference.optimization_history["0"][0].keywords.copy()
        opt_keywords["maxiter"] = 100
        opt_keywords["program"] = program

        # Run constrained optimizations for each angle
        failed_count = 0
        for angle_key, ref_molecule in track(
            reference.final_molecules.items(),
            description=f"Bond {i + 1}",
        ):
            final_mol, energy, success = run_constrained_optimization(
                ref_molecule,
                dihedral,
                model,
                opt_keywords,
                ncores=ncores,
                memory=memory,
            )
            benchmark_result.final_molecules[angle_key] = final_mol
            benchmark_result.final_energies[angle_key] = energy
            try:
                final_mol, energy, success = run_constrained_optimization(
                    ref_molecule,
                    dihedral,
                    model,
                    opt_keywords,
                    ncores=ncores,
                    memory=memory,
                )
                benchmark_result.final_molecules[angle_key] = final_mol
                benchmark_result.final_energies[angle_key] = energy
                if not success:
                    failed_count += 1
            except Exception as e:
                console.print(f"[red]Error at angle {angle_key}: {e}[/red]")
                failed_count += 1
                continue

        if failed_count > 0:
            console.print(
                f"[yellow]Warning: {failed_count} optimizations "
                f"failed for bond {i}[/yellow]"
            )

        # Save result
        result_file = output_dir / f"bond_{i}_benchmark.json"
        with open(result_file, "w") as f:
            f.write(benchmark_result.json())

        results[i] = benchmark_result
        console.print(f"[green]✓ Bond {i + 1} complete[/green]")

    return results


def create_analysis_plots(
    reference_dir: Path,
    benchmark_dir: Path,
    output_dir: Path,
    reference_label: str = "Reference",
    benchmark_label: str = "Benchmark",
) -> None:
    """
    Create analysis plots comparing reference and benchmark scans.

    Parameters
    ----------
    reference_dir : Path
        Directory with reference results
    benchmark_dir : Path
        Directory with benchmark results
    output_dir : Path
        Directory to save plots
    reference_label : str, default="Reference"
        Label for reference data
    benchmark_label : str, default="Benchmark"
        Label for benchmark data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load molecule info
    with open(reference_dir / "molecule_info.json") as f:
        mol_info = json.load(f)

    n_bonds = mol_info["n_rotatable_bonds"]
    console.print(f"[bold blue]Creating plots for {n_bonds} bonds[/bold blue]")

    # Summary data
    summary = []

    for i in range(n_bonds):
        ref_file = reference_dir / f"bond_{i}_reference.json"
        bench_file = benchmark_dir / f"bond_{i}_benchmark.json"

        if not ref_file.exists() or not bench_file.exists():
            console.print(f"[yellow]Skipping bond {i}: missing data[/yellow]")
            continue

        # Load results
        reference = TorsionDriveResult.parse_file(str(ref_file))
        benchmark = TorsionDriveResult.parse_file(str(bench_file))

        # Calculate data
        data = calculate_energy_rmsd(reference, benchmark)

        # Create plots
        bond_dir = output_dir / f"bond_{i}"
        bond_dir.mkdir(exist_ok=True)

        # Energy plot
        fig_energy = plot_energy_profile(
            data,
            reference_label=reference_label,
            target_label=benchmark_label,
        )
        fig_energy.savefig(bond_dir / "energy.pdf")
        fig_energy.savefig(bond_dir / "energy.png", dpi=300)
        plt.close(fig_energy)

        # RMSD plot
        fig_rmsd = plot_rmsd_profile(data, target_label=benchmark_label)
        fig_rmsd.savefig(bond_dir / "rmsd.pdf")
        fig_rmsd.savefig(bond_dir / "rmsd.png", dpi=300)
        plt.close(fig_rmsd)

        # Calculate metrics
        rmse = calculate_rmse(data["ref_energies"], data["target_energies"])
        max_rmsd = float(np.max(data["rmsds"]))

        summary.append(
            {
                "bond": i,
                "dihedral": mol_info["dihedrals"][i],
                "energy_rmse_kj_mol": round(rmse, 3),
                "max_rmsd_angstrom": round(max_rmsd, 3),
            }
        )

        console.print(
            f"[green]✓ Bond {i}: RMSE={rmse:.2f} kJ/mol, "
            f"Max RMSD={max_rmsd:.3f} Å[/green]"
        )

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Copy molecule image
    import shutil

    mol_img = reference_dir / "molecule.png"
    if mol_img.exists():
        shutil.copy(mol_img, output_dir / "molecule.png")

    console.print(
        f"[bold green]Analysis complete! Results in {output_dir}[/bold green]"
    )


# ============================================================================
# CLI Commands
# ============================================================================


@app.command()
def run_reference_scans(
    smiles: str = typer.Argument(..., help="SMILES string of molecule"),
    method: str = typer.Option(
        "EGRET_1.model",
        help="QM method for reference calculations",
    ),
    program: str = typer.Option(
        "mace",
        help="QCEngine program to use",
    ),
    basis: Optional[str] = typer.Option(
        None,
        help="Basis set (if applicable)",
    ),
    output_dir: Path = typer.Option(
        Path("reference_scans"),
        help="Output directory for reference scans",
    ),
    grid_spacing: int = typer.Option(
        15,
        help="Grid spacing in degrees",
    ),
    ncores: int = typer.Option(
        1,
        help="Number of CPU cores",
    ),
    memory: float = typer.Option(
        4.0,
        help="Memory in GB",
    ),
) -> None:
    """
    Run reference QM torsiondrives for all rotatable bonds in a molecule.
    """
    model = Model(method=method, basis=basis)

    run_reference_torsiondrives(
        smiles=smiles,
        model=model,
        program=program,
        output_dir=output_dir,
        grid_spacing=grid_spacing,
        ncores=ncores,
        memory=memory,
    )


@app.command()
def run_benchmark_scans(
    reference_dir: Path = typer.Argument(
        ...,
        help="Directory containing reference scan results",
    ),
    method: str = typer.Option(
        "UFF",
        help="Force field method name",
    ),
    program: str = typer.Option(
        "mace",
        help="QCEngine program to use",
    ),
    basis: Optional[str] = typer.Option(
        None,
        help="Basis set (if applicable)",
    ),
    output_dir: Path = typer.Option(
        Path("benchmark_scans"),
        help="Output directory for benchmark scans",
    ),
    ncores: int = typer.Option(
        1,
        help="Number of CPU cores",
    ),
    memory: float = typer.Option(
        1.0,
        help="Memory in GB",
    ),
) -> None:
    """
    Run benchmark force field torsiondrives using reference geometries.
    """
    model = Model(method=method, basis=basis)

    run_benchmark_torsiondrives(
        reference_dir=reference_dir,
        model=model,
        output_dir=output_dir,
        program=program,
        ncores=ncores,
        memory=memory,
    )


@app.command()
def plot(
    reference_dir: Path = typer.Argument(
        ...,
        help="Directory containing reference scan results",
    ),
    benchmark_dir: Path = typer.Argument(
        ...,
        help="Directory containing benchmark scan results",
    ),
    output_dir: Path = typer.Option(
        Path("analysis"),
        help="Output directory for plots and analysis",
    ),
    reference_label: str = typer.Option(
        "QM Reference",
        help="Label for reference data in plots",
    ),
    benchmark_label: str = typer.Option(
        "Force Field",
        help="Label for benchmark data in plots",
    ),
) -> None:
    """
    Create energy and RMSD plots comparing reference and benchmark scans.
    """
    create_analysis_plots(
        reference_dir=reference_dir,
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        reference_label=reference_label,
        benchmark_label=benchmark_label,
    )


def cli():
    app()


if __name__ == "__main__":
    cli()
