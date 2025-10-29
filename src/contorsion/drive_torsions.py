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
            "maxiter": 300,
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
    ncores: int = 30,
    memory: float = 20.0,
) -> TorsionDriveResult:
    """
    Execute a single torsiondrive calculation.

    Parameters
    ----------
    td_input : TorsionDriveInput
        Input specification for the torsiondrive
    ncores : int, default=30
        Number of CPU cores to use
    memory : float, default=20.0
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
    ncores: int = 30,
    memory: float = 20.0,
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
    ncores : int, default=30
        Number of cores
    memory : float, default=20.0
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
    ax.grid(True)
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
    ax.grid(True)
    plt.tight_layout()
    return fig


def draw_molecule_with_bonds(
    molecule: Molecule,
    dihedrals: List[Tuple[int, int, int, int]],
    output_path: Path,
    highlight_specific: Optional[int] = None,
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
    highlight_specific : Optional[int], default=None
        If provided, only highlight the dihedral at this index.
        Otherwise highlight all dihedrals.
    """
    rdkit_mol = molecule.to_rdkit()
    Draw.rdDepictor.Compute2DCoords(rdkit_mol)

    # Determine which atoms to highlight
    if highlight_specific is not None and 0 <= highlight_specific < len(dihedrals):
        # Highlight only the specific dihedral
        highlight_atoms = list(dihedrals[highlight_specific])
    else:
        # Flatten all dihedrals to get all highlighted atoms
        highlight_atoms = list(set(atom for dihedral in dihedrals for atom in dihedral))

    # Create drawing options to include atom indices
    drawer = Draw.MolDraw2DCairo(800, 800)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(rdkit_mol, highlightAtoms=highlight_atoms)
    drawer.FinishDrawing()
    drawer.WriteDrawingText(str(output_path))


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
    ncores: int = 30,
    memory: float = 20.0,
    bond_indices: Optional[List[int]] = None,
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
    ncores : int, default=30
        Number of CPU cores
    memory : float, default=20.0
        Memory in GB
    bond_indices : Optional[List[int]], default=None
        Specific bond indices to scan. If None, scan all rotatable bonds.
        Indices correspond to the order of rotatable bonds found.

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
    all_dihedrals = find_rotatable_bonds_with_dihedrals(molecule)
    console.print(f"[green]Found {len(all_dihedrals)} rotatable bonds[/green]")

    if not all_dihedrals:
        console.print("[yellow]Warning: No rotatable bonds found[/yellow]")
        return {}

    # Filter bonds if specific indices requested
    if bond_indices is not None:
        dihedrals = [
            all_dihedrals[i] for i in bond_indices if 0 <= i < len(all_dihedrals)
        ]
        console.print(
            f"[cyan]Scanning {len(dihedrals)} specific bond(s): {bond_indices}[/cyan]"
        )
        indices_to_scan = bond_indices
    else:
        dihedrals = all_dihedrals
        indices_to_scan = list(range(len(dihedrals)))

    # Save molecule info
    mol_data = {
        "smiles": smiles,
        "n_rotatable_bonds": len(all_dihedrals),
        "dihedrals": all_dihedrals,
        "scanned_indices": indices_to_scan,
    }
    with open(output_dir / "molecule_info.json", "w") as f:
        json.dump(mol_data, f, indent=2)

    # Draw molecule with all bonds
    draw_molecule_with_bonds(
        molecule,
        all_dihedrals,
        output_dir / "molecule.png",
    )

    # Run torsiondrives
    results = {}
    for idx, (bond_idx, dihedral) in enumerate(zip(indices_to_scan, dihedrals)):
        console.print(
            f"[cyan]Running torsiondrive for bond {idx + 1}/{len(dihedrals)} "
            f"(index {bond_idx}): {dihedral}[/cyan]"
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
        result_file = output_dir / f"bond_{bond_idx}_reference.json"
        with open(result_file, "w") as f:
            f.write(result.json())

        results[bond_idx] = result
        console.print(f"[green]✓ Bond {bond_idx} complete[/green]")

    return results


def run_benchmark_torsiondrives(
    reference_dir: Path,
    model: Model,
    output_dir: Path,
    program: str = "rdkit",
    ncores: int = 30,
    memory: float = 20.0,
    bond_indices: Optional[List[int]] = None,
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
    ncores : int, default=30
        Number of CPU cores
    memory : float, default=20.0
        Memory in GB
    bond_indices : Optional[List[int]], default=None
        Specific bond indices to benchmark. If None, automatically detect
        which bonds are present in the reference directory.

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

    # Auto-detect which bonds have reference data if not specified
    if bond_indices is None:
        bond_indices = []
        for i in range(len(dihedrals)):
            if (reference_dir / f"bond_{i}_reference.json").exists():
                bond_indices.append(i)
        console.print(
            f"[cyan]Auto-detected {len(bond_indices)} bonds with reference data: {bond_indices}[/cyan]"
        )

    console.print(
        f"[bold blue]Running benchmark for {len(bond_indices)} bond(s)[/bold blue]"
    )

    results = {}
    for i in bond_indices:
        reference_file = reference_dir / f"bond_{i}_reference.json"
        if not reference_file.exists():
            console.print(
                f"[yellow]Warning: Reference file for bond {i} not found[/yellow]"
            )
            continue

        console.print(f"[cyan]Processing bond {i}[/cyan]")

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

        # Get optimization keywords
        opt_keywords = reference.optimization_history["0"][0].keywords.copy()
        opt_keywords["maxiter"] = 100
        opt_keywords["program"] = program

        # Run constrained optimizations for each angle
        failed_count = 0
        for angle_key, ref_molecule in track(
            reference.final_molecules.items(),
            description=f"Bond {i}",
        ):
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
        console.print(f"[green]✓ Bond {i} complete[/green]")

    return results


def create_analysis_plots(
    reference_dir: Path,
    benchmark_dirs: List[Path],
    output_dir: Path,
    reference_label: str = "Reference",
    benchmark_labels: Optional[List[str]] = None,
) -> None:
    """
    Create analysis plots comparing reference and benchmark scans.

    Parameters
    ----------
    reference_dir : Path
        Directory with reference results
    benchmark_dirs : List[Path]
        List of directories with benchmark results to compare
    output_dir : Path
        Directory to save plots
    reference_label : str, default="Reference"
        Label for reference data
    benchmark_labels : Optional[List[str]], default=None
        Labels for benchmark data. If None, uses directory names.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load molecule info
    with open(reference_dir / "molecule_info.json") as f:
        mol_info = json.load(f)

    n_bonds = mol_info["n_rotatable_bonds"]
    all_dihedrals = [tuple(d) for d in mol_info["dihedrals"]]
    smiles = mol_info["smiles"]

    # Set default labels if not provided
    if benchmark_labels is None:
        benchmark_labels = [d.name for d in benchmark_dirs]

    console.print(
        f"[bold blue]Creating plots, comparing {len(benchmark_dirs)} benchmark(s)[/bold blue]"
    )

    # Load molecule for bond-specific highlighting
    molecule = Molecule.from_smiles(smiles)

    # Summary data for each benchmark
    summaries = {label: [] for label in benchmark_labels}

    for i in range(n_bonds):
        ref_file = reference_dir / f"bond_{i}_reference.json"
        if not ref_file.exists():
            continue

        # Check which benchmarks have this bond
        available_benchmarks = []
        for bench_dir, label in zip(benchmark_dirs, benchmark_labels):
            bench_file = bench_dir / f"bond_{i}_benchmark.json"
            if bench_file.exists():
                available_benchmarks.append((bench_dir, label, bench_file))

        if not available_benchmarks:
            console.print(f"[yellow]Skipping bond {i}: no benchmark data[/yellow]")
            continue

        # Load reference
        reference = TorsionDriveResult.parse_file(str(ref_file))

        # Create plots
        bond_dir = output_dir / f"bond_{i}"
        bond_dir.mkdir(exist_ok=True)

        # Draw molecule with this specific bond highlighted
        draw_molecule_with_bonds(
            molecule,
            all_dihedrals,
            bond_dir / "molecule_bond_highlighted.png",
            highlight_specific=i,
        )

        # Create multi-benchmark energy and RMSD plots
        fig_energy, ax_energy = plt.subplots(figsize=(8, 6))
        fig_rmsd, ax_rmsd = plt.subplots(figsize=(8, 6))

        # Plot reference data once
        first_benchmark = TorsionDriveResult.parse_file(str(available_benchmarks[0][2]))
        first_data = calculate_energy_rmsd(reference, first_benchmark)
        ax_energy.plot(
            first_data["angles"],
            first_data["ref_energies"],
            marker="o",
            label=reference_label,
            linewidth=2,
        )

        # Plot each benchmark
        for bench_dir, label, bench_file in available_benchmarks:
            benchmark = TorsionDriveResult.parse_file(str(bench_file))
            data = calculate_energy_rmsd(reference, benchmark)

            # Add to energy plot
            ax_energy.plot(
                data["angles"],
                data["target_energies"],
                marker="x",
                label=label,
                linewidth=2,
            )

            # Add to RMSD plot
            ax_rmsd.plot(
                data["angles"],
                data["rmsds"],
                marker="o",
                label=label,
                linewidth=2,
            )

            # Calculate metrics
            rmse = calculate_rmse(data["ref_energies"], data["target_energies"])
            max_rmsd = float(np.max(data["rmsds"]))

            summaries[label].append(
                {
                    "bond": i,
                    "dihedral": mol_info["dihedrals"][i],
                    "energy_rmse_kj_mol": round(rmse, 3),
                    "max_rmsd_angstrom": round(max_rmsd, 3),
                }
            )

            console.print(
                f"[green]✓ Bond {i} ({label}): RMSE={rmse:.2f} kJ/mol, "
                f"Max RMSD={max_rmsd:.3f} Å[/green]"
            )

        # Finalize energy plot
        ax_energy.set_xlabel("Dihedral Angle (°)", fontsize=12)
        ax_energy.set_ylabel("Relative Energy (kJ/mol)", fontsize=12)
        ax_energy.legend(fontsize=10)
        ax_energy.grid(True)
        plt.tight_layout()
        fig_energy.savefig(bond_dir / "energy.pdf")
        fig_energy.savefig(bond_dir / "energy.png", dpi=300)
        plt.close(fig_energy)

        # Finalize RMSD plot
        ax_rmsd.set_xlabel("Dihedral Angle (°)", fontsize=12)
        ax_rmsd.set_ylabel("RMSD (Å)", fontsize=12)
        ax_rmsd.legend(fontsize=10)
        ax_rmsd.grid(True)
        plt.tight_layout()
        fig_rmsd.savefig(bond_dir / "rmsd.pdf")
        fig_rmsd.savefig(bond_dir / "rmsd.png", dpi=300)
        plt.close(fig_rmsd)

    # Save summaries
    for label, summary in summaries.items():
        safe_label = label.replace(" ", "_").replace("/", "_")
        summary_file = output_dir / f"summary_{safe_label}.json"
        with open(summary_file, "w") as f:
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
        "small",
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
    bond_indices: Optional[str] = typer.Option(
        None,
        help="Comma-separated bond indices to scan (e.g., '0,2,5'). If not provided, scan all rotatable bonds.",
    ),
    ncores: int = typer.Option(
        30,
        help="Number of CPU cores",
    ),
    memory: float = typer.Option(
        20.0,
        help="Memory in GB",
    ),
) -> None:
    """
    Run reference QM torsiondrives for all rotatable bonds in a molecule.
    """
    model = Model(method=method, basis=basis)

    # Parse bond indices if provided
    parsed_indices = None
    if bond_indices:
        parsed_indices = [int(x.strip()) for x in bond_indices.split(",")]

    run_reference_torsiondrives(
        smiles=smiles,
        model=model,
        program=program,
        output_dir=output_dir,
        grid_spacing=grid_spacing,
        ncores=ncores,
        memory=memory,
        bond_indices=parsed_indices,
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
    bond_indices: Optional[str] = typer.Option(
        None,
        help="Comma-separated bond indices to benchmark (e.g., '0,2,5'). If not provided, auto-detect from reference directory.",
    ),
    ncores: int = typer.Option(
        30,
        help="Number of CPU cores",
    ),
    memory: float = typer.Option(
        20.0,
        help="Memory in GB",
    ),
) -> None:
    """
    Run benchmark force field torsiondrives using reference geometries.
    By default, only runs benchmarks for bonds that have reference data.
    """
    model = Model(method=method, basis=basis)

    # Parse bond indices if provided
    parsed_indices = None
    if bond_indices:
        parsed_indices = [int(x.strip()) for x in bond_indices.split(",")]

    run_benchmark_torsiondrives(
        reference_dir=reference_dir,
        model=model,
        output_dir=output_dir,
        program=program,
        ncores=ncores,
        memory=memory,
        bond_indices=parsed_indices,
    )


@app.command()
def plot(
    reference_dir: Path = typer.Argument(
        ...,
        help="Directory containing reference scan results",
    ),
    benchmark_dirs: List[Path] = typer.Argument(
        ...,
        help="One or more directories containing benchmark scan results",
    ),
    output_dir: Path = typer.Option(
        Path("analysis"),
        help="Output directory for plots and analysis",
    ),
    reference_label: str = typer.Option(
        "QM Reference",
        help="Label for reference data in plots",
    ),
    benchmark_labels: Optional[str] = typer.Option(
        None,
        help="Comma-separated labels for benchmark methods (e.g., 'UFF,MMFF94'). If not provided, uses directory names.",
    ),
) -> None:
    """
    Create energy and RMSD plots comparing reference and benchmark scans.
    Can compare multiple benchmark methods simultaneously.
    """
    # Parse benchmark labels if provided
    parsed_labels = None
    if benchmark_labels:
        parsed_labels = [x.strip() for x in benchmark_labels.split(",")]
        if len(parsed_labels) != len(benchmark_dirs):
            console.print(
                f"[red]Error: Number of labels ({len(parsed_labels)}) must match number of benchmark directories ({len(benchmark_dirs)})[/red]"
            )
            raise typer.Exit(1)

    create_analysis_plots(
        reference_dir=reference_dir,
        benchmark_dirs=benchmark_dirs,
        output_dir=output_dir,
        reference_label=reference_label,
        benchmark_labels=parsed_labels,
    )


def cli():
    with plt.style.context("ggplot"):
        app()


if __name__ == "__main__":
    cli()
