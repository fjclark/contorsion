import subprocess
import sys
from pathlib import Path

def test_small_molecule_torsion_scan(tmp_path):

    output_dir = tmp_path / "reference_scans"

    command = [
        "contorsion",
        "run-reference-scans",
        "CCO",
        "--method", "small",
        "--program", "mace",
        "--output-dir", str(output_dir),
        "--grid-spacing", "15",
        "--ncores", "4",
    ]
    
    # Run a torsion scan
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Check that the command completed successfully
    assert result.returncode == 0, (
        f"Contorsion command failed!\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    # Check that the output directory was created
    assert output_dir.exists(), "Output directory was not created"

    # Check that the output directory contains files
    output_files = list(output_dir.rglob("*"))
    assert any(path.is_file() for path in output_files), (
        "No output files were generated"
    )
