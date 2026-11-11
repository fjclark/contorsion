"""QCEngine harness for any machine-learned potential supported by OpenMM-ML.

The QCElemental model method is passed directly to ``openmmml.MLPotential``,
e.g. ``orb-v3-conservative-omol``, ``mace-off23-small``, ``aimnet2`` or ``ani2x``.
Optional keywords (``device`` and any other ``createSystem`` arguments) are
forwarded to ``MLPotential.createSystem``.
"""

from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
from qcelemental.models import AtomicResult, Provenance
from qcelemental.util import safe_version, which_import
from qcengine.exceptions import InputError
from qcengine.programs.model import ProgramHarness
from qcengine.units import ureg

if TYPE_CHECKING:
    import openmm
    from qcelemental.models import AtomicInput
    from qcengine.config import TaskConfig


_KJ_PER_MOL_TO_HARTREE = ureg.conversion_factor("kJ/mol", "hartree")
_BOHR_TO_NM = ureg.conversion_factor("bohr", "nm")
# pint has no molar context for forces, so build it from energy and length
_KJ_PER_MOL_NM_TO_HARTREE_PER_BOHR = _KJ_PER_MOL_TO_HARTREE * _BOHR_TO_NM


class OpenMMMLHarness(ProgramHarness):
    """Evaluate energies and gradients with an OpenMM-ML potential."""

    # Contexts are cached because building one loads the ML model, and
    # geomeTRIC calls compute() hundreds of times per torsion scan.
    _CACHE: Dict[Tuple, "openmm.Context"] = {}

    _defaults = {
        "name": "openmm-ml",
        "scratch": False,
        "thread_safe": False,
        "thread_parallel": False,
        "node_parallel": False,
        "managed_memory": False,
    }

    class Config(ProgramHarness.Config):
        pass

    @staticmethod
    def found(raise_error: bool = False) -> bool:
        return which_import(
            "openmmml",
            return_bool=True,
            raise_error=raise_error,
            raise_msg="Please install via `pixi add openmm-ml`",
        ) and which_import(
            "openff.toolkit",
            return_bool=True,
            raise_error=raise_error,
            raise_msg="Please install via `pixi add openff-toolkit`",
        )

    def get_version(self) -> str:
        self.found(raise_error=True)
        return safe_version(version("openmmml"))

    def _get_context(
        self,
        mapped_smiles: str,
        method: str,
        charge: int,
        multiplicity: int,
        keywords: Dict[str, Any],
    ) -> "openmm.Context":
        key = (
            mapped_smiles,
            method,
            charge,
            multiplicity,
            tuple(sorted((k, str(v)) for k, v in keywords.items())),
        )
        if key in self._CACHE:
            return self._CACHE[key]

        import openmm
        from openmm import unit
        from openmmml import MLPotential
        from openff.toolkit import Molecule

        off_mol = Molecule.from_mapped_smiles(
            mapped_smiles, allow_undefined_stereo=True
        )
        topology = off_mol.to_topology().to_openmm()
        system = MLPotential(method).createSystem(
            topology,
            charge=charge,
            multiplicity=multiplicity,
            **keywords,
        )
        # The ML model runs on its own torch device inside the force, so the
        # OpenMM platform only handles the (trivial) rest of the system.
        context = openmm.Context(
            system,
            openmm.VerletIntegrator(1.0 * unit.femtoseconds),
            openmm.Platform.getPlatformByName("Reference"),
        )
        self._CACHE[key] = context
        return context

    def compute(
        self, input_data: "AtomicInput", config: "TaskConfig"
    ) -> AtomicResult:
        self.found(raise_error=True)

        extras = input_data.molecule.extras or {}
        mapped_smiles = extras.get(
            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        ) or extras.get("cmiles", {}).get(
            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        )
        if mapped_smiles is None:
            raise InputError(
                "The openmm-ml harness needs a mapped SMILES in molecule.extras "
                "['canonical_isomeric_explicit_hydrogen_mapped_smiles']."
            )

        method = input_data.model.method
        context = self._get_context(
            mapped_smiles=mapped_smiles,
            method=method,
            charge=int(round(input_data.molecule.molecular_charge)),
            multiplicity=int(input_data.molecule.molecular_multiplicity),
            keywords=dict(input_data.keywords),
        )

        positions = np.asarray(input_data.molecule.geometry) * _BOHR_TO_NM
        context.setPositions(positions)
        state = context.getState(getEnergy=True, getForces=True)

        from openmm import unit

        energy = (
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            * _KJ_PER_MOL_TO_HARTREE
        )
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )

        if input_data.driver == "energy":
            return_result: Any = energy
        elif input_data.driver == "gradient":
            return_result = (
                -forces * _KJ_PER_MOL_NM_TO_HARTREE_PER_BOHR
            ).ravel().tolist()
        else:
            raise InputError(
                "The openmm-ml harness only supports the energy and gradient drivers."
            )

        ret_data = {
            "properties": {"return_energy": energy},
            "return_result": return_result,
            "extras": input_data.extras.copy(),
            "provenance": Provenance(
                creator="openmm-ml", version=self.get_version(), routine=method
            ),
            "schema_name": "qcschema_output",
            "success": True,
        }
        return AtomicResult(**{**input_data.dict(), **ret_data})
