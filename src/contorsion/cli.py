"""Small CLI shim to expose the Typer app as a clean entry point.

This module intentionally keeps to a single import so the entry point
can be referenced as `contorsion.cli:app` in packaging metadata.
"""

from .drive_torsions import app

__all__ = ["app"]
