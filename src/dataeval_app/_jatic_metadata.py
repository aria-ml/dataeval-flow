"""Metadata output for JATIC compliance [IR-3-H-12].

This module provides functionality to write metadata.json files
for datasets, models, and results as required by JATIC.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataeval_app import __version__


def write_metadata(
    output_path: Path,
    dataset_id: str,
    results: dict[str, Any],
) -> Path:
    """Write metadata.json to output directory.

    Parameters
    ----------
    output_path : Path
        Output directory path where metadata.json will be written.
    dataset_id : str
        Identifier for the dataset being processed.
    results : dict[str, Any]
        Evaluation results dictionary to include in metadata.

    Returns
    -------
    Path
        Path to the written metadata.json file.

    Examples
    --------
    >>> from pathlib import Path
    >>> from dataeval_app._jatic_metadata import write_metadata
    >>> output = Path("/tmp/output")
    >>> output.mkdir(exist_ok=True)
    >>> metadata_path = write_metadata(output, "cifar10", {"accuracy": 0.95})
    >>> metadata_path.name
    'metadata.json'
    """
    metadata = {
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "tool": "dataeval-app",
        "tool_version": __version__,
        "results": results,
    }

    output_path.mkdir(parents=True, exist_ok=True)
    metadata_file = output_path / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_file
