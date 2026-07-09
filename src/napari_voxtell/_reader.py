"""napari reader that loads NIfTI volumes in VoxTell's training orientation.

VoxTell was trained on images read with nnU-Net's ``NibabelIOWithReorient``, which
reorients to RAS (``nibabel.as_reoriented(io_orientation(affine))``) and transposes
to SimpleITK axis order. Loading the same file with napari's built-in / napari-nifti
reader yields the *raw* voxel array in a different orientation, which the model has
never seen — producing mis-aligned, low-quality masks (e.g. AMOS).

By loading through this reader, the displayed layer is already in the model's input
space, so the widget can feed ``layer.data`` straight to the predictor and overlay
the resulting mask without any further reorientation.
"""

import os
from typing import Callable, Optional

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

SUPPORTED_SUFFIXES = (".nii", ".nii.gz")


def napari_get_reader(path) -> Optional[Callable]:
    """Return the VoxTell reader if ``path`` is a NIfTI file, else None."""
    candidate = path[0] if isinstance(path, list) else path
    if isinstance(candidate, str) and candidate.lower().endswith(SUPPORTED_SUFFIXES):
        return reader_function
    return None


def _strip_nifti_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def reader_function(path):
    """Read one or more NIfTI files into RAS-reoriented napari image layers."""
    paths = [path] if isinstance(path, str) else path
    reader = NibabelIOWithReorient()

    layer_data = []
    for p in paths:
        # data: (1, Z, Y, X) in RAS-reoriented, SimpleITK-transposed order;
        # props carries the original + reoriented affine for an exact inverse.
        data, props = reader.read_images([p])
        data = data[0]
        # spacing is (z, y, x), matching the data axes after the transpose.
        spacing = props["spacing"]
        meta = {
            "name": _strip_nifti_suffix(os.path.basename(p)),
            "scale": spacing,
            "metadata": {
                "voxtell_reoriented": True,
                "voxtell_props": props,
                "voxtell_source_path": p,
            },
        }
        layer_data.append((data, meta, "image"))
    return layer_data
