from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

PATIENT_ID = "BraTS-GLI-00009-100"
DATA_ROOT = Path("./data/raw_gli/training_data1_v2")

MODALITY_SUFFIXES = {
    "t1": "t1n",
    "t1gd": "t1c",
    "t2": "t2w",
    "flair": "t2f",
    "seg": "seg",
}

MODALITY_TITLES = {
    "t1": "T1",
    "t1gd": "T1Gd",
    "t2": "T2",
    "flair": "FLAIR",
    "seg": "Segmentation",
}

SEG_LABELS = {
    0: "Background",
    1: "NCR",
    2: "ED",
    3: "ET",
}


def load_nifti(path: Path) -> np.ndarray:
    return nib.load(path).get_fdata()


def _patient_file_path(patient_id: str, data_root: Path, suffix: str) -> Path:
    return data_root / patient_id / f"{patient_id}-{suffix}.nii.gz"


def load_patient(patient_id: str, data_root: Path) -> dict[str, np.ndarray]:
    volumes: dict[str, np.ndarray] = {}

    for modality, suffix in MODALITY_SUFFIXES.items():
        path = _patient_file_path(patient_id, data_root, suffix)
        if not path.exists():
            raise FileNotFoundError(f"Missing expected NIfTI file: {path}")
        volumes[modality] = load_nifti(path)

    shapes = {modality: volume.shape for modality, volume in volumes.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        raise ValueError(f"Modalities have mismatched shapes: {shapes}")

    return volumes


def plot_axial_slice(
    volumes: dict[str, np.ndarray],
    depth_idx: int = 75,
    patient_id: str = PATIENT_ID,
) -> None:
    display_order = ["t1", "t1gd", "t2", "flair", "seg"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    seg_colors = ["black", "red", "green", "yellow"]
    seg_cmap = ListedColormap(seg_colors)
    seg_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], seg_cmap.N)

    for ax, modality in zip(axes, display_order):
        slice_2d = volumes[modality][:, :, depth_idx]

        if modality == "seg":
            im = ax.imshow(slice_2d, cmap=seg_cmap, norm=seg_norm, interpolation="nearest")
            cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
            cbar.ax.set_yticklabels([SEG_LABELS[i] for i in range(4)])
        else:
            ax.imshow(slice_2d, cmap="gray")

        ax.set_title(MODALITY_TITLES[modality])
        ax.axis("off")

    fig.suptitle(f"{patient_id} — axial slice {depth_idx}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    volumes = load_patient(PATIENT_ID, DATA_ROOT)
    plot_axial_slice(volumes, depth_idx=75, patient_id=PATIENT_ID)
