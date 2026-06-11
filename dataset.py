from pathlib import Path

from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandSpatialCropd,
    SpatialPadd,
)

from data_loader import DATA_ROOT, MODALITY_SUFFIXES

IMAGE_KEYS = ["t1n", "t1c", "t2w", "t2f"]
LABEL_KEY = "label"
PATCH_SIZE = (128, 128, 128)

_SUFFIX_BY_KEY = {
    "t1n": MODALITY_SUFFIXES["t1"],
    "t1c": MODALITY_SUFFIXES["t1gd"],
    "t2w": MODALITY_SUFFIXES["t2"],
    "t2f": MODALITY_SUFFIXES["flair"],
    LABEL_KEY: MODALITY_SUFFIXES["seg"],
}


def _patient_file_path(patient_id: str, data_root: Path, suffix: str) -> Path:
    return data_root / patient_id / f"{patient_id}-{suffix}.nii.gz"


def build_data_list(data_root: Path = DATA_ROOT) -> list[dict[str, str]]:
    data_list: list[dict[str, str]] = []

    for patient_dir in sorted(data_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        entry: dict[str, str] = {}

        for key in IMAGE_KEYS + [LABEL_KEY]:
            suffix = _SUFFIX_BY_KEY[key]
            path = _patient_file_path(patient_id, data_root, suffix)
            if not path.exists():
                raise FileNotFoundError(f"Missing expected NIfTI file: {path}")
            entry[key] = str(path)

        data_list.append(entry)

    return data_list


def get_train_transforms(patch_size: tuple[int, int, int] = PATCH_SIZE) -> Compose:
    all_keys = IMAGE_KEYS + [LABEL_KEY]

    return Compose(
        [
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            ConcatItemsd(keys=IMAGE_KEYS, name="image", dim=0),
            DeleteItemsd(keys=IMAGE_KEYS),
            CropForegroundd(keys=["image", LABEL_KEY], source_key="image"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            SpatialPadd(keys=["image", LABEL_KEY], spatial_size=patch_size, mode="constant"),
            RandSpatialCropd(
                keys=["image", LABEL_KEY],
                roi_size=patch_size,
                random_size=False,
            ),
            EnsureTyped(keys=["image", LABEL_KEY], data_type="tensor"),
        ]
    )


def create_dataloader(
    data_root: Path = DATA_ROOT,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    data_list = build_data_list(data_root)
    dataset = Dataset(data=data_list, transform=get_train_transforms())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    loader = create_dataloader(batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"image shape: {batch['image'].shape}")
    print(f"label shape: {batch['label'].shape}")
    print(f"image dtype: {batch['image'].dtype}")
    print(f"label dtype: {batch['label'].dtype}")
