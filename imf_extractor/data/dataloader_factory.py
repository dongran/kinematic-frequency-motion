from typing import Optional

from torch.utils.data import DataLoader


def build_dataloader_by_type(
    *,
    dataset_type: Optional[str],
    root: str,
    split: str,
    split_dir: Optional[str],
    imf_dir: Optional[str],
    stats_dir: Optional[str],
    sample_weight_path: Optional[str],
    sample_weight_replacement: bool,
    batch_size: int,
    num_workers: int,
    max_motion_length: int,
    min_motion_length: int,
    unit_length: int,
    debug: bool = False,
) -> DataLoader:
    """Factory to build dataloaders for different input modalities.

    - humanml_npy (default): new_joint_vecs/*.npy + imfs3/*.npy (263@20Hz baseline)
    - most_npz: MoST train.npz clips + bvhrot63 imfs3 labels (231@30Hz)
    """

    dt = str(dataset_type or "humanml_npy").lower()
    if dt in ("humanml_npy", "humanml", "npy", "default"):
        from data.imf_dataset import build_dataloader

        return build_dataloader(
            root=root,
            split=split,
            split_dir=split_dir,
            imf_dir=imf_dir,
            stats_dir=stats_dir,
            sample_weight_path=sample_weight_path,
            sample_weight_replacement=sample_weight_replacement,
            batch_size=batch_size,
            num_workers=num_workers,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            unit_length=unit_length,
            debug=debug,
        )

    if dt in ("most_npz", "most", "npz"):
        from data.imf_most_dataset import build_most_dataloader

        return build_most_dataloader(
            root=root,
            split=split,
            split_dir=split_dir,
            imf_dir=imf_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            unit_length=unit_length,
            debug=debug,
        )

    raise ValueError(f"Unknown dataset.type: {dataset_type}")

