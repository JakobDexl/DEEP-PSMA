import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import math
import torch.nn.functional as F

from sklearn.model_selection import KFold
import re

from scipy import ndimage

def load_folds(in_file="folds.json"):
    """Load splits from a JSON file."""
    with open(in_file, "r") as f:
        folds = json.load(f)
    return folds


def get_combined_affine_matrix_3d(
        angles_deg=(0.0, 0.0, 0.0),
        scales=1.0,
        device="cpu"
):
    """
    Creates a 4x4 3D affine matrix with rotation (in degrees) and scaling.

    Args:
        angles_deg (tuple of 3 floats): Rotation angles in degrees for (x, y, z) axes.
        scales (float or tuple of 3 floats): Uniform or per-axis scaling factors.
        device: torch device.

    Returns:
        torch.Tensor of shape (4, 4): Combined affine matrix.
    """
    # Convert angles to radians
    rx, ry, rz = [math.radians(a) for a in angles_deg]

    # Rotation matrices
    Rx = torch.tensor([
        [1, 0, 0, 0],
        [0, math.cos(rx), -math.sin(rx), 0],
        [0, math.sin(rx), math.cos(rx), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    Ry = torch.tensor([
        [math.cos(ry), 0, math.sin(ry), 0],
        [0, 1, 0, 0],
        [-math.sin(ry), 0, math.cos(ry), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    Rz = torch.tensor([
        [math.cos(rz), -math.sin(rz), 0, 0],
        [math.sin(rz), math.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    # Combine rotations in zyx order (R = Rz * Ry * Rx)
    R = Rz @ Ry @ Rx

    # Scaling matrix
    if isinstance(scales, (float, int)):
        sx = sy = sz = scales
    elif isinstance(scales, (tuple, list)) and len(scales) == 3:
        sx, sy, sz = scales
    else:
        raise ValueError("scales must be float or a tuple of 3 floats")

    S = torch.diag(torch.tensor([sx, sy, sz, 1.0], device=device))

    # Final affine matrix
    return R @ S


def apply_small_rotation_gpu(vol_dict, angles, scale):
    """
    Apply the same small 3D rotation to volumes in a dictionary using PyTorch.

    Args:
        vol_dict: dict of 3D numpy arrays, e.g., {"FDG_PET": ..., "FDG_TTB": ..., "PSMA_TTB": ...}
        angles: tuple of 3 floats, rotation angles in degrees (x, y, z)
        scale: float or tuple of 3 floats, scaling factor(s)

    Returns:
        dict of rotated volumes (same keys as input)
    """
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4x4 affine matrix
    affine = get_combined_affine_matrix_3d(angles_deg=angles, scales=scale, device=device)

    rotated_dict = {}
    for k, vol in vol_dict.items():
        if not isinstance(vol, np.ndarray) or vol.ndim != 3:
            rotated_dict[k] = vol
            continue

        # Add batch and channel dims: [1,1,D,H,W]
        v_torch = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(device)
        _, _, D, H, W = v_torch.shape

        # Create normalized grid
        grid = F.affine_grid(
            affine[:3].unsqueeze(0),  # [1,3,4]
            size=[1, 1, D, H, W],
            align_corners=True
        )

        # Determine interpolation mode
        mode = 'bilinear' if "PET" in k or "CT" in k else 'nearest'

        v_rot = F.grid_sample(v_torch, grid, mode=mode, padding_mode='zeros', align_corners=True)
        rotated_dict[k] = v_rot.squeeze().cpu().numpy()

    return rotated_dict


def random_flip_3d(volume_dict):
    """
    Randomly flip 3D volumes along any combination of axes.
    volume_dict: dict of {key: np.ndarray} where each array is shape (Z, Y, X)
    Returns augmented dict.
    """
    # Choose random combination of axes to flip
    axes = []
    if random.random() < 0.5: axes.append(0)  # flip Z
    if random.random() < 0.5: axes.append(1)  # flip Y
    if random.random() < 0.5: axes.append(2)  # flip X

    if axes:  # only flip if there's at least one axis selected
        for k in volume_dict:
            if isinstance(volume_dict[k], np.ndarray) and volume_dict[k].ndim == 3:
                volume_dict[k] = np.flip(volume_dict[k], axis=axes).copy()
    return volume_dict


def normalize_ct(ct):
    # min and max HU values
    hu_min, hu_max = -1024.0, 3071.0

    # scale to [0, 1]
    ct = (ct - hu_min) / (hu_max - hu_min)

    # scale to [-1, 1]
    ct = ct * 2.0 - 1.0
    return ct


class PETValDataset(Dataset):
    def __init__(self, root_dir, case_ids, normalize=True, min_size=(192, 96, 128), load_ct=False):
        self.root_dir = root_dir
        self.case_ids = case_ids
        self.normalize = normalize
        self.min_size = min_size
        self.load_ct = load_ct

    def __len__(self):
        return len(self.case_ids)

    def load_case(self, idx):
        case_id = self.case_ids[idx]
        case_path = os.path.join(self.root_dir, case_id)
        return {
            "FDG": self.load_modality(case_path, "FDG"),
            "PSMA": self.load_modality(case_path, "PSMA"),
            "case_id": case_id
        }

    def pad_volume(self, vol, pad_value=-1):
        z, y, x = vol.shape
        min_z, min_y, min_x = self.min_size

        pad_z = max(0, min_z - z)
        pad_y = max(0, min_y - y)
        pad_x = max(0, min_x - x)

        pad_before = (pad_z // 2, pad_y // 2, pad_x // 2)
        pad_after = (pad_z - pad_before[0], pad_y - pad_before[1], pad_x - pad_before[2])

        pad_width = ((pad_before[0], pad_after[0]),
                     (pad_before[1], pad_after[1]),
                     (pad_before[2], pad_after[2]))

        return np.pad(vol, pad_width, mode='constant', constant_values=pad_value)

    def load_modality(self, case_path, modality):
        pet = np.load(os.path.join(case_path, modality, "PET.npy")).astype(np.float32)
        ttb = np.load(os.path.join(case_path, modality, "TTB.npy")).astype(np.float32)
        totseg = np.load(os.path.join(case_path, modality, "TOTSEG.npy")).astype(np.float32)

        thresh_path = os.path.join(case_path, modality, "threshold.json")
        with open(thresh_path, "r") as f:
            suv_threshold = float(json.load(f)["suv_threshold"])

        # Optionally load CT
        ct = None
        if self.load_ct:
            ct_path = os.path.join(case_path, modality, "CT.npy")
            if os.path.exists(ct_path):
                ct = np.load(ct_path).astype(np.float32)
                ct = self.pad_volume(ct, -1024)

        # Pad volumes
        pet = self.pad_volume(pet, 0)
        ttb = self.pad_volume(ttb, 0)
        totseg = self.pad_volume(totseg, 0)

        meta = {}
        if self.normalize:
            pet /= suv_threshold
            pet -= 1
            totseg /= 117
            if self.load_ct:
                ct = normalize_ct(ct)

        return {
            "PET": pet,
            "TTB": ttb * (pet >= 0),
            "TOTSEG": totseg,
            "CT": ct,
            "SUV_T": suv_threshold,
            "meta": meta
        }

    def __getitem__(self, idx):
        case_data = self.load_case(idx)

        output = {
            "FDG_PET": torch.from_numpy(case_data["FDG"]["PET"]),
            "FDG_TTB": torch.from_numpy(case_data["FDG"]["TTB"]),
            "FDG_TOTSEG": torch.from_numpy(case_data["FDG"]["TOTSEG"]),
            "FDG_THRESH": case_data["FDG"]["SUV_T"],
            "PSMA_PET": torch.from_numpy(case_data["PSMA"]["PET"]),
            "PSMA_TTB": torch.from_numpy(case_data["PSMA"]["TTB"]),
            "PSMA_TOTSEG": torch.from_numpy(case_data["PSMA"]["TOTSEG"]),
            "PSMA_THRESH": case_data["PSMA"]["SUV_T"],
            "case_id": case_data["case_id"],
            "FDG_meta": case_data["FDG"]["meta"],
            "PSMA_meta": case_data["PSMA"]["meta"]
        }

        if self.load_ct:
            if case_data["FDG"]["CT"] is not None:
                output["FDG_CT"] = torch.from_numpy(case_data["FDG"]["CT"])
            if case_data["PSMA"]["CT"] is not None:
                output["PSMA_CT"] = torch.from_numpy(case_data["PSMA"]["CT"])

        return output


class PETTrainDataset(Dataset):
    def __init__(self, root_dir, case_ids, crop_size=(192, 96, 128), normalize=True, load_ct=False):
        self.root_dir = root_dir
        self.case_ids = case_ids
        self.crop_size = crop_size
        self.normalize = normalize
        self.load_ct = load_ct

    def __len__(self):
        return len(self.case_ids)

    def load_case(self, idx):
        case_id = self.case_ids[idx]
        case_path = os.path.join(self.root_dir, case_id)
        return {
            "FDG": self.load_modality(case_path, "FDG"),
            "PSMA": self.load_modality(case_path, "PSMA"),
            "case_id": case_id
        }

    def load_modality(self, case_path, modality):
        pet = np.load(os.path.join(case_path, modality, "PET.npy")).astype(np.float32)
        ttb = np.load(os.path.join(case_path, modality, "TTB.npy")).astype(np.float32)
        totseg = np.load(os.path.join(case_path, modality, "TOTSEG.npy")).astype(np.float32)

        thresh_path = os.path.join(case_path, modality, "threshold.json")
        with open(thresh_path, "r") as f:
            suv_threshold = float(json.load(f)["suv_threshold"])

        ct = None
        if self.load_ct:
            ct_path = os.path.join(case_path, modality, "CT.npy")
            if os.path.exists(ct_path):
                ct = np.load(ct_path).astype(np.float32)

        meta = {}
        if self.normalize:
            pet /= suv_threshold
            # pet -= 1
            totseg /= 117
            if self.load_ct and ct is not None:
                ct = normalize_ct(ct)

        return {
            "PET": pet,
            "TTB": ttb,
            "TOTSEG": totseg,
            "CT": ct,
            "SUV_T": suv_threshold,
            "meta": meta
        }

    def random_crop_coords_from_mask(self, mask):
        """Pick a crop starting coord from a random positive voxel in mask."""
        pos_indices = np.argwhere(mask > 0)
        if len(pos_indices) > 0:
            center_z, center_y, center_x = pos_indices[np.random.randint(len(pos_indices))]
        else:
            center_z = random.randint(0, mask.shape[0] - 1)
            center_y = random.randint(0, mask.shape[1] - 1)
            center_x = random.randint(0, mask.shape[2] - 1)

        cz, cy, cx = self.crop_size
        return center_z - cz // 2, center_y - cy // 2, center_x - cx // 2

    def crop_volume(self, vol, coords, pad_value=0):
        z, y, x = coords
        cz, cy, cx = self.crop_size
        vol_z, vol_y, vol_x = vol.shape
        end_z, end_y, end_x = z + cz, y + cy, x + cx
        pad_before_z = max(0, -z)
        pad_after_z = max(0, end_z - vol_z)
        pad_before_y = max(0, -y)
        pad_after_y = max(0, end_y - vol_y)
        pad_before_x = max(0, -x)
        pad_after_x = max(0, end_x - vol_x)
        z1, y1, x1 = max(z, 0), max(y, 0), max(x, 0)
        z2, y2, x2 = min(end_z, vol_z), min(end_y, vol_y), min(end_x, vol_x)
        cropped = vol[z1:z2, y1:y2, x1:x2]
        pad_width = ((pad_before_z, pad_after_z),
                     (pad_before_y, pad_after_y),
                     (pad_before_x, pad_after_x))
        return np.pad(cropped, pad_width, mode='constant', constant_values=pad_value)

    def __getitem__(self, idx):
        case_data = self.load_case(idx)

        fdg = case_data["FDG"]
        psma = case_data["PSMA"]

        suv_thresh_mask_prob = 1 / 3.0
        # Union of masks
        mask_union = (fdg["TTB"] > 0) | (psma["TTB"] > 0)

        # SUV > thresh masks (PET normalized so >0 == above thresh)
        suv_mask = (fdg["PET"] >= 1) | (psma["PET"] >= 1)

        # suspicious region: SUV high but not labeled
        tta_mask = suv_mask & (~mask_union)

        if random.random() < suv_thresh_mask_prob and np.any(tta_mask):
            sampler = tta_mask.astype(np.uint8)
        else:
            sampler = mask_union.astype(np.uint8)

        crop_coords = self.random_crop_coords_from_mask(sampler)

        vols = {
            "FDG_PET": self.crop_volume(fdg["PET"], crop_coords, 0),
            "FDG_TTB": self.crop_volume(fdg["TTB"] * (fdg["PET"] >= 1), crop_coords, 0),
            "FDG_TOTSEG": self.crop_volume(fdg["TOTSEG"], crop_coords, 0),
            "PSMA_PET": self.crop_volume(psma["PET"], crop_coords, 0),
            "PSMA_TTB": self.crop_volume(psma["TTB"] * (psma["PET"] >= 1), crop_coords, 0),
            "PSMA_TOTSEG": self.crop_volume(psma["TOTSEG"], crop_coords, 0),
        }

        if self.load_ct:
            if fdg["CT"] is not None:
                vols["FDG_CT"] = self.crop_volume(fdg["CT"], crop_coords, -1)
            if psma["CT"] is not None:
                vols["PSMA_CT"] = self.crop_volume(psma["CT"], crop_coords, -1)

        # TrivialAugment
        ops = []

        def random_rotation(vols, max_angle=10):
            angles = [random.uniform(-max_angle, max_angle) for _ in range(3)]
            return apply_small_rotation_gpu(vols, angles, (1., 1., 1.))

        def random_scale(vols, ran=(.9, 1.3)):
            scale = [random.uniform(ran[0], ran[1]) for _ in range(3)]
            return apply_small_rotation_gpu(vols, (0., 0., 0.), scale)

        def remove_tracer(vols):
            tracer = random.choice(["FDG", "PSMA"])
            for key in list(vols.keys()):
                if key.endswith("_TTB"):
                    continue
                elif key.startswith(tracer):
                    vols[key] = np.zeros_like(vols[key], dtype=vols[key].dtype)
            return vols


        ops.append(lambda: vols)
        ops.append(lambda: random_rotation(vols, max_angle=10))
        ops.append(lambda: random_scale(vols, ran=(.9, 1.4)))
        ops.append(lambda: random_flip_3d(vols))

        # Randomly select one
        op = random.choice(ops)
        vols = op()

        # norm to thresh = 0
        vols["FDG_PET"] -= 1
        vols["PSMA_PET"] -= 1

        output = {k: torch.from_numpy(v) for k, v in vols.items()}
        output.update({
            "FDG_THRESH": fdg["SUV_T"],
            "PSMA_THRESH": psma["SUV_T"],
            "case_id": case_data["case_id"],
            "FDG_meta": fdg["meta"],
            "PSMA_meta": psma["meta"]
        })
        return output

class PETDataModule(pl.LightningDataModule):
    def __init__(
            self,
            root_dir,
            case_ids,
            batch_size=1,
            num_workers=4,
            use_ct=False,
            crop_size=(192, 192, 192),
            normalize=True,
            fold=0,
            seed=42,
            folds_file=None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.case_ids = case_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_ct = use_ct
        self.crop_size = crop_size
        self.normalize = normalize
        self.fold = fold
        self.seed = seed
        self.folds_file = folds_file

    def setup(self, stage=None):
        if self.folds_file is not None:
            # Load from JSON instead of recomputing
            folds = load_folds(self.folds_file)
            self.train_ids = folds[str(self.fold)]["train"]
            self.val_ids = folds[str(self.fold)]["val"]
        else:
            base_case_ids = sorted(set(
                re.sub(r"_(to_fdg|to_psma)$", "", cid) for cid in self.case_ids
            ))

            # Split by base case ID
            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            train_idx, val_idx = list(kf.split(base_case_ids))[self.fold]

            train_base = {base_case_ids[i] for i in train_idx}
            val_base = {base_case_ids[i] for i in val_idx}

            # Map back to all matching case_ids
            self.train_ids = [cid for cid in self.case_ids if re.sub(r"_(to_fdg|to_psma)$", "", cid) in train_base]
            self.val_ids = [cid for cid in self.case_ids if re.sub(r"_(to_fdg|to_psma)$", "", cid) in val_base]

        # Create datasets
        self.train_dataset = PETTrainDataset(
            self.root_dir,
            self.train_ids,
            # crops_per_case=self.crops_per_case,
            crop_size=self.crop_size,
            normalize=self.normalize,
            load_ct=self.use_ct

        )
        self.val_dataset = PETValDataset(
            self.root_dir,
            self.val_ids,
            min_size=self.crop_size,
            normalize=self.normalize,
            load_ct=self.use_ct
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

