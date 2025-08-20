import numpy as np
import os
import shutil
import json
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def save_mip_images(
    fdg_pet,
    psma_pet,
    fdg_ct,
    psma_ct,  # originals
    fdg_pet_proc,
    psma_pet_proc,
    fdg_ct_proc,
    psma_ct_proc,  # processed/aligned
    fdg_ttb=None,
    psma_ttb=None,  # aligned masks (bottom row)
    fdg_ttb_orig=None,
    psma_ttb_orig=None,  # original masks (top row)
    case_name="Case",
    target_dir=None,
    view=2,
    dpi=150,
):
    """
    Make a 2x4 figure:
      Row 1: FDG PET (orig), FDG CT (orig), PSMA PET (orig), PSMA CT (orig)
      Row 2: FDG PET (proc), FDG CT (proc), PSMA PET (proc), PSMA CT (proc)
    PET is clipped to 0–20 SUV. TTB masks (if provided) are drawn as cyan contours
    on PET panels only (original masks on row 1, aligned masks on row 2).
    No CT overlays.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import SimpleITK as sitk

    def pet_mip(img):
        arr = sitk.GetArrayFromImage(img)
        arr = np.clip(arr, 0, 20)
        return np.amax(arr, axis=view)

    def ct_mip(img):
        arr = sitk.GetArrayFromImage(img)
        return np.amax(arr, axis=view)

    def mask_mip(mask_img):
        arr = sitk.GetArrayFromImage(mask_img)
        return np.amax(arr, axis=view)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # --- Row 1: originals ---
    ax = axes[0, 0]
    ax.imshow(pet_mip(fdg_pet), cmap="gray_r", origin="lower")
    if fdg_ttb_orig is not None:
        ax.contour(mask_mip(fdg_ttb_orig), levels=[0.5], colors="cyan", linewidths=1)
    ax.set_title(f"{case_name} FDG PET (orig)")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(ct_mip(fdg_ct), cmap="bone", origin="lower")
    ax.set_title(f"{case_name} FDG CT (orig)")
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(pet_mip(psma_pet), cmap="gray_r", origin="lower")
    if psma_ttb_orig is not None:
        ax.contour(mask_mip(psma_ttb_orig), levels=[0.5], colors="cyan", linewidths=1)
    ax.set_title(f"{case_name} PSMA PET (orig)")
    ax.axis("off")

    ax = axes[0, 3]
    ax.imshow(ct_mip(psma_ct), cmap="bone", origin="lower")
    ax.set_title(f"{case_name} PSMA CT (orig)")
    ax.axis("off")

    # --- Row 2: processed/aligned ---
    ax = axes[1, 0]
    ax.imshow(pet_mip(fdg_pet_proc), cmap="gray_r", origin="lower")
    if fdg_ttb is not None:
        ax.contour(mask_mip(fdg_ttb), levels=[0.5], colors="cyan", linewidths=1)
    ax.set_title(f"{case_name} FDG PET (proc)")
    ax.axis("off")

    ax = axes[1, 1]
    ax.imshow(ct_mip(fdg_ct_proc), cmap="bone", origin="lower")
    ax.set_title(f"{case_name} FDG CT (proc)")
    ax.axis("off")

    ax = axes[1, 2]
    ax.imshow(pet_mip(psma_pet_proc), cmap="gray_r", origin="lower")
    if psma_ttb is not None:
        ax.contour(mask_mip(psma_ttb), levels=[0.5], colors="cyan", linewidths=1)
    ax.set_title(f"{case_name} PSMA PET (proc)")
    ax.axis("off")

    ax = axes[1, 3]
    ax.imshow(ct_mip(psma_ct_proc), cmap="bone", origin="lower")
    ax.set_title(f"{case_name} PSMA CT (proc)")
    ax.axis("off")

    plt.tight_layout()
    if target_dir is not None:
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, f"{case_name}_mip.png")
        plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_aligned_case(aligned_data, src_path, target_path, case_id):
    """
    Save aligned FDG/PSMA images as numpy arrays with optimized dtypes:
      - PET/SUV -> float32
      - CT      -> int16
      - Masks   -> uint8
    Keep folder structure, copy threshold.json.
    """
    dtype_map = {"PET": np.float32, "CT": np.int16, "TTB": np.uint8, "TOTSEG": np.uint8}

    for tracer in ["FDG", "PSMA"]:
        tracer_src = os.path.join(src_path, tracer)
        tracer_target = os.path.join(target_path, case_id, tracer)
        os.makedirs(tracer_target, exist_ok=True)

        # Copy threshold JSON
        json_src = os.path.join(tracer_src, "threshold.json")
        if os.path.exists(json_src):
            shutil.copy(json_src, tracer_target)

        # Save all aligned volumes as numpy with correct dtype
        for key, img in aligned_data[tracer].items():
            arr = sitk.GetArrayFromImage(img)

            # Select dtype from map, default to float32 if unknown
            target_dtype = dtype_map.get(key.upper(), np.float32)
            arr = arr.astype(target_dtype)

            np.save(os.path.join(tracer_target, f"{key}.npy"), arr)


def save_aligned_case_nii(aligned_data, src_path, target_path, case_id):
    """
    Save aligned FDG/PSMA images as compressed NIfTI (.nii.gz) with optimized dtypes:
      - PET/SUV -> float32
      - CT      -> int16
      - Masks   -> uint8
    Keep folder structure, copy threshold.json.
    """
    dtype_map = {"PET": sitk.sitkFloat32, "CT": sitk.sitkInt16, "TTB": sitk.sitkUInt8, "TOTSEG": sitk.sitkUInt8}

    for tracer in ["FDG", "PSMA"]:
        tracer_src = os.path.join(src_path, tracer)
        tracer_target = os.path.join(target_path, case_id, tracer)
        os.makedirs(tracer_target, exist_ok=True)

        # Copy threshold JSON
        json_src = os.path.join(tracer_src, "threshold.json")
        if os.path.exists(json_src):
            shutil.copy(json_src, tracer_target)

        # Save all aligned volumes as .nii.gz with correct dtype
        for key, img in aligned_data[tracer].items():
            # Get the target pixel type (default float32 if unknown)
            sitk_dtype = dtype_map.get(key.upper(), sitk.sitkFloat32)

            # Cast image to correct type
            img_cast = sitk.Cast(img, sitk_dtype)

            # Save compressed NIfTI
            out_path = os.path.join(tracer_target, f"{key}.nii.gz")
            sitk.WriteImage(img_cast, out_path, useCompression=True)


def prepare_case_with_masks(
    fdg_ct,
    fdg_pet,
    fdg_ttb,
    fdg_totseg,
    psma_ct,
    psma_pet,
    psma_ttb,
    psma_totseg,
    fdg_to_psma_transform_path,  # transform that maps FDG → PSMA
    psma_to_fdg_transform_path,  # transform that maps PSMA → FDG
    fdg_thresh=0,
    psma_thresh=0,
    register_to="PSMA",  # "PSMA" or "FDG"
):
    """
    Prepare a single case for a specified registration target:
    - If register_to == "PSMA":  fixed = PSMA, moving = FDG, use FDG→PSMA transform.
    - If register_to == "FDG":   fixed = FDG,  moving = PSMA, use PSMA→FDG transform.
    - Resample CT/TTB/TOTSEG to their own PET spacing first (keeps tracer-consistent spacing).
    - Apply the chosen rigid transform to align moving → fixed.
    - Crop all aligned volumes to the union bbox of PETs above thresholds (with margin).
    Returns a dict with aligned images and both transforms.
    """
    import numpy as np
    import SimpleITK as sitk

    # ---- Helpers ----
    def resample_to_reference(img, reference, interp=sitk.sitkLinear, default_value=0):
        return sitk.Resample(img, reference, sitk.Transform(), interp, default_value, img.GetPixelID())

    # ---- Read transforms ----
    fdg_to_psma_transform = sitk.ReadTransform(fdg_to_psma_transform_path)
    psma_to_fdg_transform = sitk.ReadTransform(psma_to_fdg_transform_path)

    # ---- Resample each tracer's CT/TTB/TOTSEG to that tracer's PET ----
    fdg_ct_res = resample_to_reference(fdg_ct, fdg_pet, sitk.sitkLinear, -1024)
    fdg_ttb_res = resample_to_reference(fdg_ttb, fdg_pet, sitk.sitkNearestNeighbor, 0)
    fdg_totseg_res = resample_to_reference(fdg_totseg, fdg_pet, sitk.sitkNearestNeighbor, 0)

    psma_ct_res = resample_to_reference(psma_ct, psma_pet, sitk.sitkLinear, -1024)
    psma_ttb_res = resample_to_reference(psma_ttb, psma_pet, sitk.sitkNearestNeighbor, 0)
    psma_totseg_res = resample_to_reference(psma_totseg, psma_pet, sitk.sitkNearestNeighbor, 0)

    # ---- Choose direction explicitly ----
    register_to = register_to.upper()
    if register_to == "PSMA":
        fixed = {"CT": psma_ct_res, "PET": psma_pet, "TTB": psma_ttb_res, "TOTSEG": psma_totseg_res}
        moving = {"CT": fdg_ct_res, "PET": fdg_pet, "TTB": fdg_ttb_res, "TOTSEG": fdg_totseg_res}
        transform = fdg_to_psma_transform  # FDG → PSMA
    elif register_to == "FDG":
        fixed = {"CT": fdg_ct_res, "PET": fdg_pet, "TTB": fdg_ttb_res, "TOTSEG": fdg_totseg_res}
        moving = {"CT": psma_ct_res, "PET": psma_pet, "TTB": psma_ttb_res, "TOTSEG": psma_totseg_res}
        transform = psma_to_fdg_transform  # PSMA → FDG
    else:
        raise ValueError("register_to must be 'PSMA' or 'FDG'")

    # ---- Apply registration (moving → fixed) ----
    aligned_moving = {}
    for key in moving:
        interp = sitk.sitkLinear if key in ["CT", "PET"] else sitk.sitkNearestNeighbor
        aligned_moving[key] = sitk.Resample(moving[key], fixed["CT"], transform, interp, 0.0, moving[key].GetPixelID())

    # ---- Merge dict into aligned space ----
    if register_to == "PSMA":
        aligned = {"FDG": aligned_moving, "PSMA": fixed}
    else:  # register_to == "FDG"
        aligned = {"FDG": fixed, "PSMA": aligned_moving}

    # ---- Crop to union bbox of PETs above thresholds (z,y,x order from arrays) ----
    fdg_pet_arr = sitk.GetArrayFromImage(aligned["FDG"]["PET"])
    psma_pet_arr = sitk.GetArrayFromImage(aligned["PSMA"]["PET"])
    union_mask = (fdg_pet_arr > fdg_thresh) | (psma_pet_arr > psma_thresh)

    coords = np.array(np.where(union_mask))
    if coords.size:
        # Expand by 5 voxels on all sides
        min_idx = coords.min(axis=1) - 5
        max_idx = coords.max(axis=1) + 6
        # Clamp to image bounds
        shape = fdg_pet_arr.shape
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, shape)

        size = (max_idx - min_idx).tolist()
        start = min_idx.tolist()
        # SITK uses (x,y,z)
        start_sitk = [start[2], start[1], start[0]]
        size_sitk = [size[2], size[1], size[0]]

        for tracer in ["FDG", "PSMA"]:
            for key in ["CT", "PET", "TTB", "TOTSEG"]:
                aligned[tracer][key] = sitk.RegionOfInterest(aligned[tracer][key], size_sitk, start_sitk)

    # ---- Return aligned images + both transforms (for saving) ----
    aligned["fdg_to_psma_transform"] = fdg_to_psma_transform
    aligned["psma_to_fdg_transform"] = psma_to_fdg_transform
    aligned["registered_to"] = register_to
    return aligned


# -----------------------------------------------------------------------------------------------------------------------
base = "../data/CHALLENGE_DATA/"
target_folder = "../data/deep-psma-preprocessed-bothways/"
fdg_path = "../data/FDG_updated_params"
cases = os.listdir(base)


# os.makedirs(target_folder)
# -----------------------------------------------------------------------------------------------------------------------
def process_case(case_id):
    src_path = os.path.join(base, case_id)
    try:
        # --- Load FDG ---
        fdg_ct = sitk.ReadImage(os.path.join(src_path, "FDG", "CT.nii.gz"))
        fdg_pet = sitk.ReadImage(os.path.join(src_path, "FDG", "PET.nii.gz"))
        fdg_ttb = sitk.ReadImage(os.path.join(src_path, "FDG", "TTB.nii.gz"))
        fdg_totseg = sitk.ReadImage(os.path.join(src_path, "FDG", "totseg_24.nii.gz"))
        with open(os.path.join(src_path, "FDG", "threshold.json"), "r") as f:
            fdg_thresh = json.load(f)["suv_threshold"]

        # --- Load PSMA ---
        psma_ct = sitk.ReadImage(os.path.join(src_path, "PSMA", "CT.nii.gz"))
        psma_pet = sitk.ReadImage(os.path.join(src_path, "PSMA", "PET.nii.gz"))
        psma_ttb = sitk.ReadImage(os.path.join(src_path, "PSMA", "TTB.nii.gz"))
        psma_totseg = sitk.ReadImage(os.path.join(src_path, "PSMA", "totseg_24.nii.gz"))
        with open(os.path.join(src_path, "PSMA", "threshold.json"), "r") as f:
            psma_thresh = json.load(f)["suv_threshold"]

        # Paths to transforms (FDG→PSMA and PSMA→FDG)
        psma_to_fdg_path = os.path.join(fdg_path, f"{case_id}_fdg_rigid_params.tfm")
        fdg_to_psma_path = os.path.join(src_path, "PSMA", "rigid.tfm")

        # -----------------------------
        # 1) Register explicitly TO PSMA
        # -----------------------------
        aligned_to_psma = prepare_case_with_masks(
            fdg_ct,
            fdg_pet,
            fdg_ttb,
            fdg_totseg,
            psma_ct,
            psma_pet,
            psma_ttb,
            psma_totseg,
            fdg_to_psma_path,
            psma_to_fdg_path,
            fdg_thresh,
            psma_thresh,
            register_to="PSMA",
        )

        # Save aligned volumes/JSON (suffix on case_id)
        save_aligned_case(aligned_to_psma, src_path, target_folder, f"{case_id}_to_psma")

        # Save MIPs (suffix on case_name)
        mip_target = os.path.join(target_folder, "mips")
        save_mip_images(
            fdg_pet=fdg_pet,
            psma_pet=psma_pet,
            fdg_ct=fdg_ct,
            psma_ct=psma_ct,
            fdg_pet_proc=aligned_to_psma["FDG"]["PET"],
            psma_pet_proc=aligned_to_psma["PSMA"]["PET"],
            fdg_ct_proc=aligned_to_psma["FDG"]["CT"],
            psma_ct_proc=aligned_to_psma["PSMA"]["CT"],
            fdg_ttb=aligned_to_psma["FDG"]["TTB"],
            psma_ttb=aligned_to_psma["PSMA"]["TTB"],
            fdg_ttb_orig=fdg_ttb,
            psma_ttb_orig=psma_ttb,
            case_name=f"{case_id}_aligned_to_psma",
            target_dir=mip_target,
        )

        # ----------------------------
        # 2) Register explicitly TO FDG
        # ----------------------------
        aligned_to_fdg = prepare_case_with_masks(
            fdg_ct,
            fdg_pet,
            fdg_ttb,
            fdg_totseg,
            psma_ct,
            psma_pet,
            psma_ttb,
            psma_totseg,
            fdg_to_psma_path,
            psma_to_fdg_path,
            fdg_thresh,
            psma_thresh,
            register_to="FDG",
        )

        save_aligned_case(aligned_to_fdg, src_path, target_folder, f"{case_id}_to_fdg")

        save_mip_images(
            fdg_pet=fdg_pet,
            psma_pet=psma_pet,
            fdg_ct=fdg_ct,
            psma_ct=psma_ct,
            fdg_pet_proc=aligned_to_fdg["FDG"]["PET"],
            psma_pet_proc=aligned_to_fdg["PSMA"]["PET"],
            fdg_ct_proc=aligned_to_fdg["FDG"]["CT"],
            psma_ct_proc=aligned_to_fdg["PSMA"]["CT"],
            fdg_ttb=aligned_to_fdg["FDG"]["TTB"],
            psma_ttb=aligned_to_fdg["PSMA"]["TTB"],
            fdg_ttb_orig=fdg_ttb,
            psma_ttb_orig=psma_ttb,
            case_name=f"{case_id}_aligned_to_fdg",
            target_dir=mip_target,
        )

        return f"[INFO] Saved case {case_id} (to_psma & to_fdg)"
    except Exception as e:
        return f"[ERROR] Case {case_id} failed: {e}"


# Run in parallel with 4 workers
with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(process_case, cases), total=len(cases)))

# Print logs after completion
for r in results:
    print(r)
