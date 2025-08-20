import numpy as np
import SimpleITK as sitk
import torch


def prepare_case(
    fdg_ct,
    fdg_pet,
    fdg_totseg,
    psma_ct,
    psma_pet,
    psma_totseg,
    fdg_to_psma_transform_path,
    psma_to_fdg_transform_path,
    fdg_thresh=0,
    psma_thresh=0,
    register_to="PSMA",
    min_size=(256, 128, 128),
):
    import numpy as np
    import SimpleITK as sitk

    def resample_to_reference(img, reference, interp=sitk.sitkLinear, default_value=0):
        return sitk.Resample(img, reference, sitk.Transform(), interp, default_value, img.GetPixelID())

    def pad_to_min_size_sitk(img, min_size, pad_value=0):
        arr = sitk.GetArrayFromImage(img)
        z, y, x = arr.shape
        min_z, min_y, min_x = min_size

        pad_z = max(0, min_z - z)
        pad_y = max(0, min_y - y)
        pad_x = max(0, min_x - x)

        pad_before = (pad_z // 2, pad_y // 2, pad_x // 2)
        pad_after = (pad_z - pad_before[0], pad_y - pad_before[1], pad_x - pad_before[2])

        arr_padded = np.pad(
            arr,
            ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])),
            mode="constant",
            constant_values=pad_value,
        )

        # Create new SITK image
        out = sitk.GetImageFromArray(arr_padded)

        # Manually propagate spacing/origin/direction
        out.SetSpacing(img.GetSpacing())
        out.SetDirection(img.GetDirection())

        # Adjust origin so padded voxels are added symmetrically in physical space
        origin = np.array(img.GetOrigin())
        spacing = np.array(img.GetSpacing())
        direction = np.array(img.GetDirection()).reshape(3, 3)

        shift = np.array(pad_before[::-1]) * spacing  # reverse because numpy array is z,y,x
        shift_physical = direction.dot(shift)

        new_origin = origin - shift_physical
        out.SetOrigin(tuple(new_origin))

        return out, pad_before, arr.shape

    # ---- Read transforms ----
    fdg_to_psma_transform = fdg_to_psma_transform_path
    psma_to_fdg_transform = psma_to_fdg_transform_path

    # ---- Resample ----
    fdg_ct_res = resample_to_reference(fdg_ct, fdg_pet, sitk.sitkLinear, -1024)
    fdg_totseg_res = resample_to_reference(fdg_totseg, fdg_pet, sitk.sitkNearestNeighbor, 0)
    psma_ct_res = resample_to_reference(psma_ct, psma_pet, sitk.sitkLinear, -1024)
    psma_totseg_res = resample_to_reference(psma_totseg, psma_pet, sitk.sitkNearestNeighbor, 0)

    # ---- Choose direction ----
    register_to = register_to.upper()
    if register_to == "PSMA":
        fixed = {"CT": psma_ct_res, "PET": psma_pet, "TOTSEG": psma_totseg_res}
        moving = {"CT": fdg_ct_res, "PET": fdg_pet, "TOTSEG": fdg_totseg_res}
        transform = fdg_to_psma_transform
    elif register_to == "FDG":
        fixed = {"CT": fdg_ct_res, "PET": fdg_pet, "TOTSEG": fdg_totseg_res}
        moving = {"CT": psma_ct_res, "PET": psma_pet, "TOTSEG": psma_totseg_res}
        transform = psma_to_fdg_transform
    else:
        raise ValueError("register_to must be 'PSMA' or 'FDG'")

    # ---- Apply registration ----
    aligned_moving = {}
    for key in moving:
        interp = sitk.sitkLinear if key in ["CT", "PET"] else sitk.sitkNearestNeighbor
        aligned_moving[key] = sitk.Resample(moving[key], fixed["CT"], transform, interp, 0.0, moving[key].GetPixelID())

    if register_to == "PSMA":
        aligned = {"FDG": aligned_moving, "PSMA": fixed}
    else:
        aligned = {"FDG": fixed, "PSMA": aligned_moving}

    # ---- Crop to PET union bbox ----
    fdg_pet_arr = sitk.GetArrayFromImage(aligned["FDG"]["PET"])
    psma_pet_arr = sitk.GetArrayFromImage(aligned["PSMA"]["PET"])
    union_mask = (fdg_pet_arr > fdg_thresh) | (psma_pet_arr > psma_thresh)

    coords = np.array(np.where(union_mask))
    if coords.size:
        min_idx = coords.min(axis=1) - 5
        max_idx = coords.max(axis=1) + 6
        shape = fdg_pet_arr.shape
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, shape)

        size = (max_idx - min_idx).tolist()
        start = min_idx.tolist()
        start_sitk = [start[2], start[1], start[0]]
        size_sitk = [size[2], size[1], size[0]]

        for tracer in ["FDG", "PSMA"]:
            for key in ["CT", "PET", "TOTSEG"]:
                aligned[tracer][key] = sitk.RegionOfInterest(aligned[tracer][key], size_sitk, start_sitk)
    else:
        start, size = [0, 0, 0], fdg_pet_arr.shape

    # ---- Pad after crop ----
    pad_info = {}
    for tracer in ["FDG", "PSMA"]:
        for key in ["CT", "PET", "TOTSEG"]:
            pad_value = -1024 if key == "CT" else 0
            padded, pad_before, unpadded_shape = pad_to_min_size_sitk(aligned[tracer][key], min_size, pad_value)
            aligned[tracer][key] = padded
            pad_info[(tracer, key)] = (pad_before, unpadded_shape)

    # ---- Return ----
    aligned["fdg_to_psma_transform"] = fdg_to_psma_transform
    aligned["psma_to_fdg_transform"] = psma_to_fdg_transform
    aligned["registered_to"] = register_to
    aligned["crop_start"] = start
    aligned["crop_size"] = size
    aligned["pad_info"] = pad_info
    aligned["FDG"]["THRESH"] = fdg_thresh
    aligned["PSMA"]["THRESH"] = psma_thresh
    return aligned


def get_preprocessed_np(aligned):
    def load_modality(aligned, tracer):
        pet = sitk.GetArrayFromImage(aligned[tracer]["PET"]).astype(np.float32)
        totseg = sitk.GetArrayFromImage(aligned[tracer]["TOTSEG"]).astype(np.float32)
        suv_threshold = aligned[tracer]["THRESH"]

        pet = pet / suv_threshold - 1
        totseg = totseg / 117

        return {"PET": pet, "TOTSEG": totseg, "SUV_T": suv_threshold}

    return {
        "FDG_PET": torch.from_numpy(load_modality(aligned, "FDG")["PET"])[None, ...],
        "FDG_TOTSEG": torch.from_numpy(load_modality(aligned, "FDG")["TOTSEG"])[None, ...],
        "FDG_THRESH": aligned["FDG"]["THRESH"],
        "PSMA_PET": torch.from_numpy(load_modality(aligned, "PSMA")["PET"])[None, ...],
        "PSMA_TOTSEG": torch.from_numpy(load_modality(aligned, "PSMA")["TOTSEG"])[None, ...],
        "PSMA_THRESH": aligned["PSMA"]["THRESH"],
        "meta": {
            "crop_start": aligned["crop_start"],
            "crop_size": aligned["crop_size"],
            "pad_info": aligned["pad_info"],
            "orig_shape": sitk.GetArrayFromImage(aligned[aligned["registered_to"]]["PET"]).shape,
            "registered_to": aligned["registered_to"],
        },
    }


def uncrop_and_unpad(pred, crop_start, crop_size, orig_shape, pad_before, unpadded_shape):
    """
    Undo padding and cropping safely:
      - Remove padding
      - Insert back into original uncropped fixed image shape
    """
    # Remove padding
    z0, y0, x0 = pad_before
    z1, y1, x1 = z0 + unpadded_shape[0], y0 + unpadded_shape[1], x0 + unpadded_shape[2]
    pred_unpadded = pred[z0:z1, y0:y1, x0:x1]

    # Place into original uncropped fixed space
    out = np.zeros(orig_shape, dtype=pred.dtype)
    z, y, x = crop_start
    dz, dy, dx = pred_unpadded.shape

    # Clip if out-of-bounds
    z_max = min(z + dz, orig_shape[0])
    y_max = min(y + dy, orig_shape[1])
    x_max = min(x + dx, orig_shape[2])

    dz_clip = z_max - z
    dy_clip = y_max - y
    dx_clip = x_max - x

    out[z : z + dz_clip, y : y + dy_clip, x : x + dx_clip] = pred_unpadded[:dz_clip, :dy_clip, :dx_clip]

    return out


def uncrop_prediction(pred_channel, original, aligned, channel_map={"FDG": 0, "PSMA": 1}):
    """
    Save prediction numpy array [C,Z,Y,X] to MHA,
    undoing crop & pad safely
    """
    reg_to = aligned["registered_to"]
    # ch_idx = channel_map[reg_to]

    # pred_channel = pred_np[ch_idx]

    # Get pad/crop info
    pad_before, unpadded_shape = aligned["pad_info"][(reg_to, "PET")]
    crop_start = aligned["crop_start"]
    crop_size = aligned["crop_size"]
    # Original image shape BEFORE crop/pad
    orig_shape = sitk.GetArrayFromImage(original).shape

    # Undo crop/pad safely
    uncropped = uncrop_and_unpad(pred_channel, crop_start, crop_size, orig_shape, pad_before, unpadded_shape)

    # Save using fixed PET meta
    fixed_pet = original
    sitk_img = sitk.GetImageFromArray(uncropped.astype(np.float32))
    sitk_img.CopyInformation(fixed_pet)
    # sitk.WriteImage(sitk_img, save_path)
    return sitk_img


def expand_contract_label(label, distance=5.0):
    """expand or contract sitk label image by distance indicated.
    negative values will contract, positive values expand.
    returns sitk image of adjusted label"""
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter()  # filter to calculate distance map
    distance_filter.SetUseImageSpacing(True)  # confirm use of image spacing mm
    distance_filter.SquaredDistanceOff()  # default distances computed as d^2, set to linear distance
    dmap = distance_filter.Execute(
        label
    )  # execute on label, returns SITK image in same space with voxels as distance from surface
    dmap_ar = sitk.GetArrayFromImage(dmap)  # array of above SITK image
    new_label_ar = (dmap_ar <= distance).astype("int16")  # binary array of values less than or equal to selected d
    new_label = sitk.GetImageFromArray(new_label_ar)  # create new SITK image and copy spatial information
    new_label.CopyInformation(label)
    return new_label


def refine_my_ttb_label(
    ttb_image,
    pet_image,
    totseg_multilabel,
    expansion_radius_mm=5.0,
    pet_threshold_value=3.0,
    totseg_non_expand_values=[5],
    normal_image=None,
):
    """refine ttb label by expanding and rethresholding the original prediction
    set expansion radius mm to control distance that inferred TTB boundary is initially grown
    set pet_threshold_value to match designated value for ground truth contouring workflow
    (eg PSMA PET SUV=3).
    Includes option to avoid growing the label in certain
    tissue types in the total segmentator label (ex PSMA avoid expanding into liver, could
    include [2,3,5,21] to also avoid kidneys and urinary bladder)
    For other organ values see "total" class map from:
    https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py
    Lastly, possible to include the "normal" tissue inferred label from the baseline example
    algorithm and will similarly avoid expanding into this region"""
    ttb_original_array = sitk.GetArrayFromImage(
        ttb_image
    )  # original TTB array for inpainting back voxels in certain tissues
    ttb_expanded_image = expand_contract_label(
        ttb_image, expansion_radius_mm
    )  # expand inferred label with function above
    ttb_expanded_array = sitk.GetArrayFromImage(ttb_expanded_image)  # convert to numpy array
    pet_threshold_array = (
        sitk.GetArrayFromImage(pet_image) >= pet_threshold_value
    )  # get numpy array of PET image voxels above threshold value
    ttb_rethresholded_array = np.logical_and(
        ttb_expanded_array, pet_threshold_array
    )  # remove expanded TTB voxels below PET threshold

    # loop through total segmentator tissue #s and use original TTB prediction in those labels
    totseg_multilabel = sitk.Resample(
        totseg_multilabel, ttb_image, sitk.TranslationTransform(3), sitk.sitkNearestNeighbor, 0
    )
    totseg_multilabel_array = sitk.GetArrayFromImage(totseg_multilabel)
    for totseg_value in totseg_non_expand_values:
        # paint the original TTB prediction into the totseg tissue regions - probably of most relevance for PSMA liver VOI
        ttb_rethresholded_array[totseg_multilabel_array == totseg_value] = ttb_original_array[
            totseg_multilabel_array == totseg_value
        ]

    # check if inferred normal array is included and if so set TTB voxels to background
    if normal_image is not None:
        normal_array = sitk.GetArrayFromImage(normal_image)
        ttb_rethresholded_array[normal_array > 0] = 0
    ttb_rethresholded_image = sitk.GetImageFromArray(
        ttb_rethresholded_array.astype("int8")
    )  # create output image & copy information
    ttb_rethresholded_image.CopyInformation(ttb_image)
    return ttb_rethresholded_image


def resample_logits_to_other(pred_sitk, reference, transform, interp=sitk.sitkNearestNeighbor):
    # sitk.sitkLinear
    return sitk.Resample(
        pred_sitk,
        reference,  # FDG or PSMA PET (original res)
        transform,
        interp,  # logits: continuous → linear
        0.0,
        sitk.sitkFloat32,
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
