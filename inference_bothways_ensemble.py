"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
from glob import glob
import SimpleITK
import SimpleITK as sitk
import os
import torch
from jpsma.datatools import (
    prepare_case,
    uncrop_prediction,
    get_preprocessed_np,
    resample_logits_to_other,
    refine_my_ttb_label,
)
from jpsma.model import NNUnet

import numpy as np

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

##to test locally (without putting into docker container) uncomment:
##INPUT_PATH = Path("test/input/interf0")
##OUTPUT_PATH = Path("test/output/interf0")

RESOURCE_PATH = Path("resources")

print("inference file location", __file__)


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        (
            "fdg-ct-image",
            "fdg-ct-image-organ-segmentation",
            "fdg-pet",
            "fdg-pet-suv-threshold",
            "psma-ct-image",
            "psma-ct-image-organ-segmentation",
            "psma-pet-ga-68",
            "psma-pet-suv-threshold",
            "psma-to-fdg-registration",
        ): interf0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input
    input_psma_ct_image = load_image_file_as_sitk(
        location=INPUT_PATH / "images/psma-ct",
    )  # equiv to case/PSMA/CT.nii.gz
    input_psma_ct_image_organ_segmentation = load_image_file_as_sitk(
        location=INPUT_PATH / "images/psma-ct-organ-segmentation",
    )  # equiv to case/PSMA/totseg_24.nii.gz
    input_psma_pet_ga_68 = load_image_file_as_sitk(
        location=INPUT_PATH / "images/psma-pet-ga-68",
    )  # equiv to case/PSMA/PET.nii.gz
    input_psma_pet_suv_threshold = load_json_file(
        location=INPUT_PATH / "psma-pet-suv-threshold.json",
    )  # float value (eg 3.0)
    input_fdg_ct_image = load_image_file_as_sitk(
        location=INPUT_PATH / "images/fdg-ct",
    )  # equiv to case/FDG/CT.nii.gz
    input_fdg_ct_image_organ_segmentation = load_image_file_as_sitk(
        location=INPUT_PATH / "images/fdg-ct-organ-segmentation",
    )  # equiv to case/FDG/totseg_24.nii.gz
    input_fdg_pet = load_image_file_as_sitk(
        location=INPUT_PATH / "images/fdg-pet",
    )  # equiv to case/FDG/PET.nii.gz
    input_fdg_pet_suv_threshold = load_json_file(
        location=INPUT_PATH / "fdg-pet-suv-threshold.json",
    )  # float value (eg 2.11)
    input_psma_to_fdg_registration = load_json_file(
        location=INPUT_PATH / "psma-to-fdg-registration.json",
    )  # just 4x4 parameters stored as json dictionary,

    fdg_fixed_transform, psma_fixed_transform = convert_registration_json_to_transforms(input_psma_to_fdg_registration)
    # fdg_fixed_transform equivalent to euler 3d transform in case/FDG/rigid.tfm
    # psma_fixed_transform equivalent to euler 3d transform in case/PSMA/rigid.tfm
    _show_torch_cuda_info()
    # Statements preceding should provide all image/threshold/registration objects as provided in training data.
    # Modify inference scripts below and create output objects output_psma_ttb_label, output_fdg_ttb_label

    # Load models
    model_paths = os.listdir("weights/")
    model_paths = [p for p in model_paths if p.endswith(".ckpt")]
    models = [
        NNUnet.load_from_checkpoint("weights/" + p, map_location="cuda", sw_batch_size=1).eval().cuda()
        for p in model_paths
    ]
    print("X")
    tmp = {}
    for i, tracer in enumerate(["FDG", "PSMA"]):
        input = input_psma_pet_ga_68 if tracer == "PSMA" else input_fdg_pet
        aligned = prepare_case(
            input_fdg_ct_image,
            input_fdg_pet,
            input_fdg_ct_image_organ_segmentation,
            input_psma_ct_image,
            input_psma_pet_ga_68,
            input_psma_ct_image_organ_segmentation,
            psma_fixed_transform,  # transform that maps FDG → PSMA
            fdg_fixed_transform,  # transform that maps PSMA → FDG
            fdg_thresh=input_fdg_pet_suv_threshold,
            psma_thresh=input_psma_pet_suv_threshold,
            register_to=tracer,  # "PSMA" or "FDG"
            min_size=(256, 128, 128),
        )

        batch = get_preprocessed_np(aligned)

        volume = torch.cat(
            [
                batch["FDG_PET"][:, None, ...],
                batch["FDG_TOTSEG"][:, None, ...],
                batch["PSMA_PET"][:, None, ...],
                batch["PSMA_TOTSEG"][:, None, ...],
            ],
            dim=1,
        ).cuda()

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float32):
                logits = []
                for model in models:
                    logit = model.sliding_window_inference(volume)
                    logits.append(logit)

                stacked = torch.stack(logits, dim=0)
                mean_logit = stacked.mean(dim=0)
                preds = torch.ge(torch.sigmoid(mean_logit), 0.5)  # multi-label output in [0,1]
                preds_np = preds.cpu().numpy()

        fdg = preds_np[0, 0]
        psma = preds_np[0, 1]

        uncropped_fdg = uncrop_prediction(fdg, input, aligned)
        uncropped_psma = uncrop_prediction(psma, input, aligned)
        uncropped_tot = uncrop_prediction(
            (batch[f"{tracer}_TOTSEG"][0].cpu().numpy() * 117).astype(np.uint8), input, aligned
        )
        tmp[tracer] = {
            "FDG": sitk.GetArrayFromImage(uncropped_fdg),
            "PSMA": sitk.GetArrayFromImage(uncropped_psma),
            "TOTSEG": uncropped_tot,
        }

        if tracer == "FDG":
            psma_res = sitk.GetArrayFromImage(
                resample_logits_to_other(
                    uncropped_psma, input_psma_pet_ga_68, psma_fixed_transform, interp=sitk.sitkLinear
                )
            )
        else:
            fdg_res = sitk.GetArrayFromImage(
                resample_logits_to_other(uncropped_fdg, input_fdg_pet, fdg_fixed_transform, interp=sitk.sitkLinear)
            )

    fdg_mask = ((fdg_res + tmp["FDG"]["FDG"]) / 2) > 0
    psma_mask = ((psma_res + tmp["PSMA"]["PSMA"]) / 2) > 0

    pred_fdg = fdg_mask * (sitk.GetArrayFromImage(input_fdg_pet) >= input_fdg_pet_suv_threshold)
    pred_psma = psma_mask * (sitk.GetArrayFromImage(input_psma_pet_ga_68) >= input_psma_pet_suv_threshold)

    pred_fdg = sitk.GetImageFromArray(pred_fdg.astype(np.int8))
    # pred_fdg.CopyInformation(input_fdg_pet)
    pred_fdg = refine_my_ttb_label(
        pred_fdg,
        input_fdg_pet,
        tmp["FDG"]["TOTSEG"],
        expansion_radius_mm=5.0,
        pet_threshold_value=input_fdg_pet_suv_threshold,
        totseg_non_expand_values=[2, 3, 5, 21],
    )

    pred_psma = sitk.GetImageFromArray(pred_psma.astype(np.int8))
    # pred_psma.CopyInformation(input_psma_pet_ga_68)
    pred_psma = refine_my_ttb_label(
        pred_psma,
        input_psma_pet_ga_68,
        tmp["PSMA"]["TOTSEG"],
        expansion_radius_mm=5.0,
        pet_threshold_value=input_psma_pet_suv_threshold,
        totseg_non_expand_values=[2, 3, 5, 21],
    )

    write_sitk_image_file(
        location=OUTPUT_PATH / "images/fdg-pet-ttb",
        image=pred_fdg,
    )

    write_sitk_image_file(
        location=OUTPUT_PATH / "images/psma-pet-ttb",
        image=pred_psma,
    )
    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tif")) + glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def load_image_file_as_sitk(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tif")) + glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return result


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tif to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def write_sitk_image_file(location, image):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tif to match the expected output
    suffix = ".mha"

    # image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def convert_registration_json_to_transforms(archive_json):
    # function to convert grand challenge registration archive json to simple itk Euler 3D registration (*.tfm) as provided in training dataset
    # note 4x4 matrix stored on archive json files is unique to the challenge (may not correspond with other documentation) so use
    # this function to convert if required for image processing pipelines
    ##    archive_json=read_json_file(json_fname) #read in dictionary from archive storage json format, stored as PSMA (moving) to FDG (fixed)
    matrix = np.array(archive_json["3d_affine_transform"]).reshape(4, 4)  # get stored parameter matrix
    mx = matrix[:3, :3]  # first 3x3 corresponds to Euler matrix
    center = matrix[3, :3]  # xyz centre encoded in first 3 values of bottom row
    tx = matrix[:3, 3]  # xyz translation encoded in first 3 values of last column
    fdg_fixed_transform = sitk.Euler3DTransform()  # create new SITK Euler 3D Transform
    fdg_fixed_transform.SetMatrix(mx.reshape(9))  # Set matrix values
    fdg_fixed_transform.SetCenter(center)  # Set Center of Rotation
    fdg_fixed_transform.SetTranslation(tx)  # Set Translation
    psma_fixed_transform = fdg_fixed_transform.GetInverse()  # compute inverse for FDG to PSMA rigid registration
    return fdg_fixed_transform, psma_fixed_transform


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
