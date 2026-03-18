import os
import argparse
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def run_inference(input_path, output_dir, weights_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Convert input to nii.gz if needed
    if not input_path.endswith(".nii.gz"):
        img = sitk.ReadImage(input_path)
        converted = input_path.replace(input_path.split(".")[-1], "").rstrip(".") + "_0000.nii.gz"
        sitk.WriteImage(img, converted)
        input_path = converted

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=True
    )

    predictor.initialize_from_trained_model_folder(
        weights_dir,
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth"
    )

    predictor.predict_from_files(
        os.path.dirname(input_path),
        output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1
    )
    print(f"Inference complete. Output saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input scan")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--weights", required=True, help="Path to nnUNet model folder")
    args = parser.parse_args()
    run_inference(args.input, args.output, args.weights)
