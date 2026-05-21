# Image Generation Guide

This repository contains a small batch wrapper for generating synthetic galaxy images and the main image creation script it calls.

## Main Scripts

- `run_create_images.py` runs a four-class batch generation job.
- `create_images_real.py` is the main image generator.
- `new_paramter_sampling.py` creates the parameter CSV files consumed by the image generator.

## Recommended Workflow

From the repository root, run:

```bash
python run_create_images.py
```

That wrapper will:

1. Loop over `base_class` values `1` through `4`.
2. Generate `5000` samples for each class with `new_paramter_sampling.py`.
3. Call `create_images_real.py` to build the images.
4. Write outputs into `./output_full_run_rescaling/`.

The generated input CSV files are written under `./inputs/` with the prefix `full_run_rescaling_<base_class>`.

## What `run_create_images.py` Does

The script currently uses these values:

- `num_images = 5000`
- `input_path = ./inputs`
- `output_path = ./output_full_run_rescaling/`
- `image_mode = Euclid`

For each class, it runs commands equivalent to:

```bash
python new_paramter_sampling.py 5000 -o ./inputs/full_run_rescaling_1
python create_images_real.py --base_class 1 --input_path ./inputs -o ./output_full_run_rescaling/ --deflector_params full_run_rescaling_1_deflector.csv --source_params full_run_rescaling_1_source.csv --image_mode Euclid
```

## `create_images_real.py` Command-Line Options

The main required options are:

- `--base_class {1,2,3,4}`: Selects the galaxy class.
- `--input_path PATH`: Directory containing the CSV input files.
- `-o, --output_path PATH`: Directory where the output images and logs are written.

Optional inputs:

- `--edge_galaxy FILE`: (Deprecated/optional) Additional CSV file for edge-galaxy parameters. The default batch wrapper no longer uses or generates these files; provide one only for legacy workflows that require edge galaxies.
- `--deflector_params FILE`: Deflector parameter CSV.
- `--source_params FILE`: Source parameter CSV.
- `--image_mode {Euclid,Roman}`: Selects the instrument setup.
- `--comparison`: Keeps the output in AB units for comparison runs.
- `--verbose`: Enables extra logging.

The script creates the output directory if needed and writes an `error_log.txt` file inside it.

## Customizing a Run

If you want different output settings, edit `run_create_images.py`:

- Change `num_images` to generate more or fewer samples.
- Change `output_path` to send images somewhere else.
- Switch `image_mode` between `Euclid` and `Roman`.
- Uncomment the Roman comparison command if you want a Roman-style run.

## Expected Inputs

The batch script expects the generated CSV files to follow this naming pattern:

- `<prefix>_additional.csv`
- `<prefix>_deflector.csv`
- `<prefix>_source.csv`

where `<prefix>` is something like `full_run_rescaling_1`.

## Output

The output directory contains the generated images and an error log for each run. Existing files in the output folder may be overwritten depending on how the script is run.

## Notes

- Run the scripts from the repository root so the relative paths resolve correctly.
- Make sure the `inputs/` and output directories exist or can be created.
- The repository also contains notebooks for interactive testing, but the batch workflow is centered on `run_create_images.py`.