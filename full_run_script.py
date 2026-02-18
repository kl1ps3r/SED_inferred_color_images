import os
from argparse import ArgumentParser

parser = ArgumentParser(description="Run the full image generation pipeline for multiple base classes.")
parser.add_argument('num_images', type=int, default=250, help='Number of images to generate per base class.')
parser.add_argument('-o', '--output_dir', type=str, default='./output_v2', help='Directory to save the generated images.')
parser.add_argument('--use_existing_inputs', action='store_true', help='Use existing input CSV files instead of generating new ones.', default=False)
args = parser.parse_args()

NUM_IMAGES_PER_BASE = int(args.num_images / 4)

base_classes = [1, 2, 3, 4]

for base_class in base_classes:

    inputs_root = './inputs'
    inputs_prefix = f'large_test_{base_class}_{args.output_dir.split("/")[-1].split("_")[-1]}'
    if not args.use_existing_inputs:
        os.system(f'python galaxy_parameters.py {NUM_IMAGES_PER_BASE} -o {inputs_root +'/'+ inputs_prefix}')
        os.system(f'python generate_additional_galaxies_csv.py --num_images {NUM_IMAGES_PER_BASE} --output_path {inputs_root +'/'+ inputs_prefix}_additional.csv')
    os.system(f'python create_images_real.py --base_class {base_class} --input_path {inputs_root} -o {args.output_dir} --edge_galaxy {inputs_prefix}_additional.csv --deflector_params {inputs_prefix}_deflector.csv --source_params {inputs_prefix}_source.csv')