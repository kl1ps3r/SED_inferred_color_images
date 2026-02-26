import os

num_images = 10
base_class = 4
input_path = './inputs'
inputs_prefix = f'roman_comparison_{base_class}'
output_path = './roman_comparison_output'

os.system(f'python galaxy_parameters.py {num_images} -o {input_path}/{inputs_prefix}')
#os.system(f'python generate_additional_galaxies_csv.py --num_images {num_images} --output_path {input_path}/{inputs_prefix}_additional.csv')
os.system(f'python create_images_real.py --base_class {base_class} --input_path {input_path} -o {output_path} --edge_galaxy {inputs_prefix}_additional.csv --deflector_params {inputs_prefix}_deflector.csv --source_params {inputs_prefix}_source.csv --image_mode Euclid --comparison --dont_add_noise')
os.system(f'python create_images_real.py --base_class {base_class} --input_path {input_path} -o {output_path} --edge_galaxy {inputs_prefix}_additional.csv --deflector_params {inputs_prefix}_deflector.csv --source_params {inputs_prefix}_source.csv --image_mode Roman --comparison --dont_add_noise')