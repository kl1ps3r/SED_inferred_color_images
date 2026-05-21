import os

for i in range(1, 5):
    num_images = 5000
    base_class = i
    input_path = './inputs'
    inputs_prefix = f'full_run_rescaling_{base_class}'
    output_path = './output_full_run_rescaling/'

    os.system(f'python new_paramter_sampling.py {num_images} -o {input_path}/{inputs_prefix}')
    os.system(f'python create_images_real.py --base_class {base_class} --input_path {input_path} -o {output_path} --deflector_params {inputs_prefix}_deflector.csv --source_params {inputs_prefix}_source.csv --image_mode Euclid')
    
    
    
    
    
    # If you need Roman comparison runs (no noise, AB units), enable this and provide an appropriate additional-galaxy CSV only if you have one.
    # os.system(f'python create_images_real.py --base_class {base_class} --input_path {input_path} -o {output_path} --deflector_params {inputs_prefix}_deflector.csv --source_params {inputs_prefix}_source.csv --image_mode Roman --comparison --dont_add_noise')

