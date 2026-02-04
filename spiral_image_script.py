import argparse
import image_creator

def augment_base():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create spiral galaxy images.')
    parser.add_argument('--base_class', type=int, choices=[1, 2, 3, 4], required=True,
                        help='Base class for the spiral galaxy (1, 2, 3, or 4).')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input directory containing necessary files.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output directory where results will be saved.')
    args = parser.parse_args()

     