import pandas as pd
import numpy as np
from scipy.ndimage import zoom

from astropy.io import fits

def generate_noise_image(image_size, filters=['VIS', 'NIR_Y', 'NIR_J', 'NIR_H'], reference_file='./out.csv'):
    """
    Generate a synthetic noise image.

    Parameters:
    - image_size: tuple of ints, size of the output image (height, width) in arcseconds
    - filters: list of str, filter names
    - reference_file: str, path to CSV file with reference noise levels
    Returns:
    - noise_image_list: list of np.ndarray, list of noise images for each filter
    """
    df = pd.read_csv(reference_file)
    noise_image_list = []

    rng = np.random.default_rng()

    for f in filters:
        std_dev = df[df['filter'] == f]['global_std'].iloc[rng.integers(0, len(df)//4)]  # Randomly select a noise level from the reference file
        mean = df[df['filter'] == f]['global_mean'].iloc[rng.integers(0, len(df)//4)]

        if f == 'VIS':
            pix_scale = 0.1  # arcsec/pixel
        else:
            pix_scale = 0.3  # arcsec/pixel

        # change image shape from arcseconds to pixels
        image_shape_pixels = tuple(int(dim / pix_scale) for dim in image_size)

        noise_image = rng.normal(loc=mean, scale=std_dev, size=image_shape_pixels)

        # NIR filters use bilinear interpolation to match VIS resolution
        if f in ['NIR_Y', 'NIR_J', 'NIR_H']:

            zoom_factor = pix_scale / 0.1  # VIS pixel scale is 0.1 arcsec/pixel
            noise_image = zoom(noise_image, zoom_factor, order=1, mode='nearest', prefilter=False)  # bilinear interpolation

        noise_image_list.append(noise_image)
    return noise_image_list

if __name__ == "__main__":
    noise_images = generate_noise_image((15, 15))
    
    # to fits files
    filter_names = ['VIS', 'NIR_Y', 'NIR_J', 'NIR_H']
    for i, img in enumerate(noise_images):
        hdu = fits.PrimaryHDU(img)
        hdu.writeto(f'noise_image_{filter_names[i]}.fits', overwrite=True)