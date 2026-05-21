import pandas as pd
import numpy as np
from scipy.ndimage import zoom

from astropy.io import fits

EUCLID_FILTERS = ['VIS', 'NIR_Y', 'NIR_J', 'NIR_H']
ROMAN_FILTERS = ['F106', 'F129', 'F158']

ROMAN_5_SIG_MAGS_SINGLE =   {'F106': 26.1, 'F129': 26.0, 'F158': 26.0}
ROMAN_5_SIG_MAGS_FULL =     {'F106': 26.5, 'F129': 26.4, 'F158': 26.4} # https://roman-docs.stsci.edu/roman-community-defined-surveys/high-latitude-wide-area-survey

def generate_noise_image(image_size, filters=EUCLID_FILTERS, reference_file='./out.csv', roman_mode=None):
    """
    Generate a synthetic noise image.

    Parameters:
    - image_size: tuple of ints, size of the output image (height, width) in arcseconds
    - filters: list of str, filter names
    - reference_file: str, path to CSV file with reference noise levels
    Returns:
    - noise_image_list: list of np.ndarray, list of noise images for each filter
    """
    noise_image_list = []

    rng = np.random.default_rng()

    for f in filters:
        if f in EUCLID_FILTERS:
            print(f)            
            df = pd.read_csv(reference_file)
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
        elif f in ROMAN_FILTERS:
            if roman_mode is None:
                raise ValueError('For a roman filter noise, roman_mode cannot be None.')
            with fits.open(f'./roman_psfs/psf_{f}.fits') as hdul:
                psf = hdul[0].data

            normalized_psf = psf / np.sum(psf)

            s = np.sum(normalized_psf**2)

            if roman_mode[0] == 's':
                flux_1_sig = 10 ** (- 0.4 * (ROMAN_5_SIG_MAGS_SINGLE[f]))   * 3631.0 / 5.0
            elif roman_mode[0] == 'f':
                flux_1_sig = 10 ** (- 0.4 * (ROMAN_5_SIG_MAGS_FULL[f]))     * 3631.0 / 5.0
            else:
                raise ValueError(f'roman_mode must be either full or single, not {roman_mode}.')
            
            # Roman pixel scale
            pix_scale = 0.11  # arcsec/pixel for Roman WFI
            
            # Convert image size from arcseconds to pixels
            image_shape_pixels = tuple(int(dim / pix_scale) for dim in image_size)
            
            noise_image = rng.normal(0, flux_1_sig * np.sqrt(s), image_shape_pixels)

        else:
            raise NotImplementedError(f'Filter {f} is not implemented to generate noise.')

        noise_image_list.append(noise_image)
    return noise_image_list

if __name__ == "__main__":
    # Test with Roman F106 filter, single pass mode
    noise_images = generate_noise_image((15, 15), ['F106'], roman_mode='single')
    print(f"Generated noise image with shape: {noise_images[0].shape}")
    print(f"Noise statistics: mean={noise_images[0].mean():.3e}, std={noise_images[0].std():.3e}")
    
    '''# to fits files
    filter_names = ['F106']
    for i, img in enumerate(noise_images):
        hdu = fits.PrimaryHDU(img)
        hdu.writeto(f'noise_image_{filter_names[i]}.fits', overwrite=True)'''