import image_creator
import generate_noise
import argparse

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

from reproject import reproject_interp

from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.image_util as image_util
from lenstronomy.ImSim.image_model import ImageModel

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.wcs import WCS

import pickle as pkl
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import copy
import traceback
from datetime import datetime

def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
    return data

def augment_light(model_params, augments, source=False):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3, Ob0=0.05)

    angular_diameter_distance = cosmo.angular_diameter_distance(augments[-1])   # in kpc

    model_params_augmented = model_params.copy()

    model_params_augmented['R_sersic'] = ((augments[0] / angular_diameter_distance.to(u.kpc).value) * u.radian).to(u.arcsec).value
    model_params_augmented['n_sersic'] = augments[1]

    if not source:
        model_params_augmented['q'] = augments[2]

        model_params_augmented['phi'] += augments[6]
        model_params_augmented['center_x'] += augments[4]
        model_params_augmented['center_y'] += augments[5]
        return model_params_augmented, augments[3], augments[7]
    else:
        model_params_augmented['phi'] += augments[5]
        model_params_augmented['center_x'] += augments[3]
        model_params_augmented['center_y'] += augments[4]
        return model_params_augmented, augments[2], augments[6]

def create_image_data(kwargs_model, kwargs_params, pixel_scale, num_pixels, exp_time, bkg_rms, psf_fwhm, 
                      lens_redshifts, source_redshifts, cosmo, add_noise=True):
    
    kwargs_data = {
        'background_rms': bkg_rms,  # rms of background noise
        'exposure_time': exp_time,  # exposure time (or a map per pixel)
        'ra_at_xy_0': -num_pixels*pixel_scale/2,  # RA at (0,0) pixel
        'dec_at_xy_0': -num_pixels*pixel_scale/2,  # DEC at (0,0) pixel
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]]),  # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix). Make sure it's units are arcseconds or the angular units you want to model.
        'image_data': np.zeros((num_pixels, num_pixels))
    }

    kwargs_psf = {
        'psf_type': 'GAUSSIAN', 
        'fwhm': psf_fwhm, 
        'pixel_size': pixel_scale, 
        'truncation': 12
        }
    
    kwargs_numerics = {'supersampling_factor': 4, 'supersampling_convolution': False}

    coords = Coordinates(kwargs_data['transform_pix2angle'], kwargs_data['ra_at_xy_0'], kwargs_data['dec_at_xy_0'])
    kwargs_pixel = {'nx': num_pixels, 'ny': num_pixels,  # number of pixels per axis
                'ra_at_xy_0': kwargs_data['ra_at_xy_0'],  # RA at pixel (0,0)
                'dec_at_xy_0': kwargs_data['dec_at_xy_0'],  # DEC at pixel (0,0)
                'transform_pix2angle': kwargs_data['transform_pix2angle']} 

    data_class = ImageData(**kwargs_data)
    psf_class = PSF(**kwargs_psf)

    pixel_grid = PixelGrid(**kwargs_pixel)

    multi_band_pixel_grids = [PixelGrid(**kwargs_pixel)]

    lens_model_class = LensModel(kwargs_model['lens_model_list'], lens_redshift_list=lens_redshifts, cosmo=cosmo)
    source_model_class = LightModel(kwargs_model['source_light_model_list'], source_redshift_list=source_redshifts)
    lens_light_model_class = LightModel(kwargs_model['lens_light_model_list'])

    image_model = ImageModel(data_class, psf_class, lens_model_class=lens_model_class, 
                        source_model_class=source_model_class, lens_light_model_class=lens_light_model_class,
                        kwargs_numerics=kwargs_numerics)
    
    # generate image
    image_model = image_model.image(kwargs_params['kwargs_lens'], kwargs_params['kwargs_source'], kwargs_lens_light=kwargs_params['kwargs_lens_light'], kwargs_ps=None)

    poisson_noise = image_util.add_poisson(image_model, exp_time=exp_time)
    background_noise = data_class.background_rms * np.random.normal(size=image_model.shape)

    if add_noise:
        image_real = image_model + poisson_noise #+ background_noise
    else:
        image_real = image_model

    data_class.update_data(image_real)
    kwargs_data['image_data'] = image_real

    return kwargs_data

def create_edge_galaxy_image_data(edge_light_kwargs_by_band, kwargs_model, cosmo, add_noise=False):
    """
    Generate image data for additional (unlensed) edge galaxies in native filter grids.

    Parameters
    ----------
    edge_light_kwargs_by_band : dict
        Mapping of band name to list of kwargs_lens_light entries (one per galaxy).
        Example keys: 'VIS', 'NIR_Y', 'NIR_J', 'NIR_H'.
    kwargs_model : dict
        Lenstronomy model dictionary containing 'lens_model_list' and 'lens_light_model_list'.
    cosmo : astropy.cosmology
        Cosmology instance used for LensModel.
    add_noise : bool
        Whether to add poisson noise to the edge-galaxy-only images.

    Returns
    -------
    dict
        Mapping of band name to kwargs_data dict with image_data for that band.
    """
    band_meta = {
        'VIS': image_creator.default_Euclid_VIS_image_meta,
        'NIR_Y': image_creator.default_Euclid_NIR_Y_image_meta,
        'NIR_J': image_creator.default_Euclid_NIR_J_image_meta,
        'NIR_H': image_creator.default_Euclid_NIR_H_image_meta,
    }

    edge_images = {}

    for band, edge_kwargs_lens_light in edge_light_kwargs_by_band.items():
        if band not in band_meta:
            raise ValueError(f"Unknown band '{band}'. Expected one of {list(band_meta.keys())}.")

        meta = band_meta[band]
        kwargs_data = {
            'background_rms': meta['background_rms'],
            'exposure_time': meta['exposure_time'],
            'ra_at_xy_0': -meta['num_pix'] * meta['pixel_scale'] / 2,
            'dec_at_xy_0': -meta['num_pix'] * meta['pixel_scale'] / 2,
            'transform_pix2angle': np.array([[meta['pixel_scale'], 0], [0, meta['pixel_scale']]]),
            'image_data': np.zeros((meta['num_pix'], meta['num_pix']))
        }

        kwargs_psf = {
            'psf_type': 'GAUSSIAN',
            'fwhm': meta['psf_fwhm'],
            'pixel_size': meta['pixel_scale'],
            'truncation': 12
        }

        kwargs_numerics = {'supersampling_factor': 4, 'supersampling_convolution': False}

        data_class = ImageData(**kwargs_data)
        psf_class = PSF(**kwargs_psf)

        # Repeat the lens light model for each galaxy in this band
        base_lens_light_model_list = kwargs_model['lens_light_model_list']
        if len(base_lens_light_model_list) != 1:
            raise ValueError("Expected a single lens_light_model_list entry to repeat for edge galaxies.")
        lens_light_model_list = base_lens_light_model_list * len(edge_kwargs_lens_light)
        lens_light_model_class = LightModel(lens_light_model_list)

        lens_model_class = LensModel(kwargs_model['lens_model_list'], lens_redshift_list=[0.0 for _ in kwargs_model['lens_model_list']], cosmo=cosmo)

        image_model = ImageModel(
            data_class,
            psf_class,
            lens_light_model_class=lens_light_model_class,
            lens_model_class=lens_model_class,
            kwargs_numerics=kwargs_numerics
        )

        image_edge = image_model.image(kwargs_lens=None, kwargs_lens_light=edge_kwargs_lens_light)

        if add_noise:
            image_edge = image_edge + image_util.add_poisson(image_edge, exp_time=meta['exposure_time'])

        kwargs_data['image_data'] = image_edge
        edge_images[band] = kwargs_data

    return edge_images

if __name__ == "__main__":
    zeropoints = np.array([25.74, 29.8, 30.0, 29.9])

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3, Ob0=0.05)

    parser = argparse.ArgumentParser(description='Create spiral galaxy images.')
    parser.add_argument('--base_class', type=str, required=True, choices=['1', '2', '3', '4'],
                        help='Base class for the spiral galaxy (1, 2, 3, or 4).')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input directory containing input files.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output directory where results will be saved.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output.')
    args = parser.parse_args('--base_class 4 --input_path . --output_path test_noise'.split())

    # Set up error logging
    error_log_path = f"{args.output_path}/error_log.txt"
    
    with open(f"{args.input_path}/base_class/main_{args.base_class}/kwargs.pkl", 'rb') as f:
        data = pkl.load(f)
        kwargs_params, kwargs_models, multiband_list = data['params'], data['models'], data['multiband_list']

    deflector_file = f"{args.input_path}/test_params_deflector.csv"
    source_file = f"{args.input_path}/test_params_source.csv"

    deflector_augments = pd.read_csv(deflector_file)
    source_augments = pd.read_csv(source_file)
    
    successful_count = 0
    failed_count = 0
    
    for i, (deflector_row, source_row) in enumerate(zip(deflector_augments.itertuples(index=False), source_augments.itertuples(index=False))):
        try:
            if args.verbose:
                print(f"\n{'='*60}")
                print(f"Processing row {i}...")
                print(f"{'='*60}")

            # Augment lens and source light model parameters
            kwargs_params['kwargs_lens_light'][0], ab_mag_deflector, redshift_deflector = augment_light(kwargs_params['kwargs_lens_light'][0], deflector_row)
            kwargs_params['kwargs_source'][0], ab_mag_source, redshift_source = augment_light(kwargs_params['kwargs_source'][0], source_row, source=True)

            redshift_dict = {'lens': redshift_deflector, 'source': redshift_source}

            # load set SED paths
            SED_directory = '/Users/admin/Documents/euclid_color_profiles/swire_library_as_csv/'

            SED_paths = {'source': '/Users/admin/Documents/creating_SED_from_stellar_pop/sed_observed_test.csv', 'lens': SED_directory + 'SA_template_norm.csv'}

            # calculate amplitudes for each filter
            color_maker = image_creator.SED_color_calculator(SED_paths, cosmology=cosmo, target_mags=redshift_dict)

            amps = color_maker.get_amplitudes({'lens': ab_mag_deflector, 'source': ab_mag_source}, kwargs_models, kwargs_params, redshift_dict)
            amplitudes = np.array((amps['lens'], amps['source']))
            scaling = amplitudes[0, 0]
            #print(scaling)
            amplitudes *= 1 / scaling

            # set parameters for each band and update parameters values
            VIS_kwargs_params = copy.deepcopy(kwargs_params)
            NIR_Y_kwargs_params = copy.deepcopy(kwargs_params)
            NIR_J_kwargs_params = copy.deepcopy(kwargs_params)
            NIR_H_kwargs_params = copy.deepcopy(kwargs_params)

            VIS_kwargs_params['kwargs_lens_light'][0]['amp'], VIS_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 0]
            NIR_Y_kwargs_params['kwargs_lens_light'][0]['amp'], NIR_Y_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 1]
            NIR_J_kwargs_params['kwargs_lens_light'][0]['amp'], NIR_J_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 2]
            NIR_H_kwargs_params['kwargs_lens_light'][0]['amp'], NIR_H_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 3]

            ab_mags = np.zeros_like(amplitudes)
            fluxes = np.zeros_like(amplitudes)

            # get the target ab magnitudes
            for i, target in enumerate(['lens', 'source']):
                for band_index in range(amplitudes.shape[1]):
                    ab_mags[i, band_index] = color_maker.get_ab_magnitude(SED=color_maker.SEDs[target], filter_throughput=color_maker.filter_throughputs[band_index])

                    match band_index:
                        case 0:
                            _kwargs_params = VIS_kwargs_params
                            meta=image_creator.default_Euclid_VIS_image_meta
                        case 1:
                            _kwargs_params = NIR_Y_kwargs_params
                            meta=image_creator.default_Euclid_NIR_Y_image_meta
                        case 2:
                            _kwargs_params = NIR_J_kwargs_params
                            meta=image_creator.default_Euclid_NIR_J_image_meta
                        case 3:
                            _kwargs_params = NIR_H_kwargs_params
                            meta=image_creator.default_Euclid_NIR_H_image_meta
                    '''print(f'Computing flux for {target} in band {band_index}')
                    print(_kwargs_params['kwargs_lens_light'][0]['amp'], _kwargs_params['kwargs_source'][0]['amp'])'''
                    fluxes[i, band_index] = color_maker.compute_flux(kwargs_params=_kwargs_params, kwargs_model=kwargs_models, meta=meta, z_source=redshift_dict[target], 
                                                                    z_lens=redshift_dict[target], to_compute=[target], convergence_factor=1e-2)


            ab_mags[0, :] += -ab_mags[0, 0] + ab_mag_deflector  # normalize to vis lens ab mag target
            ab_mags[1, :] += -ab_mags[1, 0] + ab_mag_source  # normalize to vis source ab mag 
            
            
            if False:
                print('Amplitudes:\n', amplitudes)
                print('AB Magnitudes:\n', ab_mags)
                #print(ab_mags[0] - ab_mags[0,0], '\n',ab_mags[1] - ab_mags[1,0])
                print(f'actual mags\n{-2.5 * np.log10(fluxes)}')

            ab_mags_diffs = np.array([ab_mags[0] - ab_mags[0,0], ab_mags[1] - ab_mags[1,0]])
            actual_mags = -2.5 * np.log10(fluxes)# + zeropoints[0]
            actual_mag_diffs = np.array([actual_mags[0] - actual_mags[0,0], actual_mags[1] - actual_mags[1,0]])

            if args.verbose:
                print(f'Actual mags: \n{actual_mags}')
                print(f'Actual mag diffs: \n{actual_mag_diffs}')
                print(f'Target AB mags: \n{ab_mags}')
                print(f'Target AB mag diffs: \n{ab_mags_diffs}')
            #print(actual_mags[:,0] - ab_mags[:,0])
            # rescale amplitues based on this
            amp_scale_factors = 10 ** (-0.4 * (ab_mags_diffs - actual_mag_diffs))

            amplitudes *= amp_scale_factors

            VIS_kwargs_params['kwargs_lens_light'][0]['amp'], VIS_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 0]
            NIR_Y_kwargs_params['kwargs_lens_light'][0]['amp'], NIR_Y_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 1]
            NIR_J_kwargs_params['kwargs_lens_light'][0]['amp'], NIR_J_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 2]
            NIR_H_kwargs_params['kwargs_lens_light'][0]['amp'], NIR_H_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 3]

            test_fluxes = np.zeros_like(amplitudes)
            test_ab_mags = np.zeros_like(amplitudes)
            for i, target in enumerate(['lens', 'source']):
                for band_index in range(amplitudes.shape[1]):
                    match band_index:
                        case 0:
                            _kwargs_params = VIS_kwargs_params
                            meta=image_creator.default_Euclid_VIS_image_meta
                        case 1:
                            _kwargs_params = NIR_Y_kwargs_params
                            meta=image_creator.default_Euclid_NIR_Y_image_meta
                        case 2:
                            _kwargs_params = NIR_J_kwargs_params
                            meta=image_creator.default_Euclid_NIR_J_image_meta
                        case 3:
                            _kwargs_params = NIR_H_kwargs_params
                            meta=image_creator.default_Euclid_NIR_H_image_meta
                    '''print(f'Computing flux for {target} in band {band_index}')
                    print(_kwargs_params['kwargs_lens_light'][0]['amp'], _kwargs_params['kwargs_source'][0]['amp'])'''
                    test_fluxes[i, band_index] = color_maker.compute_flux(kwargs_params=_kwargs_params, kwargs_model=kwargs_models, meta=meta, z_source=redshift_dict[target], 
                                                                                    z_lens=redshift_dict[target], to_compute=[target], convergence_factor=1e-2)
            
            test_ab_mags = -2.5 * np.log10(test_fluxes)
            test_ab_mag_diffs = np.array([test_ab_mags[0] - test_ab_mags[0,0], test_ab_mags[1] - test_ab_mags[1,0]])
            
            if args.verbose:
                print('Post-rescale fluxes:\n', test_fluxes)
                print('Post-rescale AB mags:\n', test_ab_mags)
                print('Post-rescale AB mag diffs:\n', test_ab_mag_diffs)
                print('Expected (target) AB mag diffs:\n', ab_mags_diffs)


            VIS_kwargs_data = create_image_data(kwargs_models, VIS_kwargs_params, image_creator.default_Euclid_VIS_image_meta['pixel_scale'], image_creator.default_Euclid_VIS_image_meta['num_pix'], 
                                                image_creator.default_Euclid_VIS_image_meta['exposure_time'], image_creator.default_Euclid_VIS_image_meta['background_rms'], 
                                                image_creator.default_Euclid_VIS_image_meta['psf_fwhm'], [redshift_deflector], [redshift_source], cosmo)
            NIR_Y_kwargs_data = create_image_data(kwargs_models, NIR_Y_kwargs_params, image_creator.default_Euclid_NIR_Y_image_meta['pixel_scale'], image_creator.default_Euclid_NIR_Y_image_meta['num_pix'], 
                                                image_creator.default_Euclid_NIR_Y_image_meta['exposure_time'], image_creator.default_Euclid_NIR_Y_image_meta['background_rms'], 
                                                image_creator.default_Euclid_NIR_Y_image_meta['psf_fwhm'], [redshift_deflector], [redshift_source], cosmo)
            NIR_J_kwargs_data = create_image_data(kwargs_models, NIR_J_kwargs_params, image_creator.default_Euclid_NIR_J_image_meta['pixel_scale'], image_creator.default_Euclid_NIR_J_image_meta['num_pix'], 
                                                image_creator.default_Euclid_NIR_J_image_meta['exposure_time'], image_creator.default_Euclid_NIR_J_image_meta['background_rms'], 
                                                image_creator.default_Euclid_NIR_J_image_meta['psf_fwhm'], [redshift_deflector], [redshift_source], cosmo)
            NIR_H_kwargs_data = create_image_data(kwargs_models, NIR_H_kwargs_params, image_creator.default_Euclid_NIR_H_image_meta['pixel_scale'], image_creator.default_Euclid_NIR_H_image_meta['num_pix'], 
                                                image_creator.default_Euclid_NIR_H_image_meta['exposure_time'], image_creator.default_Euclid_NIR_H_image_meta['background_rms'], 
                                                image_creator.default_Euclid_NIR_H_image_meta['psf_fwhm'], [redshift_deflector], [redshift_source], cosmo)
            
            # After amplitude rescaling, colors are correct but absolute normalization may be off
            # Compute total target magnitude (lens + source combined) for VIS filter as reference
            total_target_mag_VIS = -2.5 * np.log10(10**(-0.4 * ab_mags[0, 0]) + 10**(-0.4 * ab_mags[1, 0]))
            total_actual_mag_VIS = -2.5 * np.log10(test_fluxes[0, 0] + test_fluxes[1, 0])
            
            # Single global scaling factor to fix absolute normalization (same for all filters)
            global_mag_offset = total_target_mag_VIS - total_actual_mag_VIS
            
            if args.verbose:
                print('Total target VIS mag (lens+source):', total_target_mag_VIS)
                print('Total actual VIS mag (post-rescale):', total_actual_mag_VIS)
                print('Global magnitude offset to apply:', global_mag_offset)
                print('Global flux scale factor:', 10 ** (-0.4 * global_mag_offset))
            
            # Direct approach: compute target pixel values from Euclid formula
            # m_AB = -2.5*log10(sum(pixel_values)) + ZP
            # Therefore: sum(pixel_values) = 10^(0.4 * (ZP - m_AB))
            
            # Compute target total magnitude (lens + source) for each filter
            total_target_mags = -2.5 * np.log10(10**(-0.4 * ab_mags[0, :]) + 10**(-0.4 * ab_mags[1, :]))
            
            # Compute what the sum of pixel values should be for each filter
            target_pixel_sums = 10 ** (0.4 * (zeropoints - total_target_mags))
            
            # Compute actual sums from post-rescale images (in lenstronomy units)
            actual_sums = np.array([np.sum(VIS_kwargs_data['image_data']),
                                    np.sum(NIR_Y_kwargs_data['image_data']),
                                    np.sum(NIR_J_kwargs_data['image_data']),
                                    np.sum(NIR_H_kwargs_data['image_data'])])
            
            # Scale each filter to achieve target pixel sum
            scale_factors = target_pixel_sums / actual_sums
            
            if args.verbose:
                print('Target total mags:', total_target_mags)
                print('Target pixel sums:', target_pixel_sums)
                print('Actual sums (lenstronomy units):', actual_sums)
                print('Scale factors per filter:', scale_factors)
            
            VIS_kwargs_data['image_data'] *= scale_factors[0]
            NIR_Y_kwargs_data['image_data'] *= scale_factors[1]
            NIR_J_kwargs_data['image_data'] *= scale_factors[2]
            NIR_H_kwargs_data['image_data'] *= scale_factors[3]
            
            # Verify: calculate mags after scaling using Euclid formula
            total_pixel_values = np.array([np.sum(VIS_kwargs_data['image_data']), 
                                        np.sum(NIR_Y_kwargs_data['image_data']),
                                        np.sum(NIR_J_kwargs_data['image_data']), 
                                        np.sum(NIR_H_kwargs_data['image_data'])])
            total_mags = -2.5 * np.log10(total_pixel_values) + zeropoints
            if args.verbose:
                print('Verified total mags after scaling:', total_mags)

            if False:
                print('Rescaled Amplitudes:\n', amplitudes)
                recomped_fluxes = np.zeros_like(amplitudes)

                for i, target in enumerate(['lens', 'source']):
                    for band_index in range(amplitudes.shape[1]):

                        match band_index:
                            case 0:
                                _kwargs_params = copy.deepcopy(VIS_kwargs_params)
                                _kwargs_params['kwargs_lens_light'][0]['amp'], _kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 0]
                                meta=image_creator.default_Euclid_VIS_image_meta
                            case 1:
                                _kwargs_params = copy.deepcopy(NIR_Y_kwargs_params)
                                _kwargs_params['kwargs_lens_light'][0]['amp'], _kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 1]
                                meta=image_creator.default_Euclid_NIR_Y_image_meta
                            case 2:
                                _kwargs_params = copy.deepcopy(NIR_J_kwargs_params)
                                _kwargs_params['kwargs_lens_light'][0]['amp'], _kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 2]
                                meta=image_creator.default_Euclid_NIR_J_image_meta
                            case 3:
                                _kwargs_params = copy.deepcopy(NIR_H_kwargs_params)
                                _kwargs_params['kwargs_lens_light'][0]['amp'], _kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 3]
                                meta=image_creator.default_Euclid_NIR_H_image_meta
                        recomped_fluxes[i, band_index] = color_maker.compute_flux(kwargs_params=_kwargs_params, kwargs_model=kwargs_models, meta=
                        image_creator.default_Euclid_VIS_image_meta, z_source=redshift_dict[target], 
                                                                        z_lens=redshift_dict[target], to_compute=[target], convergence_factor=1e-2)
            if False:
                print(f'Recomputed mags after rescaling amplitudes:\n{-2.5 * np.log10(recomped_fluxes)}')

                recomped_mag_diffs = np.array([-2.5 * np.log10(recomped_fluxes[0]) - (-2.5 * np.log10(recomped_fluxes[0,0])), -2.5 * np.log10(recomped_fluxes[1]) - (-2.5 * np.log10(recomped_fluxes[1,0]))])
                print(f'MAG diffs:\nExpected:\n{ab_mags_diffs}\nActual:\n{actual_mag_diffs}\nRecomputed:\n{recomped_mag_diffs}')
            #image_rescale_factors = 10 ** (-0.4 * ())

            # construct WCS objects for each band
            # create header object for VIS and NIR filters

            VIS_coords = Coordinates(VIS_kwargs_data['transform_pix2angle'], VIS_kwargs_data['ra_at_xy_0'], VIS_kwargs_data['dec_at_xy_0'])
            VIS_central_pix = np.array(VIS_kwargs_data['image_data'].shape) // 2

            VIS_WCS = WCS(naxis=2)
            VIS_WCS.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            VIS_WCS.wcs.cunit = ['deg', 'deg']
            VIS_WCS.wcs.crval = VIS_coords.map_pix2coord(*VIS_central_pix)
            VIS_WCS.wcs.crpix = VIS_central_pix
            VIS_WCS.wcs.cdelt = [0.1/3600, 0.1/3600]

            VIS_header = VIS_WCS.to_header()
            VIS_header['naxis'] = 2
            VIS_header['naxis1'], VIS_header['naxis2'] = VIS_kwargs_data['image_data'].shape
            VIS_header['simple'] = True

            NIR_coords = Coordinates(NIR_Y_kwargs_data['transform_pix2angle'], NIR_Y_kwargs_data['ra_at_xy_0'], NIR_Y_kwargs_data['dec_at_xy_0'])
            NIR_central_pix = np.array(NIR_Y_kwargs_data['image_data'].shape) // 2

            NIR_WCS = WCS(naxis=2)
            NIR_WCS.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            NIR_WCS.wcs.cunit = ['deg', 'deg']
            NIR_WCS.wcs.crval = NIR_coords.map_pix2coord(*NIR_central_pix)
            NIR_WCS.wcs.crpix = NIR_central_pix
            NIR_WCS.wcs.cdelt = [0.3/3600, 0.3/3600]

            NIR_header = NIR_WCS.to_header()
            NIR_header['naxis'] = 2
            NIR_header['naxis1'], NIR_header['naxis2'] = NIR_Y_kwargs_data['image_data'].shape
            NIR_header['simple'] = True

            # reproject NIR images to VIS grid
            NIR_Y_reprojected, _ = reproject_interp((NIR_Y_kwargs_data['image_data'], NIR_WCS), VIS_WCS, shape_out=VIS_kwargs_data['image_data'].shape, order='bilinear')
            NIR_J_reprojected, _ = reproject_interp((NIR_J_kwargs_data['image_data'], NIR_WCS), VIS_WCS, shape_out=VIS_kwargs_data['image_data'].shape, order='bilinear')
            NIR_H_reprojected, _ = reproject_interp((NIR_H_kwargs_data['image_data'], NIR_WCS), VIS_WCS, shape_out=VIS_kwargs_data['image_data'].shape, order='bilinear')

            # replace any NaN values that may have been introduced during reprojection with zeros
            NIR_Y_reprojected = np.nan_to_num(NIR_Y_reprojected)
            NIR_J_reprojected = np.nan_to_num(NIR_J_reprojected)
            NIR_H_reprojected = np.nan_to_num(NIR_H_reprojected)

            # generate noise for all bands
            noise_list = generate_noise.generate_noise_image((15, 15))

            VIS_kwargs_data['image_data'] += noise_list[0]
            NIR_Y_reprojected += noise_list[1]
            NIR_J_reprojected += noise_list[2]
            NIR_H_reprojected += noise_list[3]

            # save images and headers
            data_list = [VIS_kwargs_data['image_data'], NIR_Y_reprojected, NIR_J_reprojected, NIR_H_reprojected]

            NIR_Y_header = VIS_header.copy()
            NIR_J_header = VIS_header.copy()
            NIR_H_header = VIS_header.copy()

            NIR_Y_header['FILTER'] = 'NIR_Y'
            NIR_J_header['FILTER'] = 'NIR_J'
            NIR_H_header['FILTER'] = 'NIR_H'

            header_list = [VIS_header, NIR_Y_header, NIR_J_header, NIR_H_header]
            filters = ['VIS', 'NIR_Y', 'NIR_J', 'NIR_H']

            output_suffix = f'euclid_spiral_baseclass_{args.base_class}_img_{i}.fits'

            image_creator.save_to_fits(data_list, header_list, filters, args.output_path + '/' + output_suffix)
            
            successful_count += 1
            if args.verbose:
                print(f"✓ Successfully processed row {i}")
        
        except Exception as e:
            failed_count += 1
            error_msg = f"\n[{datetime.now().isoformat()}] Row {i}: {str(e)}\n{traceback.format_exc()}\n"
            print(f"✗ Error processing row {i}: {str(e)}")
            
            # Log error to file
            with open(error_log_path, 'a') as f:
                f.write(error_msg)
    
    # Summary report
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successful: {successful_count}/{len(deflector_augments)}")
    print(f"Failed: {failed_count}/{len(deflector_augments)}")
    if failed_count > 0:
        print(f"Error log saved to: {error_log_path}")
    print(f"{'='*60}")
