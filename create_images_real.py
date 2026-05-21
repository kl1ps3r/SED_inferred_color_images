from glob import glob
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
from tqdm import tqdm
import os

from pyphot.svo import get_pyphot_filter

from glob import glob

def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
    return data

def augment_light(model_params, augments, rad, source=False, lens_rad_ratio=None, system_rotation=0.0, old_lens_model=None, new_lens_model=None):

    model_params_augmented = model_params.copy()

    model_params_augmented['R_sersic'] = rad
    model_params_augmented['n_sersic'] = augments[1]

    if not source:
        model_params_augmented['q'] = augments[2]

        model_params_augmented['phi'] += augments[6]
        #model_params_augmented['center_x'] += augments[4]
        #model_params_augmented['center_y'] += augments[5]
        return model_params_augmented, augments[3], augments[7]
    else:
        model_params_augmented['phi'] += augments[5] + system_rotation
        model_params_augmented['center_x'] += augments[3]
        model_params_augmented['center_y'] += augments[4]

        model_params_augmented['center_x'] = model_params_augmented['center_x'] * np.cos(system_rotation) - model_params_augmented['center_y'] * np.sin(system_rotation)
        model_params_augmented['center_y'] = model_params_augmented['center_x'] * np.sin(system_rotation) + model_params_augmented['center_y'] * np.cos(system_rotation)

        if lens_rad_ratio is not None:
            model_params_augmented['center_x'] *= lens_rad_ratio
            model_params_augmented['center_y'] *= lens_rad_ratio

        return model_params_augmented, augments[2], augments[6]

def augment_lens_main(model_params, augments, rad):
    model_params_augmented = model_params.copy()

    #model_params_augmented['center_x'] += augments[3]
    #model_params_augmented['center_y'] += augments[4]
    _, phi = e_to_q_phi(model_params_augmented['e1'], model_params_augmented['e2'])
    e1, e2 = q_phi_to_e(augments[2], phi + augments[6])
    model_params_augmented['e1'] = e1
    model_params_augmented['e2'] = e2

    model_params_augmented['theta_E'] = rad
    return model_params_augmented

def augment_lens_point(model_params, augments, rad):
    model_params_augmented = model_params.copy()

    #model_params_augmented['center_x'] += augments[3]
    #model_params_augmented['center_y'] += augments[4]

    #model_params_augmented['theta_E'] *= 10**np.random.lognormal(0.0, 0.1)  # add some scatter to the subhalo Einstein radius
    return model_params_augmented

def q_phi_to_e(q, phi):
    e = (1 - q) / (1 + q)
    e1 = e * np.cos(2 * phi)
    e2 = e * np.sin(2 * phi)
    return e1, e2

def e_to_q_phi(e1, e2):
    e = np.sqrt(e1**2 + e2**2)
    q = (1 - e) / (1 + e)
    phi = 0.5 * np.arctan2(e2, e1)
    return q, phi

def create_image_data(kwargs_model, kwargs_params, kwargs_data, kwargs_psf, cosmo, lens_redshifts, source_redshifts, add_noise=True, **kwargs):
    
    '''kwargs_data = {
        'background_rms': bkg_rms,  # rms of background noise
        'exposure_time': exp_time,  # exposure time (or a map per pixel)
        'ra_at_xy_0': -num_pixels*pixel_scale/2,  # RA at (0,0) pixel
        'dec_at_xy_0': -num_pixels*pixel_scale/2,  # DEC at (0,0) pixel
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]]),  # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix). Make sure it's units are arcseconds or the angular units you want to model.
        'image_data': np.zeros((num_pixels, num_pixels))
    }
    
    kwargs_psf = kwargs.get('kwargs_psf', {
        'psf_type': 'GAUSSIAN', 
        'fwhm': psf_fwhm, 
        'pixel_size': pixel_scale, 
        'truncation': 12
        })'''
    
    kwargs_numerics = {'supersampling_factor': 4, 'supersampling_convolution': False}

    coords = Coordinates(kwargs_data['transform_pix2angle'], kwargs_data['ra_at_xy_0'], kwargs_data['dec_at_xy_0'])
    kwargs_pixel = {'nx': kwargs_data['image_data'].shape[1], 'ny': kwargs_data['image_data'].shape[0],  # number of pixels per axis
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

    image_model = ImageModel(data_class, psf_class, 
                        lens_model_class=lens_model_class, 
                        source_model_class=source_model_class,
                        lens_light_model_class=lens_light_model_class,
                        kwargs_numerics=kwargs_numerics)
    
    # generate image
    image_model = image_model.image(kwargs_params['kwargs_lens'], kwargs_params['kwargs_source'], kwargs_lens_light=kwargs_params['kwargs_lens_light'], kwargs_ps=None)# ,
    #print(exp_time)
    #poisson_noise = image_util.add_poisson(image_model, exp_time=exp_time)
    #background_noise = data_class.background_rms * np.random.normal(size=image_model.shape)

    '''    poisson_files = glob('./output_test_poisson_noise/*.png')
    next_index = np.max([int(f.split('/')[-1].split('.')[0]) for f in poisson_files]) + 1 if poisson_files else 0
    plt.imshow(poisson_noise, origin='lower', cmap='gray')
    plt.savefig(f'./output_test_poisson_noise/{next_index}.png')'''

    if add_noise:
        image_real = image_model #+ poisson_noise #+ background_noise
    else:
        image_real = image_model

    data_class.update_data(image_real)
    kwargs_data['image_data'] = image_real

    return kwargs_data

def _edge_params_from_row(row, cosmo):

    params = {}

    if hasattr(row, 'redshift'):
        if cosmo is None:
            raise ValueError("cosmo must be provided when edge galaxy rows include redshift")
        angular_diameter_distance = cosmo.angular_diameter_distance(row.redshift)
        params['R_sersic'] = ((row.effective_radius_kpc / angular_diameter_distance.to(u.kpc).value) * u.radian).to(u.arcsec).value
    else:
        params['R_sersic'] = row.effective_radius_kpc

    params['n_sersic'] = row.sersic_index
    params['q'] = row.axis_ratio
    params['phi'] = row.angle
    params['center_x'] = row.pos_x
    params['center_y'] = row.pos_y
    params['amp'] = 1.0

    return params

def load_edge_galaxies_by_image(edge_galaxy_csv_path):
    edge_df = pd.read_csv(edge_galaxy_csv_path)
    if 'image_num' not in edge_df.columns:
        raise ValueError("edge galaxy CSV must include an 'image_num' column")
    return edge_df

def build_edge_galaxy_kwargs_by_band(edge_rows, kwargs_models, color_maker, cosmo, **kwargs):
    bands = ['VIS', 'NIR_Y', 'NIR_J', 'NIR_H']
    band_meta = {
        'VIS': image_creator.default_Euclid_VIS_image_meta,
        'NIR_Y': image_creator.default_Euclid_NIR_Y_image_meta,
        'NIR_J': image_creator.default_Euclid_NIR_J_image_meta,
        'NIR_H': image_creator.default_Euclid_NIR_H_image_meta,
    }

    edge_kwargs_by_band = {band: [] for band in bands}

    for row in edge_rows.itertuples(index=False):

        edge_params = _edge_params_from_row(row, cosmo)
        target_ab_mag = row.AB_magnitude

        amplitudes = np.ones(len(bands))
        fluxes = np.zeros(len(bands))

        for band_index, band in enumerate(bands):
            meta = band_meta[band]
            edge_params_band = {'kwargs_lens_light': [edge_params.copy()]}
            edge_params_band['kwargs_lens_light'][0]['amp'] = amplitudes[band_index]
            edge_params_band['kwargs_lens_light'][0]['center_x'] = 0.0
            edge_params_band['kwargs_lens_light'][0]['center_y'] = 0.0

            fluxes[band_index] = color_maker.compute_flux(
                kwargs_params=edge_params_band,
                kwargs_model=kwargs_models,
                meta=meta,
                z_lens=row.redshift,
                z_source=row.redshift,
                to_compute=['lens'],
                convergence_factor=1e-2,
            )

        # Calculate magnitude differences from unit amplitudes
        actual_mags = -2.5 * np.log10(fluxes)

        # Get SED-based colors (relative magnitudes across bands)
        redshift_SED = color_maker.redshift(color_maker.SEDs['lens'], row.redshift)

        # SED-predicted AB mags for this galaxy (arbitrary normalization)
        sed_ab_mags = np.array([
            color_maker.get_ab_magnitude(
                SED=redshift_SED,
                filter_throughput=color_maker.filter_throughputs[band_index]
            )
            for band_index in range(len(bands))
        ])
        
        # Get color differences from SED (these are physical colors)
        sed_color_diffs = sed_ab_mags - sed_ab_mags[0]
        
        # Get color differences from actual fluxes with unit amplitude
        actual_color_diffs = actual_mags - actual_mags[0]

        if args.verbose:
            print(f"Edge galaxy target_ab_mag (VIS): {target_ab_mag}")
            print(f"Edge galaxy SED color diffs: {sed_color_diffs}")
            print(f"Edge galaxy actual color diffs (unit amp): {actual_color_diffs}")
            print(f"DEBUG_EDGE_GALAXY: target_ab_mag={target_ab_mag}")
            print(f"DEBUG_EDGE_GALAXY: actual_mags(unit amp)={actual_mags}")
            print(f"DEBUG_EDGE_GALAXY: sed_color_diffs={sed_color_diffs}")
            print(f"DEBUG_EDGE_GALAXY: actual_color_diffs={actual_color_diffs}")

        # Scale amplitudes to match SED colors (this gives correct relative fluxes)
        color_scale_factors = 10 ** (-0.4 * (sed_color_diffs - actual_color_diffs))
        amplitudes *= color_scale_factors
        
        # Note: These amplitudes have correct colors but arbitrary absolute normalization
        # The absolute flux calibration (to match target_ab_mag) happens later via per-band
        # scaling using the zeropoint formula: sum(pixels) = 10^(0.4*(ZP - m_AB))
        if args.verbose:
            print(f"DEBUG_EDGE_GALAXY: amplitudes(after scaling)={amplitudes}")
        for band_index, band in enumerate(bands):
            edge_params_band = edge_params.copy()
            edge_params_band['amp'] = amplitudes[band_index]
            edge_kwargs_by_band[band].append(edge_params_band)

    return edge_kwargs_by_band

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

    for band_index, (band, edge_kwargs_lens_light) in enumerate(edge_light_kwargs_by_band.items()):
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

def euclid_image_loop(color_maker, kwargs_models, kwargs_params, cosmo, zeropoints, edge_galaxy_df, error_log_path, deflector_row, source_row, args, filters):

    
    # Capture warnings for this specific row
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        
        try:
            if args.verbose:
                print(f"\n{'='*60}")
                print(f"Processing row {j}...")
                print(f"{'='*60}")

            #_kwargs_params = {'kwargs_lens': [{}, {}], 'kwargs_source': [{}], 'kwargs_lens_light': [{}]}

            angular_diameter_distance_deflector = cosmo.angular_diameter_distance(deflector_row[-1])   # in kpc
            angular_diameter_distance_source = cosmo.angular_diameter_distance(source_row[-1])   # in kpc
            
            deflector_rad = ((deflector_row[0] / angular_diameter_distance_deflector.to(u.kpc).value) * u.radian).to(u.arcsec).value
            source_rad = ((source_row[0] / angular_diameter_distance_source.to(u.kpc).value) * u.radian).to(u.arcsec).value
            
            system_rotation = deflector_row[6]

            #print(f"Deflector rad (arcsec): {deflector_rad}, Source rad (arcsec): {source_rad}")

            #lens_m_to_l = kwargs_params['kwargs_lens'][0]['theta_E'] / kwargs_params['kwargs_lens_light'][0]['R_sersic']

            lens_rad_ratio = deflector_rad / kwargs_params['kwargs_lens_light'][0]['R_sersic']
            
            # Augment lens and source light model parameters
            kwargs_params['kwargs_lens_light'][0], vis_ab_mag_deflector, redshift_deflector = augment_light(kwargs_params['kwargs_lens_light'][0], deflector_row, deflector_rad)

            kwargs_params['kwargs_lens'][0] = augment_lens_main(kwargs_params['kwargs_lens'][0], deflector_row, deflector_rad)
            kwargs_params['kwargs_lens'][1] = augment_lens_point(kwargs_params['kwargs_lens'][1], deflector_row, deflector_rad)

            kwargs_params['kwargs_source'][0], vis_ab_mag_source, redshift_source = augment_light(kwargs_params['kwargs_source'][0], source_row, source_rad, source=True,  
                                                                                                  system_rotation=system_rotation, lens_rad_ratio=lens_rad_ratio
                                                                                                  )


            redshift_dict = {'lens': redshift_deflector, 'source': redshift_source}

            # calculate amplitudes for each filter
            
            redshifted_SEDs = {'lens': color_maker.redshift(color_maker.SEDs['lens'], redshift_deflector),
                                'source': color_maker.redshift(color_maker.SEDs['source'], redshift_source)}

            amps = color_maker.get_amplitudes({'lens': vis_ab_mag_deflector, 'source': vis_ab_mag_source}, kwargs_models, kwargs_params, redshift_dict)
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
            for i, target in (enumerate(['lens', 'source'])):
                for band_index in range(amplitudes.shape[1]):
                    ab_mags[i, band_index] = color_maker.get_ab_magnitude(SED=redshifted_SEDs[target], filter_throughput=color_maker.filter_throughputs[band_index])

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


            ab_mags[0, :] += -ab_mags[0, 0] + vis_ab_mag_deflector  # normalize to vis lens ab mag target
            ab_mags[1, :] += -ab_mags[1, 0] + vis_ab_mag_source  # normalize to vis source ab mag 
            
            
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

            VIS_kwargs_data = create_image_data(kwargs_models, VIS_kwargs_params, color_maker.kwargs_data['VIS'], color_maker.kwargs_psf['VIS'], [redshift_deflector], [redshift_source], cosmo)
            NIR_Y_kwargs_data = create_image_data(kwargs_models, NIR_Y_kwargs_params, color_maker.kwargs_data['NIR_Y'], color_maker.kwargs_psf['NIR_Y'], [redshift_deflector], [redshift_source], cosmo)
            NIR_J_kwargs_data = create_image_data(kwargs_models, NIR_J_kwargs_params, color_maker.kwargs_data['NIR_J'], color_maker.kwargs_psf['NIR_J'], [redshift_deflector], [redshift_source], cosmo)
            NIR_H_kwargs_data = create_image_data(kwargs_models, NIR_H_kwargs_params, color_maker.kwargs_data['NIR_H'], color_maker.kwargs_psf['NIR_H'], [redshift_deflector], [redshift_source], cosmo)

        
            
            # Direct approach: compute target pixel values from Euclid formula
            # m_AB = -2.5*log10(sum(pixel_values)) + ZP
            # Therefore: sum(pixel_values) = 10^(0.4 * (ZP - m_AB))
            
            # Compute target total magnitude (lens + source) for each filter
            total_target_mags = -2.5 * np.log10(10**(-0.4 * ab_mags[0, :]) )# + 10**(-0.4 * ab_mags[1, :])
            
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

                print(f"Scale factors to apply to each filter: {-2.5 * np.log10(scale_factors)}")
                print('Target total mags:', total_target_mags)
                print('Target pixel sums:', target_pixel_sums)
                print('Actual sums (lenstronomy units):', actual_sums)
                print('Scale factors per filter:', scale_factors)
            
            VIS_kwargs_data['image_data'] *= scale_factors[0]
            NIR_Y_kwargs_data['image_data'] *= scale_factors[1]
            NIR_J_kwargs_data['image_data'] *= scale_factors[2]
            NIR_H_kwargs_data['image_data'] *= scale_factors[3]

            if edge_galaxy_df is not None:
                
                edge_rows = edge_galaxy_df[edge_galaxy_df['image_num'] == j]

                # DEBUG_EDGE_GALAXY: row selection summary
                if args.verbose:
                    print(f"DEBUG_EDGE_GALAXY: image_num={j}, edge_rows={len(edge_rows)}")
                if len(edge_rows) > 0:
                    if args.verbose:
                        print("DEBUG_EDGE_GALAXY: sample rows:\n", edge_rows.head(3))
                        # DEBUG_EDGE_GALAXY: VIS half-size check
                        vis_half_size = image_creator.default_Euclid_VIS_image_meta['num_pix'] * image_creator.default_Euclid_VIS_image_meta['pixel_scale'] / 2
                        print(f"DEBUG_EDGE_GALAXY: VIS half-size (arcsec)={vis_half_size}")
                        print("DEBUG_EDGE_GALAXY: max |pos_x|=", edge_rows['pos_x'].abs().max(),
                            "max |pos_y|=", edge_rows['pos_y'].abs().max())

                if args.verbose:
                    print(edge_rows.shape)

                if len(edge_rows) > 0:
                    # DEBUG_EDGE_GALAXY: scale factors applied to main images
                    if args.verbose:
                        print(f"DEBUG_EDGE_GALAXY: scale_factors={scale_factors}")
                    edge_kwargs_by_band = build_edge_galaxy_kwargs_by_band(
                        edge_rows,
                        kwargs_models,
                        color_maker,
                        cosmo
                    )

                    edge_images = create_edge_galaxy_image_data(edge_kwargs_by_band, kwargs_models, cosmo, add_noise=False)
                    # DEBUG_EDGE_GALAXY: edge image stats before adding
                    if args.verbose:
                        if 'VIS' in edge_images:
                            print("DEBUG_EDGE_GALAXY: edge VIS sum/max:", np.sum(edge_images['VIS']['image_data']), np.max(edge_images['VIS']['image_data']))
                        if 'NIR_Y' in edge_images:
                            print("DEBUG_EDGE_GALAXY: edge NIR_Y sum/max:", np.sum(edge_images['NIR_Y']['image_data']), np.max(edge_images['NIR_Y']['image_data']))
                        if 'NIR_J' in edge_images:
                            print("DEBUG_EDGE_GALAXY: edge NIR_J sum/max:", np.sum(edge_images['NIR_J']['image_data']), np.max(edge_images['NIR_J']['image_data']))
                        if 'NIR_H' in edge_images:
                            print("DEBUG_EDGE_GALAXY: edge NIR_H sum/max:", np.sum(edge_images['NIR_H']['image_data']), np.max(edge_images['NIR_H']['image_data']))

                    # Apply the same scale factors as the main galaxies
                    # Edge galaxies use the same SED-based amplitude calculation, so they're in the
                    # same lenstronomy flux units. Using the same scale_factors ensures consistent
                    # flux calibration without overcorrecting for edge galaxies that are partially
                    # outside the frame (which would have artificially low pixel sums).
                    
                    if args.verbose:
                        print("DEBUG_EDGE_GALAXY: using main galaxy scale_factors:", scale_factors)
                        print("DEBUG_EDGE_GALAXY: scale_factors (mag units):", -2.5 * np.log10(scale_factors))

                    # Add scaled edge galaxies to main images
                    if 'VIS' in edge_images:
                        VIS_kwargs_data['image_data'] += edge_images['VIS']['image_data'] * scale_factors[0]
                        if args.verbose:
                            print("DEBUG_EDGE_GALAXY: edge VIS scaled sum/max:", 
                            np.sum(edge_images['VIS']['image_data'] * scale_factors[0]), 
                            np.max(edge_images['VIS']['image_data'] * scale_factors[0]))
                    if 'NIR_Y' in edge_images:
                        NIR_Y_kwargs_data['image_data'] += edge_images['NIR_Y']['image_data'] * scale_factors[1]
                    if 'NIR_J' in edge_images:
                        NIR_J_kwargs_data['image_data'] += edge_images['NIR_J']['image_data'] * scale_factors[2]
                    if 'NIR_H' in edge_images:
                        NIR_H_kwargs_data['image_data'] += edge_images['NIR_H']['image_data'] * scale_factors[3]
            
            # add poisson noise
            for band_index, (band_data, meta) in enumerate(zip([VIS_kwargs_data, NIR_Y_kwargs_data, NIR_J_kwargs_data, NIR_H_kwargs_data],
                                                                [image_creator.default_Euclid_VIS_image_meta, image_creator.default_Euclid_NIR_Y_image_meta, image_creator.default_Euclid_NIR_J_image_meta, image_creator.default_Euclid_NIR_H_image_meta])):
                poisson_noise = image_util.add_poisson(band_data['image_data'], exp_time=meta['exposure_time'])
                if not args.dont_add_noise or not args.comparison:
                    band_data['image_data'] += poisson_noise


            if args.verbose:
                # Verify: calculate mags after scaling using Euclid formula
                total_pixel_values = np.array([np.sum(VIS_kwargs_data['image_data']), 
                                            np.sum(NIR_Y_kwargs_data['image_data']),
                                            np.sum(NIR_J_kwargs_data['image_data']), 
                                            np.sum(NIR_H_kwargs_data['image_data'])])
                total_mags = -2.5 * np.log10(total_pixel_values) + zeropoints
                print('Verified total mags after scaling:', total_mags)

            nir_sums_before_reprojection = np.array([np.sum(NIR_Y_kwargs_data['image_data']), np.sum(NIR_J_kwargs_data['image_data']), np.sum(NIR_H_kwargs_data['image_data'])])

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

            if args.dont_add_noise:
                # generate noise for all bands
                if args.image_mode == 'Euclid':
                    noise_list = generate_noise.generate_noise_image((15, 15), reference_file=f'{args.input_path}/noise_values.csv')
                elif args.image_mode == 'Roman':
                    noise_list = generate_noise.generate_noise_image((15, 15), reference_file=None, filters=['F106', 'F129', 'F158'], roman_mode='single')

                VIS_kwargs_data['image_data'] += noise_list[0]
                NIR_Y_reprojected += noise_list[1]
                NIR_J_reprojected += noise_list[2]
                NIR_H_reprojected += noise_list[3]

            nir_sum_after_reprojection = np.array([np.sum(NIR_Y_reprojected), np.sum(NIR_J_reprojected),  np.sum(NIR_H_reprojected)])

            # save images and headers
            data_list = [VIS_kwargs_data['image_data'], 
                         NIR_Y_reprojected * nir_sums_before_reprojection[0] / nir_sum_after_reprojection[0], 
                         NIR_J_reprojected * nir_sums_before_reprojection[1] / nir_sum_after_reprojection[1], 
                         NIR_H_reprojected * nir_sums_before_reprojection[2] / nir_sum_after_reprojection[2]]

            if args.verbose:
                # Verify: calculate mags after scaling using Euclid formula
                total_pixel_values = np.array([np.sum(data_list[0]), 
                                            np.sum(data_list[1]),
                                            np.sum(data_list[2]), 
                                            np.sum(data_list[3])])
                total_mags = -2.5 * np.log10(total_pixel_values) + zeropoints
                print('Verified total mags after scaling:', total_mags)

            NIR_Y_header = VIS_header.copy()
            NIR_J_header = VIS_header.copy()
            NIR_H_header = VIS_header.copy()

            NIR_Y_header['FILTER'] = 'NIR_Y'
            NIR_J_header['FILTER'] = 'NIR_J'
            NIR_H_header['FILTER'] = 'NIR_H'

            header_list = [VIS_header, NIR_Y_header, NIR_J_header, NIR_H_header]
            filters = ['VIS', 'NIR_Y', 'NIR_J', 'NIR_H']

            output_suffix = f'euclid_spiral_baseclass_{args.base_class}_img_{j}.fits'
            if args.verbose:
                print(f"Saving FITS file for row {j} to {args.output_path}/{output_suffix}...")
            image_creator.save_to_fits(data_list, header_list, filters, args.output_path + '/' + output_suffix)
            
            if args.verbose:
                print(f"✓ Successfully processed row {j}")
                
            # Log any warnings that occurred during processing
            if caught_warnings:
                with open(error_log_path, 'a') as f:
                    f.write(f'\nWarnings for row {j}:\n')
                    for warning in caught_warnings:
                        f.write(f"[{datetime.now().isoformat()}] {warning.category.__name__}: {warning.message}\n")

            return True, kwargs_params  # indicate success


        except Exception as e:
            error_msg = f"\n[{datetime.now().isoformat()}] Row {j}: {str(e)}\n{traceback.format_exc()}\n"
            print(f"✗ Error processing row {j}: {str(e)}")
        
            # Log error to file
            with open(error_log_path, 'a') as f:
                f.write(f'Error processing row {j}:\n')
                f.write(error_msg)

            return False, kwargs_params  # indicate failure

def roman_image_loop(SED_paths, kwargs_models, kwargs_params, cosmo, zeropoints, edge_galaxy_df, error_log_path, deflector_row, source_row, args):
    # Capture warnings for this specific row
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        
        try:
            if args.verbose:
                print(f"\n{'='*60}")
                print(f"Processing row {j}...")
                print(f"{'='*60}")

            # Augment lens and source light model parameters
            kwargs_params['kwargs_lens_light'][0], vis_ab_mag_deflector, redshift_deflector = augment_light(kwargs_params['kwargs_lens_light'][0], deflector_row)
            kwargs_params['kwargs_source'][0], vis_ab_mag_source, redshift_source = augment_light(kwargs_params['kwargs_source'][0], source_row, source=True)

            kwargs_params['kwargs_lens'][0] = augment_lens_main(kwargs_params['kwargs_lens'][0], deflector_row)
            kwargs_params['kwargs_lens'][1] = augment_lens_point(kwargs_params['kwargs_lens'][1], deflector_row)

            redshift_dict = {'lens': redshift_deflector, 'source': redshift_source}

            # calculate amplitudes for each filter
            color_maker = image_creator.SED_color_calculator(SED_paths, cosmology=cosmo, target_mags=redshift_dict, telescope=args.image_mode)
            redshifted_SEDs = {'lens': color_maker.redshift(color_maker.SEDs['lens'], redshift_deflector),
                                'source': color_maker.redshift(color_maker.SEDs['source'], redshift_source)}

            amps = color_maker.get_amplitudes({'lens': vis_ab_mag_deflector, 'source': vis_ab_mag_source}, kwargs_models, kwargs_params, redshift_dict)
            amplitudes = np.array((amps['lens'], amps['source']))
            scaling = amplitudes[0, 0]
            #print(scaling)
            amplitudes *= 1 / scaling

            # set parameters for each band and update parameters values
            VIS_kwargs_params = copy.deepcopy(kwargs_params)
            F106_kwargs_params = copy.deepcopy(kwargs_params)
            F129_kwargs_params = copy.deepcopy(kwargs_params)
            F158_kwargs_params = copy.deepcopy(kwargs_params)

            VIS_kwargs_params['kwargs_lens_light'][0]['amp'], VIS_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 0]
            F106_kwargs_params['kwargs_lens_light'][0]['amp'], F106_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 1]
            F129_kwargs_params['kwargs_lens_light'][0]['amp'], F129_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 2]
            F158_kwargs_params['kwargs_lens_light'][0]['amp'], F158_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 3]

            ab_mags = np.zeros_like(amplitudes)
            fluxes = np.zeros_like(amplitudes)

            # get the target ab magnitudes
            for i, target in (enumerate(['lens', 'source'])):
                for band_index in range(amplitudes.shape[1]):
                    ab_mags[i, band_index] = color_maker.get_ab_magnitude(SED=redshifted_SEDs[target], filter_throughput=color_maker.filter_throughputs[band_index])

                    match band_index:
                        case 0:
                            _kwargs_params = VIS_kwargs_params
                            meta=image_creator.default_Euclid_VIS_image_meta
                        case 1:
                            _kwargs_params = F106_kwargs_params
                            meta=image_creator.roman_image_meta
                        case 2:
                            _kwargs_params = F129_kwargs_params
                            meta=image_creator.roman_image_meta
                        case 3:
                            _kwargs_params = F158_kwargs_params
                            meta=image_creator.roman_image_meta
                    '''print(f'Computing flux for {target} in band {band_index}')
                    print(_kwargs_params['kwargs_lens_light'][0]['amp'], _kwargs_params['kwargs_source'][0]['amp'])'''
                    fluxes[i, band_index] = color_maker.compute_flux(kwargs_params=_kwargs_params, kwargs_model=kwargs_models, meta=meta, z_source=redshift_dict[target], 
                                                                    z_lens=redshift_dict[target], to_compute=[target], convergence_factor=1e-2)


            ab_mags[0, :] += -ab_mags[0, 0] + vis_ab_mag_deflector  # normalize to vis lens ab mag target
            ab_mags[1, :] += -ab_mags[1, 0] + vis_ab_mag_source  # normalize to vis source ab mag 
            
            
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
            F106_kwargs_params['kwargs_lens_light'][0]['amp'], F106_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 1]
            F129_kwargs_params['kwargs_lens_light'][0]['amp'], F129_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 2]
            F158_kwargs_params['kwargs_lens_light'][0]['amp'], F158_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 3]

            VIS_kwargs_data = create_image_data(kwargs_models, VIS_kwargs_params, color_maker.kwargs_data['VIS'], color_maker.kwargs_psf['VIS'], [redshift_deflector], [redshift_source], cosmo)
            F106_kwargs_data = create_image_data(kwargs_models, F106_kwargs_params, color_maker.kwargs_data['F106'], color_maker.kwargs_psf['F106'], [redshift_deflector], [redshift_source], cosmo)
            F129_kwargs_data = create_image_data(kwargs_models, F129_kwargs_params, color_maker.kwargs_data['F129'], color_maker.kwargs_psf['F129'], [redshift_deflector], [redshift_source], cosmo)
            F158_kwargs_data = create_image_data(kwargs_models, F158_kwargs_params, color_maker.kwargs_data['F158'], color_maker.kwargs_psf['F158'], [redshift_deflector], [redshift_source], cosmo)

        
            
            # Direct approach: compute target pixel values from Euclid formula
            # m_AB = -2.5*log10(sum(pixel_values)) + ZP
            # Therefore: sum(pixel_values) = 10^(0.4 * (ZP - m_AB))
            
            # Compute target total magnitude (lens + source) for each filter
            total_target_mags = -2.5 * np.log10(10**(-0.4 * ab_mags[0, :]) + 10**(-0.4 * ab_mags[1, :]))
            
            # Compute what the sum of pixel values should be for each filter
            target_pixel_sums = 10 ** (0.4 * (zeropoints - total_target_mags))
            
            # Compute actual sums from post-rescale images (in lenstronomy units)
            actual_sums = np.array([np.sum(VIS_kwargs_data['image_data']),
                                    np.sum(F106_kwargs_data['image_data']),
                                    np.sum(F129_kwargs_data['image_data']),
                                    np.sum(F158_kwargs_data['image_data'])])
            
            # Scale each filter to achieve target pixel sum
            scale_factors = target_pixel_sums / actual_sums

            
            if args.verbose:

                print(f"Scale factors to apply to each filter: {-2.5 * np.log10(scale_factors)}")
                print('Target total mags:', total_target_mags)
                print('Target pixel sums:', target_pixel_sums)
                print('Actual sums (lenstronomy units):', actual_sums)
                print('Scale factors per filter:', scale_factors)
            
            VIS_kwargs_data['image_data'] *= scale_factors[0]
            F106_kwargs_data['image_data'] *= scale_factors[1]
            F129_kwargs_data['image_data'] *= scale_factors[2]
            F158_kwargs_data['image_data'] *= scale_factors[3]

            if edge_galaxy_df is not None:
                
                edge_rows = edge_galaxy_df[edge_galaxy_df['image_num'] == j]

                # DEBUG_EDGE_GALAXY: row selection summary
                if args.verbose:
                    print(f"DEBUG_EDGE_GALAXY: image_num={j}, edge_rows={len(edge_rows)}")
                if len(edge_rows) > 0:
                    if args.verbose:
                        print("DEBUG_EDGE_GALAXY: sample rows:\n", edge_rows.head(3))
                        # DEBUG_EDGE_GALAXY: VIS half-size check
                        vis_half_size = image_creator.default_Euclid_VIS_image_meta['num_pix'] * image_creator.default_Euclid_VIS_image_meta['pixel_scale'] / 2
                        print(f"DEBUG_EDGE_GALAXY: VIS half-size (arcsec)={vis_half_size}")
                        print("DEBUG_EDGE_GALAXY: max |pos_x|=", edge_rows['pos_x'].abs().max(),
                            "max |pos_y|=", edge_rows['pos_y'].abs().max())

                if args.verbose:
                    print(edge_rows.shape)

                if len(edge_rows) > 0:
                    # DEBUG_EDGE_GALAXY: scale factors applied to main images
                    if args.verbose:
                        print(f"DEBUG_EDGE_GALAXY: scale_factors={scale_factors}")
                    edge_kwargs_by_band = build_edge_galaxy_kwargs_by_band(
                        edge_rows,
                        kwargs_models,
                        color_maker,
                        cosmo
                    )

                    edge_images = create_edge_galaxy_image_data(edge_kwargs_by_band, kwargs_models, cosmo, add_noise=False)
                    # DEBUG_EDGE_GALAXY: edge image stats before adding
                    if args.verbose:
                        if 'VIS' in edge_images:
                            print("DEBUG_EDGE_GALAXY: edge VIS sum/max:", np.sum(edge_images['VIS']['image_data']), np.max(edge_images['VIS']['image_data']))
                        if 'F106' in edge_images:
                            print("DEBUG_EDGE_GALAXY: edge F106 sum/max:", np.sum(edge_images['F106']['image_data']), np.max(edge_images['F106']['image_data']))
                        if 'F129' in edge_images:
                            print("DEBUG_EDGE_GALAXY: edge F129 sum/max:", np.sum(edge_images['F129']['image_data']), np.max(edge_images['F129']['image_data']))
                        if 'F158' in edge_images:
                            print("DEBUG_EDGE_GALAXY: edge F158 sum/max:", np.sum(edge_images['F158']['image_data']), np.max(edge_images['F158']['image_data']))

                    # Apply the same scale factors as the main galaxies
                    # Edge galaxies use the same SED-based amplitude calculation, so they're in the
                    # same lenstronomy flux units. Using the same scale_factors ensures consistent
                    # flux calibration without overcorrecting for edge galaxies that are partially
                    # outside the frame (which would have artificially low pixel sums).
                    
                    if args.verbose:
                        print("DEBUG_EDGE_GALAXY: using main galaxy scale_factors:", scale_factors)
                        print("DEBUG_EDGE_GALAXY: scale_factors (mag units):", -2.5 * np.log10(scale_factors))

                    # Add scaled edge galaxies to main images
                    if 'VIS' in edge_images:
                        VIS_kwargs_data['image_data'] += edge_images['VIS']['image_data'] * scale_factors[0]
                        if args.verbose:
                            print("DEBUG_EDGE_GALAXY: edge VIS scaled sum/max:", 
                            np.sum(edge_images['VIS']['image_data'] * scale_factors[0]), 
                            np.max(edge_images['VIS']['image_data'] * scale_factors[0]))
                    if 'F106' in edge_images:
                        F106_kwargs_data['image_data'] += edge_images['F106']['image_data'] * scale_factors[1]
                    if 'F129' in edge_images:
                        F129_kwargs_data['image_data'] += edge_images['F129']['image_data'] * scale_factors[2]
                    if 'F158' in edge_images:
                        F158_kwargs_data['image_data'] += edge_images['F158']['image_data'] * scale_factors[3]
            
            # add poisson noise
            for band_index, (band_data, meta) in enumerate(zip([VIS_kwargs_data, F106_kwargs_data, F129_kwargs_data, F158_kwargs_data],
                                                                [image_creator.default_Euclid_VIS_image_meta, image_creator.roman_image_meta, image_creator.roman_image_meta, image_creator.roman_image_meta])):
                poisson_noise = image_util.add_poisson(band_data['image_data'], exp_time=meta['exposure_time'])
                #band_data['image_data'] += poisson_noise


            if args.verbose:
                # Verify: calculate mags after scaling using Euclid formula
                total_pixel_values = np.array([np.sum(VIS_kwargs_data['image_data']), 
                                            np.sum(F106_kwargs_data['image_data']),
                                            np.sum(F129_kwargs_data['image_data']), 
                                            np.sum(F158_kwargs_data['image_data'])])
                total_mags = -2.5 * np.log10(total_pixel_values) + zeropoints
                print('Verified total mags after scaling:', total_mags)

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

            roman_coords = Coordinates(F106_kwargs_data['transform_pix2angle'], F106_kwargs_data['ra_at_xy_0'], F106_kwargs_data['dec_at_xy_0'])
            roman_central_pix = np.array(F106_kwargs_data['image_data'].shape) // 2

            roman_WCS = WCS(naxis=2)
            roman_WCS.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            roman_WCS.wcs.cunit = ['deg', 'deg']
            roman_WCS.wcs.crval = roman_coords.map_pix2coord(*roman_central_pix)
            roman_WCS.wcs.crpix = roman_central_pix
            roman_WCS.wcs.cdelt = [0.3/3600, 0.3/3600]

            roman_header = roman_WCS.to_header()
            roman_header['naxis'] = 2
            roman_header['naxis1'], roman_header['naxis2'] = F106_kwargs_data['image_data'].shape
            roman_header['simple'] = True

            # generate noise for all bands
            if args.dont_add_noise:
                if args.image_mode == 'Euclid':
                    noise_list = generate_noise.generate_noise_image((15, 15), reference_file=f'{args.input_path}/noise_values.csv')
                elif args.image_mode == 'Roman':
                    noise_list = generate_noise.generate_noise_image((15, 15), reference_file=f'{args.input_path}/noise_values.csv', filters=['VIS'])
                    noise_list.append(generate_noise.generate_noise_image((15, 15), reference_file=None, filters=['F106', 'F129', 'F158'], roman_mode='single'))

                VIS_kwargs_data['image_data'] += noise_list[0]
                F106_kwargs_data['image_data'] += noise_list[1]
                F129_kwargs_data['image_data'] += noise_list[2]
                F158_kwargs_data['image_data'] += noise_list[3]

            # save images and headers
            data_list = [F106_kwargs_data['image_data'], F129_kwargs_data['image_data'], F158_kwargs_data['image_data']]

            F106_header = roman_header.copy()
            F129_header = roman_header.copy()
            F158_header = roman_header.copy()

            F106_header['FILTER'] = 'F106'
            F129_header['FILTER'] = 'F129'
            F158_header['FILTER'] = 'F158'

            header_list = [F106_header, F129_header, F158_header]
            filters = ['F106', 'F129', 'F158']

            output_suffix = f'roman_spiral_baseclass_{args.base_class}_img_{j}.fits'
            if args.verbose:
                print(f"Saving FITS file for row {j} to {args.output_path}/{output_suffix}...")
            image_creator.save_to_fits(data_list, header_list, filters, args.output_path + '/' + output_suffix)
            
            if args.verbose:
                print(f"✓ Successfully processed row {j}")
                
            # Log any warnings that occurred during processing
            if caught_warnings:
                with open(error_log_path, 'a') as f:
                    f.write(f'\nWarnings for row {j}:\n')
                    for warning in caught_warnings:
                        f.write(f"[{datetime.now().isoformat()}] {warning.category.__name__}: {warning.message}\n")

            return True  # indicate success


        except Exception as e:
            error_msg = f"\n[{datetime.now().isoformat()}] Row {j}: {str(e)}\n{traceback.format_exc()}\n"
            print(f"✗ Error processing row {j}: {str(e)}")
        
            # Log error to file
            with open(error_log_path, 'a') as f:
                f.write(f'Error processing row {j}:\n')
                f.write(error_msg)

            return False  # indicate failure

if __name__ == "__main__":
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3, Ob0=0.05)

    parser = argparse.ArgumentParser(description='Create spiral galaxy images.')
    parser.add_argument('--base_class', type=str, required=True, choices=['1', '2', '3', '4'],
                        help='Base class for the spiral galaxy (1, 2, 3, or 4).')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input directory containing input files.')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='Path to the output directory where results will be saved.')
    # The legacy `--edge_galaxy` option has been removed; edge galaxies are no longer required.
    parser.add_argument('--deflector_params', type=str, default='deflector.csv', help='deflector parameter CSV files (optional).')
    parser.add_argument('--source_params', type=str, default='source.csv', help='source parameter CSV files (optional).')
    parser.add_argument('--image_mode', type=str, choices=['Euclid', 'Roman'], default='Euclid', help='Whether to generate images in Euclid or Roman mode. This affects the noise generation and image properties.')
    parser.add_argument('--comparison', action='store_true', help='Run in comparison mode to make images still in AB units (not scaled by filter zeropoints).')
    parser.add_argument('--dont_add_noise', action='store_false', help='Whether to add noise to the images.', default=False)
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    args = parser.parse_args()

    # create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Set up error logging
    error_log_path = f"{args.output_path}/error_log.txt"
    # Clear existing error log
    with open(error_log_path, 'w') as f:
        f.write(f"Error log for create_images_real.py - {datetime.now()}\n\n")

    # Set up warning handler to log warnings to file
    import warnings
    import logging
    
    # Configure logging to capture warnings
    logging.basicConfig(
        filename=error_log_path,
        level=logging.WARNING,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Redirect warnings to logging system
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')

    # set SED paths
            
    SED_paths = {'source': f'{args.input_path}/SEDs/sed_rest_frame_star_forming.csv', 'lens': f'{args.input_path}/SEDs/Ell13_template_norm.csv'}
    
    with open(f"{args.input_path}/base_class/main_{args.base_class}/kwargs.pkl", 'rb') as f:
        data = pkl.load(f)
        kwargs_params, kwargs_models, multiband_list = data['params'], data['models'], data['multiband_list']

    filters = [get_pyphot_filter(filter) for filter in ['Euclid/VIS.vis', 'Euclid/NISP.Y', 'Euclid/NISP.J', 'Euclid/NISP.H']]

    deflector_file = f"{args.input_path}/{args.deflector_params}"
    source_file = f"{args.input_path}/{args.source_params}"

    deflector_augments = pd.read_csv(deflector_file)
    source_augments = pd.read_csv(source_file)

    # Additional/edge-galaxy CSVs are deprecated; ensure no dependency on them.
    edge_galaxy_df = None

    if args.comparison:
        zeropoints = np.array([0, 0, 0, 0])  # no scaling for comparison mode
    else:
        zeropoints = np.array([25.74, 29.8, 30.0, 29.9])

    color_maker = image_creator.SED_color_calculator(SED_paths, cosmology=cosmo, telescope=args.image_mode, filter_throughputs=filters)
    
    successful_count = 0
    failed_count = 0

    # temp
    source_poses = []
    deflector_radii = []

    for j, (deflector_row, source_row) in enumerate(tqdm(zip(deflector_augments.itertuples(index=False), source_augments.itertuples(index=False)), total=len(deflector_augments))):
        if args.image_mode == 'Euclid':
            success, out_kwargs_params = euclid_image_loop(color_maker, kwargs_models, kwargs_params, cosmo, zeropoints, edge_galaxy_df, error_log_path, deflector_row, source_row, args, filters)

            source_poses.append((out_kwargs_params['kwargs_source'][0]['center_x'], out_kwargs_params['kwargs_source'][0]['center_y']))
            deflector_radii.append(kwargs_params['kwargs_lens'][0]['theta_E'])

        elif args.image_mode == 'Roman':
            success = roman_image_loop(SED_paths, kwargs_models, kwargs_params, cosmo, zeropoints, edge_galaxy_df, error_log_path, deflector_row, source_row, args, filters)

        if success:
            successful_count += 1
        else:
            failed_count += 1      
        
    # Summary report
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successful: {successful_count}/{len(deflector_augments)}")
    print(f"Failed: {failed_count}/{len(deflector_augments)}")
    if failed_count > 0:
        print(f"Error log saved to: {error_log_path}")
    print(f"{'='*60}")

    source_poses = np.array(source_poses)
    deflector_radii = np.array(deflector_radii)

    '''plt.plot(np.sqrt(source_poses[:,0] ** 2 + source_poses[:,1] ** 2), deflector_radii, 'o')
    plt.xlabel('Source Position')
    plt.ylabel('Deflector Radius')
    plt.title('Source Position vs Deflector Radius')
    plt.show()
'''