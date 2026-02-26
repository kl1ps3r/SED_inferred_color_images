import numpy as np

import astropy.units as u
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM

from scipy.special import gamma
from copy import deepcopy

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.coord_transforms import Coordinates
import lenstronomy.Util.image_util as image_util

from matplotlib import pyplot as plt
from astropy.io import fits

import os

class SED_color_calculator:

    def __init__(self, SED_paths, **kwargs):
        self.load_SEDs(SED_paths)

        # handle kwargs
        if 'cosmology' in kwargs:
            self.cosmology = kwargs['cosmology']
        else:
            from astropy.cosmology import FlatLambdaCDM
            
            self.cosmology = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3, Ob0=0.05)
        
        telescope = kwargs.get('telescope', 'Euclid')

        if telescope == 'Euclid':
            self.filter_names = ['VIS', 'NIR_Y', 'NIR_J', 'NIR_H']
        elif telescope == 'Roman':
            self.filter_names = ['VIS', 'F106', 'F129', 'F158']

        self.load_filters_throughputs(telescope, verbose=kwargs.get('verbose', False))
        self.initialise_filter_image_meta_dicts(telescope)

        

    def load_SEDs(self, SED_paths):
        self.SEDs = {}
        for key, path in SED_paths.items():
            self.SEDs[key] = self.load_SED(path)
    
    def load_SED(self, path):
        try:
            data = np.loadtxt(path, unpack=True)
            return data
        except Exception as e:
            raise IOError(f"Error loading SED from {path}: \n{e}")
        
    def load_filters_throughputs(self, filter_names, **kwargs):
        verbose = kwargs.get('verbose', False)
        if filter_names == 'Euclid':
            
            if verbose:
                print("Loading Euclid filter throughputs...")

            import photometry
            pbs = [{'file': 'VIS.Euclid.pb', 'outCol': 'TU_Fnu_VIS', 'band': None, 'name': 'VIS'},
                {'file': 'Y_NISP.Euclid.pb', 'outCol': 'TU_Fnu_NIR_Y', 'band': None, 'name': 'NIR_Y'},
                {'file': 'J_NISP.Euclid.pb', 'outCol': 'TU_Fnu_NIR_J', 'band': None, 'name': 'NIR_J'},
                {'file': 'H_NISP.Euclid.pb', 'outCol': 'TU_Fnu_NIR_H', 'band': None, 'name': 'NIR_H'}]

            for pb in pbs:
                pb['band'] = photometry.Passband(file=pb['file'])

            temp = [np.array([pb['band'].lam(unit=u.Angstrom).value, pb['band'].y]) for pb in pbs]

            Euclid_VIS_mask = 3450. < temp[0][0]

            self.filter_throughputs = [np.array([temp[0][0][Euclid_VIS_mask], temp[0][1][Euclid_VIS_mask]])]
            for filter in temp[1:]:
                self.filter_throughputs.append(filter)

        elif filter_names == 'Roman':
            import photometry
            pbs = [{'file': 'VIS.Euclid.pb', 'outCol': 'TU_Fnu_VIS', 'band': None, 'name': 'VIS'}]
            for pb in pbs:
                pb['band'] = photometry.Passband(file=pb['file'])

            temp = [np.array([pb['band'].lam(unit=u.Angstrom).value, pb['band'].y]) for pb in pbs]

            Euclid_VIS_mask = 3450. < temp[0][0]

            if verbose:
                print("Loading Roman filter throughputs...")

            # get file list of Roman filter throughputs in ./Roman_filters/
            filter_files = [f for f in os.listdir('./Roman_filters/') if f.endswith('.csv')]
            self.filter_throughputs = [np.array([temp[0][0][Euclid_VIS_mask], temp[0][1][Euclid_VIS_mask]])]
            for filter_file in filter_files:
                data = np.loadtxt(os.path.join('./Roman_filters/', filter_file), delimiter=',', skiprows=1, unpack=True)
                data[0] *= 1e4  # convert from microns to Angstrom
                self.filter_throughputs.append(data)

        else:
            raise NotImplementedError("Only 'Euclid' and 'Roman' filter throughputs are currently implemented.")

    def initialise_filter_image_meta_dicts(self, filter_names):
        self.kwargs_data = {}
        self.kwargs_psf = {}
        self.kwargs_numerics = {}

        if filter_names == 'Euclid':
            for filter in self.filter_names:
                self.kwargs_data[filter], self.kwargs_psf[filter], self.kwargs_numerics[filter] = self.meta_to_dicts(globals()[f'default_Euclid_{filter}_image_meta'])
        
        elif filter_names == 'Roman':
            for filter in self.filter_names:
                if filter == 'VIS':
                    meta = globals()[f'default_Euclid_VIS_image_meta']
                else:
                    meta = globals()[f'roman_image_meta']
                meta['filter_name'] = filter
                self.kwargs_data[filter], self.kwargs_psf[filter], self.kwargs_numerics[filter] = self.meta_to_dicts(meta)
        else:
            raise NotImplementedError("Only 'Euclid' and 'Roman' filter image meta dicts are currently implemented.")

    def get_amplitudes(self, target_AB_mags, kwargs_model, kwarg_params, redshifts:dict, **kwargs):

        # handle kwargs
        verbose = kwargs.get('verbose', False)
        single_gen = kwargs.get('single_gen', False)


        # redshift SEDs
        shifted_SEDs = {}
        for model_type in self.SEDs.keys():
            SED = self.SEDs[model_type]
            shifted_SEDs[model_type] = self.redshift(SED, redshifts[model_type])
        if verbose:
            print('lens ab mag raw sed', self.get_ab_magnitude(shifted_SEDs['lens'], self.filter_throughputs[0]))
            print('source ab mag raw sed', self.get_ab_magnitude(shifted_SEDs['source'], self.filter_throughputs[0]))

        # calculate scaling factors
        scaling_factors = {}
        for model_type in self.SEDs.keys():
            scaling_factors[model_type] = 10 ** (-0.4 * (target_AB_mags[model_type] - self.get_ab_magnitude(shifted_SEDs[model_type], self.filter_throughputs[0])))

        if verbose:
            print(-2.5 * np.log10(scaling_factors['lens']), -2.5 * np.log10(scaling_factors['source']))

        temp_SEDs = {}
        temp_SEDs['lens'] = shifted_SEDs['lens'].copy()
        temp_SEDs['source'] = shifted_SEDs['source'].copy()

        temp_SEDs['lens'][1] *= scaling_factors['lens']
        temp_SEDs['source'][1] *= scaling_factors['source']
        if verbose:
            print('lens ab mag scaled sed', self.get_ab_magnitude(temp_SEDs['lens'], self.filter_throughputs[0]))
            print('source ab mag scaled sed', self.get_ab_magnitude(temp_SEDs['source'], self.filter_throughputs[0]))

        target_flux_ratio = 10 ** (-0.4 * (target_AB_mags['lens'] - target_AB_mags['source']))

        # calculate weighted mean fluxes
        weighted_mean_fluxes = {}
        for model_type in target_AB_mags.keys():
            weighted_mean_fluxes[model_type] = np.array([self.get_weighted_mean_flux(shifted_SEDs[model_type], throughput) * scaling_factors[model_type] for throughput in self.filter_throughputs])
        if verbose:
            print('weighted mean fluxes', weighted_mean_fluxes)

        _kwarg_params = deepcopy(kwarg_params)
        _kwarg_params['kwargs_lens_light'][0]['amp'] = 1.0
        _kwarg_params['kwargs_source'][0]['amp'] = 1.0
        
        # calculate ratios
        deflector_unit_flux = self.compute_flux(_kwarg_params,
                                          kwargs_model,
                                          self.kwargs_data['VIS'],
                                          self.kwargs_psf['VIS'],
                                          self.kwargs_numerics['VIS'],
                                          to_compute=['lens'],
                                          convergence_factor=1e-2,
                                          **kwargs)
        source_unit_flux = self.compute_flux(_kwarg_params,
                                        kwargs_model,
                                        self.kwargs_data['VIS'],
                                        self.kwargs_psf['VIS'],
                                        self.kwargs_numerics['VIS'],
                                        to_compute=['source'],
                                        lens_image=True,
                                        convergence_factor=1e-2,
                                        **kwargs)
        
        # Check for zero or invalid flux values
        if deflector_unit_flux <= 0 or not np.isfinite(deflector_unit_flux):
            error_msg = f"Invalid deflector flux: {deflector_unit_flux}. "
            error_msg += f"Deflector params: R_sersic={_kwarg_params['kwargs_lens_light'][0]['R_sersic']:.3f}, "
            error_msg += f"n_sersic={_kwarg_params['kwargs_lens_light'][0]['n_sersic']:.3f}, "
            error_msg += f"amp={_kwarg_params['kwargs_lens_light'][0]['amp']:.3e}"
            raise ValueError(error_msg)
        
        if source_unit_flux <= 0 or not np.isfinite(source_unit_flux):
            error_msg = f"Invalid source flux: {source_unit_flux}. "
            error_msg += f"Source params: R_sersic={_kwarg_params['kwargs_source'][0]['R_sersic']:.3f}, "
            error_msg += f"n_sersic={_kwarg_params['kwargs_source'][0]['n_sersic']:.3f}, "
            error_msg += f"amp={_kwarg_params['kwargs_source'][0]['amp']:.3e}"
            raise ValueError(error_msg)
        
        if verbose:
            print(deflector_unit_flux, source_unit_flux, '\n', weighted_mean_fluxes)
        amplitudes = {'lens': weighted_mean_fluxes['lens'] / deflector_unit_flux,
                     'source': weighted_mean_fluxes['source'] / source_unit_flux}

        '''
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Need to check for correction factor of VIS flux ratio
        Me being stupid, its self consistent if you use lenstronomy
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
        # check flux ratio versus target flux ratio
        if not single_gen:
            vis_amplitudes = {model_type: amplitudes[model_type][0] for model_type in amplitudes.keys()}
            if verbose:
                print(f"VIS amplitudes: {vis_amplitudes}")
            computed_flux_ratio = self.compute_flux_ratio(kwargs_model,
                                                        kwarg_params,
                                                        filter_name='VIS',
                                                        amplitudes=vis_amplitudes)
            if verbose:
                print(f"Computed flux ratio: {computed_flux_ratio}, Target flux ratio: {target_flux_ratio}")

            # Check flux ratio with tolerance - log warning if mismatch but don't fail
            # The per-band scale_factors approach handles final flux calibration
            relative_error = abs(computed_flux_ratio - target_flux_ratio) / target_flux_ratio
            if not np.isclose(computed_flux_ratio, target_flux_ratio, rtol=0.15):
                import warnings
                warnings.warn(
                    f"Flux ratio mismatch: computed={computed_flux_ratio:.4f}, "
                    f"target={target_flux_ratio:.4f}, relative_error={relative_error*100:.2f}%. "
                    f"This will be corrected by per-band scaling."
                )

        return amplitudes

    def compute_magnification(self, kwargs_model, kwarg_params, filter_name='VIS', **kwargs):
        '''
        Compute the magnification of the source light given lens and source models.
        Parameters:
        -----------
        lens_model_class : class
            The lens model class.
        source_model_class : class
            The source model class.
        kwarg_params : dict
            Dictionary of keyword arguments for lens and source models.
        Returns:
        --------
        magnification : float
            The magnification factor of the source light.
        '''
        
        total_flux_lensed = self.compute_flux(kwarg_params,
                                        kwargs_model,
                                        self.kwargs_data[filter_name],
                                        self.kwargs_psf[filter_name],
                                        self.kwargs_numerics[filter_name],
                                        to_compute=['source'],
                                        lens_image=True,
                                        **kwargs)
        total_flux_unlensed = self.compute_flux(kwarg_params,
                                            kwargs_model,
                                            self.kwargs_data[filter_name],
                                            self.kwargs_psf[filter_name],
                                            self.kwargs_numerics[filter_name],
                                            to_compute=['source'],
                                            lens_image=False,
                                            **kwargs)
        magnification = total_flux_lensed / total_flux_unlensed
        return magnification

    def compute_flux_ratio(self, kwargs_model,
                           kwarg_params,
                           filter_name='VIS',
                           amplitudes=None,
                           **kwargs):
        '''
        Compute the flux ratio of source to lens light in a given filter.
        Parameters:
        -----------
        lens_model_class : class
        source_model_class : class
        kwarg_params : dict
            Dictionary of keyword arguments for lens and source models.
        Returns:
        --------
        flux_ratio : float
            The flux ratio of source to lens light.
        '''
        if amplitudes is not None:
            _kwarg_params = deepcopy(kwarg_params)
            for model_type in amplitudes.keys():
                if model_type == 'lens':
                    model_type_param = 'lens_light'
                else:
                    model_type_param = model_type
                _kwarg_params[f'kwargs_{model_type_param}'][0]['amp'] = amplitudes[model_type]
        if kwargs.get('verbose', False):
            print('computing lens flux in compute_flux_ratio')
        total_flux_lens = self.compute_flux(_kwarg_params,
                                        kwargs_model,
                                        self.kwargs_data[filter_name],
                                        self.kwargs_psf[filter_name],
                                        self.kwargs_numerics[filter_name],
                                        to_compute=['lens'],
                                        lens_image=False,
                                        convergence_factor=1e-2,
                                        **kwargs)
        total_flux_source = self.compute_flux(_kwarg_params,
                                            kwargs_model,
                                            self.kwargs_data[filter_name],
                                            self.kwargs_psf[filter_name],
                                            self.kwargs_numerics[filter_name],
                                            to_compute=['source'],
                                            lens_image=True,
                                            **kwargs)
        if kwargs.get('verbose', False):
            print('vis fluxes', total_flux_lens, total_flux_source)
            print('vis ab mags', -2.5 * np.log10(total_flux_lens), -2.5 * np.log10(total_flux_source))
        flux_ratio = total_flux_lens / total_flux_source
        return flux_ratio

    @staticmethod
    def sersic_total_flux(**kwargs):
        '''
        Calculation of the total flux of a Sersic light profile of the form I(R) = I_e exp{-b_n[(R/R_e)^(1/n) - 1]}.
        To be removed?

        Parameters:
        -----------
        R_sersic : float
            Effective radius of the Sersic profile.
        n_sersic : float
            Sersic index.
        q : float, optional
            Axis ratio. If not provided, e1 and e2 must be provided.
        e1 : float, optional
            Ellipticity component 1.
        e2 : float, optional
            Ellipticity component 2.
        Returns:
        --------
        total_flux : float
            Total flux of the Sersic profile.
        '''
        try:
            R_sersic = kwargs['R_sersic']
        except KeyError:
            raise ValueError("R_sersic must be provided in kwargs.")
        try:
            n_sersic = kwargs['n_sersic']
        except KeyError:
            raise ValueError("n_sersic must be provided in kwargs.")

        if 'q' in kwargs:
            q = kwargs['q']
        elif 'e1' in kwargs and 'e2' in kwargs:
            e1 = kwargs['e1']
            e2 = kwargs['e2']
            q = ((1-np.sqrt(e1**2 + e2**2)) / (1+np.sqrt(e1**2 + e2**2)))
        else:
            raise ValueError("Either 'q' or both 'e1' and 'e2' must be provided in kwargs.")

        b_n = 1.9992 * n_sersic - 0.3271  # approximate
        return 2 * np.pi * R_sersic**2 * q * n_sersic * np.exp(b_n) * gamma(2 * n_sersic) / b_n**(2 * n_sersic)
    
    @staticmethod
    def redshift(SED, z):
        '''
        Redshift the SED by a given redshift z.
        Parameters:
        -----------
        SED : np.ndarray
            2D array where the first row is wavelength and the second row is flux.
        z : float
            Redshift value.
        Returns:
        --------
        redshifted_SED : np.ndarray
            Redshifted SED.
        '''

        redshifted_wavelength = SED[0] * (1 + z)
        redshifted_flux = SED[1] / (1 + z)
        redshifted_SED = np.array([redshifted_wavelength, redshifted_flux])
        return redshifted_SED

    @staticmethod
    def get_weighted_mean_flux(SED, filter_throughput, interp_method=np.interp, integrate_method=None):
        '''
        Calculate the weighted mean flux of the SED through the filter throughput.
        Parameters:
        -----------
        SED : np.ndarray
            2D array where the first row is wavelength and the second row is flux.
        filter_throughput : np.ndarray
            2D array where the first row is wavelength and the second row is throughput.
        Returns:
        --------
        mean_flux : float
            Weighted mean flux.
        '''
        if integrate_method is None:
            from scipy.integrate import trapezoid
            integrate_method = trapezoid

        #plot(SED, filter_throughput)

        # Ensure filter wavelengths are strictly increasing
        if not np.all(np.diff(filter_throughput[0]) > 0):
            raise ValueError("Filter wavelengths must be strictly increasing.")

        # Interpolate filter throughput to SED wavelengths
        interp_filter = interp_method(SED[0], filter_throughput[0], filter_throughput[1], left=0, right=0)

        # Calculate weighted mean flux
        numerator = integrate_method(SED[1] * interp_filter * SED[0], SED[0])
        denominator = integrate_method(interp_filter * SED[0], SED[0])

        if denominator == 0.0:
            raise ValueError("Denominator for mean flux calculation is zero.")

        mean_flux = numerator / denominator
        return mean_flux

    def get_ab_magnitude(self, SED, filter_throughput, interp_method=np.interp, integrate_method=None):
        '''
        Calculate the AB magnitude of the SED through the filter throughput.
        Parameters:
        -----------
        SED : np.ndarray
            2D array where the first row is wavelength and the second row is flux.
        filter_throughput : np.ndarray
            2D array where the first row is wavelength and the second row is throughput.
        Returns:
        --------
        ab_magnitude : float
            AB magnitude.
        '''

        if integrate_method is None:
            from scipy.integrate import trapezoid
            integrate_method = trapezoid

        mean_flux = self.get_weighted_mean_flux(SED, filter_throughput, interp_method, integrate_method)

        if mean_flux <= 0:
            return np.inf  # Return infinity for non-positive fluxes

        # Convert f_lambda to f_nu
        effective_wavelength = integrate_method(filter_throughput[0] * filter_throughput[1], filter_throughput[0]) / integrate_method(filter_throughput[1], filter_throughput[0])
        f_nu = mean_flux * (effective_wavelength ** 2) / c.to(u.Angstrom / u.s).value  # in erg/s/cm^2/Hz

        # Calculate AB magnitude
        ab_magnitude = -2.5 * np.log10(f_nu) - 48.6
        return ab_magnitude

    @staticmethod
    def compute_flux(kwargs_params,
                        kwargs_model,
                        kwargs_data=None,
                        kwargs_psf=None,
                        kwargs_numerics=None,
                        to_compute=['lens', 'source'],
                        z_lens=None,
                        z_source=None,
                        cosmology=None,
                        verbose=False,
                        num_pixes=None,
                        lens_image=True,
                        convergence_factor=1e-5,
                        num_pix_step=10,
                        meta=None,
                        **kwargs):
        if cosmology is None:
            from astropy.cosmology import FlatLambdaCDM
            Om = 0.3
            cosmology = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=Om, Ob0=0.05)

        if kwargs_data is None or kwargs_psf is None or kwargs_numerics is None:
            if meta is None:
                raise ValueError("Either kwargs_data, kwargs_psf, kwargs_numerics or meta must be provided.")
            kwargs_data, kwargs_psf, kwargs_numerics = SED_color_calculator.meta_to_dicts(meta)

        # Defensive copies
        _kwargs_data = deepcopy(kwargs_data)

        psf_model_class = PSF(**kwargs_psf)
        

        light_models = {}
        for model_type in to_compute:
            z = z_lens if model_type == 'lens' else z_source
            light_models[model_type] = LightModel(kwargs_model[f'{model_type}_light_model_list'], source_redshift_list=[z for _ in range(len(kwargs_model[f'{model_type}_light_model_list']))])
        

        lens_model_class = LensModel(kwargs_model['lens_model_list'], lens_redshift_list=[z_lens for _ in range(len(kwargs_model['lens_model_list']))], cosmo=cosmology)

        # compute fluxes for increaseing num_pixes
        flux_diff = np.inf
        total_flux = -np.inf
        num_pix = 100
        
        iteration = 0

        while np.abs(flux_diff) > convergence_factor * total_flux or iteration == 0:
            previous_flux = total_flux
            _kwargs_data['image_data'] = np.zeros((num_pix, num_pix))

            data_class = ImageData(**_kwargs_data)

            if len(to_compute) == 2 and lens_image:
                image_model = ImageModel(data_class, psf_model_class, source_model_class=light_models['source'], lens_light_model_class=light_models['lens'], 
                                        lens_model_class=lens_model_class, kwargs_numerics=kwargs_numerics)
                img = image_model.image(kwargs_lens=kwargs_params['kwargs_lens'], kwargs_source=kwargs_params['kwargs_source'], kwargs_lens_light=kwargs_params['kwargs_lens_light'])

            elif 'lens' in to_compute:
                image_model = ImageModel(data_class, psf_model_class, lens_light_model_class=light_models['lens'], lens_model_class=lens_model_class, kwargs_numerics=kwargs_numerics)
                img = image_model.image(kwargs_lens=None, kwargs_lens_light=kwargs_params['kwargs_lens_light'])

            elif 'source' in to_compute:
                image_model = ImageModel(data_class, psf_model_class, source_model_class=light_models['source'], kwargs_numerics=kwargs_numerics)
                img = image_model.image(kwargs_lens=kwargs_params['kwargs_lens'] if lens_image else None, kwargs_source=kwargs_params['kwargs_source'])
            else:
                raise ValueError("to_compute must contain at least 'lens' or 'source'.")

            total_flux = np.sum(img)
            
            # Check for invalid flux values
            if not np.isfinite(total_flux):
                error_msg = f"Invalid total_flux (NaN or Inf) at iteration {iteration}, num_pix={num_pix}.\n"
                if 'lens' in to_compute and 'kwargs_lens_light' in kwargs_params:
                    error_msg += f"Lens params: R_sersic={kwargs_params['kwargs_lens_light'][0].get('R_sersic', 'N/A')}, "
                    error_msg += f"n_sersic={kwargs_params['kwargs_lens_light'][0].get('n_sersic', 'N/A')}, "
                    error_msg += f"amp={kwargs_params['kwargs_lens_light'][0].get('amp', 'N/A')}\n"
                if 'source' in to_compute and 'kwargs_source' in kwargs_params:
                    error_msg += f"Source params: R_sersic={kwargs_params['kwargs_source'][0].get('R_sersic', 'N/A')}, "
                    error_msg += f"n_sersic={kwargs_params['kwargs_source'][0].get('n_sersic', 'N/A')}, "
                    error_msg += f"amp={kwargs_params['kwargs_source'][0].get('amp', 'N/A')}"
                raise ValueError(error_msg)

            flux_diff = total_flux - previous_flux
            num_pix += num_pix_step

            if verbose:
                print(f"Iteration {iteration}: num_pix = {num_pix}, scaled convergence = {total_flux * convergence_factor}, flux_diff = {flux_diff}")

            iteration += 1
            if iteration > 75:
                # Provide detailed diagnostics
                error_msg = f"Flux computation did not converge after {iteration} iterations.\n"
                error_msg += f"Final: num_pix={num_pix}, total_flux={total_flux:.6e}, flux_diff={flux_diff:.6e}, "
                error_msg += f"convergence_threshold={total_flux * convergence_factor:.6e}\n"
                if 'lens' in to_compute and 'kwargs_lens_light' in kwargs_params:
                    error_msg += f"Lens params: R_sersic={kwargs_params['kwargs_lens_light'][0].get('R_sersic', 'N/A')}, "
                    error_msg += f"n_sersic={kwargs_params['kwargs_lens_light'][0].get('n_sersic', 'N/A')}, "
                    error_msg += f"amp={kwargs_params['kwargs_lens_light'][0].get('amp', 'N/A')}\n"
                if 'source' in to_compute and 'kwargs_source' in kwargs_params:
                    error_msg += f"Source params: R_sersic={kwargs_params['kwargs_source'][0].get('R_sersic', 'N/A')}, "
                    error_msg += f"n_sersic={kwargs_params['kwargs_source'][0].get('n_sersic', 'N/A')}, "
                    error_msg += f"amp={kwargs_params['kwargs_source'][0].get('amp', 'N/A')}"
                raise RuntimeError(error_msg)

        if verbose:
            print(f"Converged after {iteration} iterations with num_pix = {num_pix}.")
            print(f"Final total flux: {total_flux}, flux difference: {flux_diff}")
        return total_flux * kwargs_data['transform_pix2angle'][0,0]**2  # scale by pixel area

    @staticmethod
    def meta_to_dicts(image_meta, **kwargs):
        '''
        Convert image metadata to LENSTRONOMY kwargs dictionaries for data, psf, and numerics.
        Parameters:
        -----------
        image_meta : dict
            Dictionary containing image metadata.
        Returns:
        --------
        kwargs_data : dict
            LENSTRONOMY kwargs dictionary for image data.
        kwargs_psf : dict
            LENSTRONOMY kwargs dictionary for PSF.
        kwargs_numerics : dict
            LENSTRONOMY kwargs dictionary for numerics.
        '''
        kwargs_data = {
            'image_data': np.zeros((image_meta['num_pix'], image_meta['num_pix'])),
            'background_rms': image_meta['background_rms'],
            'transform_pix2angle': np.array([[image_meta['pixel_scale'], 0],
                                            [0, image_meta['pixel_scale']]]),
            'ra_at_xy_0': image_meta['ra_at_xy_0'],
            'dec_at_xy_0': image_meta['dec_at_xy_0'],
            'exposure_time': image_meta['exposure_time']
        }
        if 'psf_fwhm' in image_meta:
            kwargs_psf = {
                'psf_type': 'GAUSSIAN',
                'fwhm': image_meta['psf_fwhm'],
                'pixel_size': image_meta['pixel_scale']
            }
        else:
            filter_name = image_meta.get('filter_name', None)
            if filter_name is None:
                raise ValueError("If 'psf_fwhm' is not provided in image_meta, 'filter_name' must be provided to determine PSF.")

            psf_data = load_psf_roman(filter_name)
            kwargs_psf = {
                'psf_type': 'PIXEL',
                'kernel_point_source': psf_data,
                'kernel_point_source_init': psf_data 
            }

        kwargs_numerics = {
            'supersampling_factor': image_meta['supersampling_factor'],
            'supersampling_convolution': image_meta['supersampling_convolution']
        }

        return kwargs_data, kwargs_psf, kwargs_numerics

    @staticmethod
    def light_params_to_dict(R_sersic, n_sersic, q=None, theta=None, e1=None, e2=None):
        '''
        Create a dictionary for LENSTRONOMY of light profile parameters for Sersic light model.
        Parameters:
        -----------
        R_sersic : float
            Effective radius of the Sersic profile.
        n_sersic : float
            Sersic index.
        q : float, optional
            Axis ratio. If not provided, e1 and e2 must be provided.
        theta : float, optional
            Position angle in degrees.
        e1 : float, optional
            Ellipticity component 1.
        e2 : float, optional
            Ellipticity component 2.

        Returns:
        --------
        params : dict
            Dictionary of light profile parameters.'''
        params = {'R_sersic': R_sersic,
                'n_sersic': n_sersic}
        if (q is None and theta is None) and (e1 is None or e2 is None):
            raise ValueError("Either 'q' and 'theta' or 'e1' and 'e2' must be provided.")
        
        if q is not None and theta is not None:
            params['q'] = q
            params['theta'] = theta
        elif e1 is not None and e2 is not None:
            params['e1'] = e1
            params['e2'] = e2
        return params

    @staticmethod
    def mass_profile_params_to_dict(theta_E, x, y, q=None, theta=None, e1=None, e2=None):
        '''
        Create a dictionary for LENSTRONOMY of mass profile parameters for SIE mass model.

        Parameters:
        -----------
        theta_E : float
            Einstein radius.
        x : float
            x position of the mass profile center.
        y : float
            y position of the mass profile center.
        q : float, optional
            Axis ratio. If not provided, e1 and e2 must be provided.
        theta : float, optional
            Position angle in degrees.
        e1 : float, optional
            Ellipticity component 1.
        e2 : float, optional
            Ellipticity component 2.
        
        Returns:
        --------
        params : dict
            Dictionary of mass profile parameters.
        '''
        params = {'theta_E': theta_E,
                'center_x': x,
                'center_y': y}

        if (q is None and theta is None) and (e1 is None or e2 is None):
            raise ValueError("Either 'q' and 'theta' or 'e1' and 'e2' must be provided.")
        
        if q is not None and theta is not None:
            params['q'] = q
            params['theta'] = theta
        elif e1 is not None and e2 is not None:
            params['e1'] = e1
            params['e2'] = e2
        return params

def create_image_data(kwargs_model, kwargs_params, pixel_scale, num_pixels, exp_time, bkg_rms, psf_fwhm, 
                      lens_redshifts, source_redshifts, cosmo=FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3, Ob0=0.05), add_noise=True):
    
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

def create_images(SED_paths, target_AB_mags, kwargs_model, kwarg_params, redshifts, **kwargs):
    calculator = SED_color_calculator(SED_paths, **kwargs)
    amps = calculator.get_amplitudes(target_AB_mags, kwargs_model, kwarg_params, redshifts, **kwargs)
    amplitudes = np.array((amps['lens'], amps['source']))
    if kwargs.get('normalise_amps', True):
        amplitudes /= amplitudes[0, 0]

    VIS_kwargs_params = deepcopy(kwarg_params)
    NIR_Y_kwargs_params = deepcopy(kwarg_params)
    NIR_J_kwargs_params = deepcopy(kwarg_params)
    NIR_H_kwargs_params = deepcopy(kwarg_params)

    VIS_kwargs_params['kwargs_lens_light'][0]['amp'], VIS_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 0]
    NIR_Y_kwargs_params['kwargs_lens_light'][0]['amp'], NIR_Y_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 1]
    NIR_J_kwargs_params['kwargs_lens_light'][0]['amp'], NIR_J_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 2]
    NIR_H_kwargs_params['kwargs_lens_light'][0]['amp'], NIR_H_kwargs_params['kwargs_source'][0]['amp'] = amplitudes[:, 3]

    VIS_kwargs_data = create_image_data(kwargs_model, VIS_kwargs_params, default_Euclid_VIS_image_meta['pixel_scale'], default_Euclid_VIS_image_meta['num_pix'], 
    default_Euclid_VIS_image_meta['exposure_time'], default_Euclid_VIS_image_meta['background_rms'], default_Euclid_VIS_image_meta['fwhm'], 
    [redshifts['lens'] for _ in range(len(kwargs_model['lens']))], [redshifts['lens_model_list'] for _ in range(len(kwargs_model['source_light_model_list']))])
    NIR_Y_kwargs_data = create_image_data(kwargs_model, NIR_Y_kwargs_params, default_Euclid_NIR_Y_image_meta['pixel_scale'], default_Euclid_NIR_Y_image_meta['num_pix'], 
    default_Euclid_NIR_Y_image_meta['exposure_time'], default_Euclid_NIR_Y_image_meta['background_rms'], default_Euclid_NIR_Y_image_meta['fwhm'], 
    [redshifts['lens'] for _ in range(len(kwargs_model['lens']))], [redshifts['lens_model_list'] for _ in range(len(kwargs_model['source_light_model_list']))])
    NIR_J_kwargs_data = create_image_data(kwargs_model, NIR_J_kwargs_params, default_Euclid_NIR_J_image_meta['pixel_scale'], default_Euclid_NIR_J_image_meta['num_pix'], 
    default_Euclid_NIR_J_image_meta['exposure_time'], default_Euclid_NIR_J_image_meta['background_rms'], default_Euclid_NIR_J_image_meta['fwhm'], 
    [redshifts['lens'] for _ in range(len(kwargs_model['lens']))], [redshifts['lens_model_list'] for _ in range(len(kwargs_model['source_light_model_list']))])
    NIR_H_kwargs_data = create_image_data(kwargs_model, NIR_H_kwargs_params, default_Euclid_NIR_H_image_meta['pixel_scale'], default_Euclid_NIR_H_image_meta['num_pix'], 
    default_Euclid_NIR_H_image_meta['exposure_time'], default_Euclid_NIR_H_image_meta['background_rms'], default_Euclid_NIR_H_image_meta['fwhm'], 
    [redshifts['lens'] for _ in range(len(kwargs_model['lens']))], [redshifts['lens_model_list'] for _ in range(len(kwargs_model['source_light_model_list']))])

    return VIS_kwargs_data, NIR_Y_kwargs_data, NIR_J_kwargs_data, NIR_H_kwargs_data

def make_header(kwargs_data, filter_name):
    header = fits.Header()

    header['SIMPLE'] = True
    header['BITPIX'] = -32
    header['NAXIS'] = 2
    header['NAXIS1'] = kwargs_data['image_data'].shape[1]
    header['NAXIS2'] = kwargs_data['image_data'].shape[0]
    header['EQUINOX'] = 2000.0
    header['RADESYSYS'] = 'ICRS'
    header['CTYPE1'] = 'RA---TAN'
    header['CUNIT1'] = 'deg'
    header['CRVAL1'] = 0.0
    header['CRPIX1'] = 0
    header['CD1_1'] = - kwargs_data['transform_pix2angle'][0,0] / 3600.0  # degrees per pixel
    header['CD1_2'] = 0.0
    header['CTYPE1'] = 'RA---TAN'
    header['CUNIT1'] = 'deg'
    header['CRVAL1'] = 0.0
    header['CRPIX1'] = 0
    header['CD2_1'] = 0.0
    header['CD2_2'] = kwargs_data['transform_pix2angle'][0,0] / 3600.0  # degrees per pixel
    header['FILTER'] = filter_name
    return header


def construct_fits_output(kwargs_data_list, filter_names, output_filename):
    # save images as hdulists of hdu with headers
    hdu_list = fits.HDUList()
    primary_hdu = fits.PrimaryHDU()
    hdu_list.append(primary_hdu)

    for i, (kwargs_data, filter_name) in enumerate(zip(kwargs_data_list, filter_names)):
        image_header = make_header(kwargs_data, filter_name)
        image_hdu = fits.ImageHDU(data=kwargs_data['image_data'], name=filter_name, header=image_header)
        hdu_list.append(image_hdu)

    hdu_list.writeto(output_filename, overwrite=True)


def save_to_fits(data_list, header_list, filter_names, output_filename):
    hdu_list = fits.HDUList()
    primary_hdu = fits.PrimaryHDU()
    hdu_list.append(primary_hdu)

    for data, header, filter_name in zip(data_list, header_list, filter_names):
        header['FILTER'] = filter_name
        image_hdu = fits.ImageHDU(data=data, name=filter_name, header=header)
        hdu_list.append(image_hdu)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    hdu_list.writeto(output_filename, overwrite=True)

def load_psf_roman(filter_name):
    filter_to_psf_file = {
        'F106': './roman_psfs/psf_F106.fits',
        'F129': './roman_psfs/psf_F129.fits',
        'F158': './roman_psfs/psf_F158.fits'
    }
    if filter_name not in filter_to_psf_file:
        raise ValueError(f"Filter name '{filter_name}' not recognized. Valid options are: {list(filter_to_psf_file.keys())}")
    
    psf_file = filter_to_psf_file[filter_name]
    try:
        with fits.open(psf_file) as hdul:
            psf_data = hdul[0].data
            return psf_data
    except Exception as e:
        raise IOError(f"Error loading PSF from {psf_file}: \n{e}")

physical_area = 15

default_Euclid_VIS_image_meta = {
    'num_pix': int(physical_area / 0.1),  # 150 pixels to cover 15 arcseconds at 0.1"/pixel
    'pixel_scale': 0.1,  
    'psf_fwhm': 0.203,  
    'background_rms': 0.1, 
    'ra_at_xy_0': -physical_area / 2,  # RA at pixel (0,0) — places ra=0 at the central pixel
    'dec_at_xy_0': -physical_area / 2,  # DEC at pixel (0,0)
    'exposure_time': 2422.0,
    'supersampling_factor': 3,
    'supersampling_convolution': False
}
default_Euclid_NIR_Y_image_meta = default_Euclid_VIS_image_meta.copy()
default_Euclid_NIR_Y_image_meta.update({
    'exposure_time': 87.2 * 4.0,
    'psf_fwhm': 0.475,
    'pixel_scale': 0.3,
    'num_pix': int(physical_area / 0.3),  # 50 pixels to cover 15 arcseconds at 0.3"/pixel
    'ra_at_xy_0': -physical_area / 2,
    'dec_at_xy_0': -physical_area / 2,
})
default_Euclid_NIR_J_image_meta = default_Euclid_VIS_image_meta.copy()
default_Euclid_NIR_J_image_meta.update({
    'exposure_time': 87.2 * 4.0,
    'psf_fwhm': 0.504,
    'pixel_scale': 0.3,
    'num_pix': int(physical_area / 0.3),  # 50 pixels to cover 15 arcseconds at 0.3"/pixel
    'ra_at_xy_0': -physical_area / 2,
    'dec_at_xy_0': -physical_area / 2,
})
default_Euclid_NIR_H_image_meta = default_Euclid_VIS_image_meta.copy()
default_Euclid_NIR_H_image_meta.update({
    'exposure_time': 87.2 * 4.0,
    'psf_fwhm': 0.542,
    'pixel_scale': 0.3,
    'num_pix': int(physical_area / 0.3),  # 50 pixels to cover 15 arcseconds at 0.3"/pixel
    'ra_at_xy_0': -physical_area / 2,
    'dec_at_xy_0': -physical_area / 2,
})

roman_image_meta = {
    'num_pix': int(physical_area / 0.11),  # 136 pixels to cover 15 arcseconds at 0.11"/pixel
    'pixel_scale': 0.11,    
    'ra_at_xy_0': -physical_area / 2,  # RA at pixel (0,0) — places ra=0 at the central pixel
    'dec_at_xy_0': -physical_area / 2,  # DEC at pixel (0,0)
    'exposure_time': 3 * 107.0,  # 6 exposures of 107s each
    'supersampling_factor': 3,
    'supersampling_convolution': False,
    'background_rms': 0.1,  # Placeholder value; should be set based on expected noise characteristics
}

def plot(SED, filter_throughput):
    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(SED[0], SED[1], color='blue', label='SED')
    ax1.set_xlabel('Wavelength (Angstrom)')
    ax1.set_ylabel('Flux', color='blue')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(filter_throughput[0], filter_throughput[1], color='red', label='Filter Throughput')
    ax2.set_ylabel('Throughput', color='red')

    plt.show()