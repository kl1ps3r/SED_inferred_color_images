import numpy as np
from astropy import units as u


def create_multi_band_image(kwargs_model, kwargs_params, kwargs_data_list, kwargs_psf, target_mags, SEDs, redshifts, filter_throughputs, **kwargs):
    # create local copy kwargs_params to avoid modifying original
    import copy
    kwargs_params_local = copy.deepcopy(kwargs_params)

    kwargs_params_local['kwargs_lens_light'][0]['amplitude'] = 1.0
    kwargs_params_local['kwargs_source_light'][0]['amplitude'] = 1.0

    # generate scaling factors for each source in each band
    amplitudes = get_amplitudes(
        target_AB_mags=target_mags,
        SEDs=SEDs,
        redshifts=redshifts,
        filter_throughputs=filter_throughputs,
        kwargs_model=kwargs_model
    )

    multi_band_data = []

def create_image_data(kwargs_model, kwargs_params, kwargs_data, kwargs_psf, kwargs_numerics=None,
                      lens_redshifts=None, source_redshifts=None, cosmo=None, add_noise=True):
    
    # sanitise inputs

    if cosmo is None:
        from astropy.cosmology import wCDM
        Om = 0.3
        w = -1
        cosmo = wCDM(H0=70 * u.km / u.s / u.Mpc, Om0=Om, Ob0=0.05, Ode0=1-Om, w0=w) # type: ignore

    kwargs_data['image_data'] = np.zeros_like(kwargs_data['image_data'])

    if kwargs_numerics is None:
        kwargs_numerics = {'supersampling_factor': 3, 'supersampling_convolution': True}

    # import lenstronomy classes
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel

    # initialise lenstronomy classes
    data_class = ImageData(**kwargs_data)
    psf_class = PSF(**kwargs_psf)

    if kwargs_model['lens_model_list'] != []:
        if lens_redshifts is None:
            raise ValueError('lens_redshifts must be provided if lens_model_list is not empty')

        from lenstronomy.LensModel.lens_model import LensModel
        lens_model_class = LensModel(kwargs_model['lens_model_list'], lens_redshift_list=lens_redshifts, cosmo=cosmo)
    else:
        lens_model_class = []
    if kwargs_model['source_light_model_list'] != []:
        if source_redshifts is None:
            raise ValueError('source_redshifts must be provided if source_light_model_list is not empty')
        from lenstronomy.LightModel.light_model import LightModel
        source_model_class = LightModel(kwargs_model['source_light_model_list'], source_redshift_list=source_redshifts)
    else:
        source_model_class = []
    if kwargs_model['lens_light_model_list'] != []:
        from lenstronomy.LightModel.light_model import LightModel
        lens_light_model_class = LightModel(kwargs_model['lens_light_model_list'])
    else:
        lens_light_model_class = []

    model_classes = {
        'lens_model_class': lens_model_class,
        'source_model_class': source_model_class,
        'lens_light_model_class': lens_light_model_class,
        'kwargs_numerics': kwargs_numerics
    }
    # initialize image model
    image_model = ImageModel(data_class, psf_class, **model_classes)

    # generate image
    image_model = image_model.image(**kwargs_params)

    # add noise if specified
    if add_noise:
        import lenstronomy.Util.image_util as image_util
        poisson_noise = image_util.add_poisson(image_model, exp_time=kwargs_data['exp_time'])
        image_real = image_model + poisson_noise
    else:
        image_real = image_model
    
    # update data_class with generated image
    data_class.update_data(image_real)
    kwargs_data['image_data'] = image_real

    return kwargs_data

def get_scale_factor(target_AB_mag: dict, SED: np.ndarray, filter_throughput: np.ndarray, interp_method=None, intergrate_method=None):

    if interp_method is None:
        from numpy import interp
        interp_method = interp
    if intergrate_method is None:
        from scipy.integrate import trapezoid
        intergrate_method = trapezoid

    # interpolate filter_throughput to SED wavelengths
    if not np.all(np.diff(filter_throughput[0]) > 0):
        raise ValueError('Filter wavelengths are not strictly increasing')
    
    # create mask of SED wavelengths where throughput exists


    interped_filter_throughput = interp_method(SED[0], filter_throughput[0], filter_throughput[1], left=0, right=0)

    f_lambda = intergrate_method(y=interped_filter_throughput * SED[1] * SED[0], x=SED[0]) / intergrate_method(y=interped_filter_throughput * SED[0], x=SED[0])
    lambda_p_2 = intergrate_method(y=interped_filter_throughput * SED[0], x=SED[0]) / intergrate_method(y=interped_filter_throughput / SED[0], x=SED[0])
    from astropy.constants import c
    AB_mag = -2.5 * np.log10(f_lambda) - 2.5 * np.log10(lambda_p_2 / c.to(u.Angstrom / u.s).value) - 48.6 # type: ignore

    return 10 ** (-0.4 * (target_AB_mag - AB_mag))

def get_weighted_mean_flux(SED: np.ndarray, filter_throughput: np.ndarray, scaling_factor: float=1.0, interp_method=None, intergrate_method=None, verbose=False):
        '''
        Helper function to get mean weighted flux of SED through filter throughput.
        Parameters
        ----------
        SED : np.ndarray
            2D array of shape (2, N) where first row is wavelength in Angstroms and second row is flux density in erg/s/cm^2/Angstrom.
        filter_throughput : np.ndarray
            2D array of shape (2, M) where first row is wavelength in Angstroms and second row is throughput (0 to 1).
        interp_method : function, optional
            Interpolation method to use. Default is numpy.interp.
        intergrate_method : function, optional
            Integration method to use. Default is scipy.integrate.trapezoid.
        Returns
        -------
        float
            Mean weighted flux of SED through filter throughput in erg/s/cm^2.
        '''
        if interp_method is None:
            from numpy import interp
            interp_method = interp
        if intergrate_method is None:
            from scipy.integrate import trapezoid
            intergrate_method = trapezoid

        # interpolate filter_throughput to SED wavelengths
        if not np.all(np.diff(filter_throughput[0]) > 0):
            raise ValueError('Filter wavelengths are not strictly increasing')

        interped_filter_throughput = interp_method(SED[0], filter_throughput[0], filter_throughput[1], left=0, right=0)

        f_lambda = intergrate_method(y=interped_filter_throughput * SED[1] * scaling_factor * SED[0], x=SED[0]) / intergrate_method(y=interped_filter_throughput * SED[0], x=SED[0])
        if verbose:
            print(f'Interpolated filter throughput: {interped_filter_throughput}\nFor Lambda: {SED[0]}')
            print(f'Numerator: {intergrate_method(y=interped_filter_throughput * SED[1] * SED[0], x=SED[0])}')
            print(f'Denominator: {intergrate_method(y=interped_filter_throughput * SED[0], x=SED[0])}')
        return f_lambda

def redshift(SED, z):
    '''
    Helper function to redshift SED to given redshift z given a cosmology.
    Parameters
    ----------
    SED : np.ndarray
        2D array of shape (2, N) where first row is wavelength in Angstroms and second row is flux density in erg/s/cm^2/Angstrom.
    z : float
        Redshift to shift SED to.
    cosmology : astropy.cosmology.FLRW
        Cosmology to use for luminosity distance calculation.
    Returns
    -------
    np.ndarray
        2D array of shape (2, N) where first row is redshifted wavelength in Angstroms and second row is redshifted flux density in erg/s/cm^2/Angstrom.
    '''
    lambda_out = SED[0] * (z + 1)
    flux_density_out = SED[1] / (1+z)

    return np.array((lambda_out, flux_density_out))


def get_amplitudes(target_AB_mags: dict, SEDs: dict, redshifts: dict, filter_throughputs: dict, kwargs_model: dict, **kwargs):

    '''
    Get a dictionary of amplitudes for sersic profiles given target AB magnitudes in each filter relative to the VIS band of the primary deflector light profile.
    Parameters
    ----------
    target_AB_mags : dict
        Dictionary of target AB magnitudes for each model type (lens_light, source_light).
    SED : dict
        Dictionary of SEDs for each model type (lens_light, source_light). Each SED is a 2D array of shape (2, N) 
        where first row is wavelength in Angstroms and second row is flux density in erg/s/cm^2/Angstrom.
    redshifts : dict
        Dictionary of redshifts for each model type (lens_light, source_light).
    filter_throughputs : dict
        Dictionary of filter throughputs for each filter. Each throughput is a 2D array of shape (2, N) 
        where first row is wavelength in Angstroms and second row is throughput (0 to 1).
    kwargs_model : dict
        Dictionary of lenstronomy kwargs_model.
    kwargs : dict
        Additional keyword arguments which can include interpolation method and integration method.
    Returns
    -------
    dict
        Dictionary of amplitudes for each model type (lens_light, source_light) for each filter.
    '''

    '''
    Do i need to scale by luminosity distance squared? as I will be scaling the SED fluxes directly to observed fluxes.
    if 'cosmo' not in kwargs:
        from astropy.cosmology import wCDM
        Om = 0.3
        w = -1
        kwargs['cosmo'] = wCDM(H0=70 * u.km / u.s / u.Mpc, Om0=Om, Ob0=0.05, Ode0=1-Om, w0=w)
    '''

    weighted_mean_fluxes = {}

    for model_type in target_AB_mags.keys():                # model types: lens_light, source_light

        scale_factor = get_scale_factor(target_AB_mags[model_type], SEDs[model_type], filter_throughputs[0], **kwargs)
        shifted_SED = redshift(SEDs[model_type], redshifts[model_type])

        weighted_mean_fluxes[model_type] = np.array([get_weighted_mean_flux(shifted_SED, throughput, scale_factor, **kwargs) for throughput in filter_throughputs])

    filter_ratios = weighted_mean_fluxes['lens_light'] / weighted_mean_fluxes['source_light']

    amplitudes = {
        'lens_light': weighted_mean_fluxes['lens_light'] / weighted_mean_fluxes['lens_light'][0],
        'source_light': weighted_mean_fluxes['source_light'] / weighted_mean_fluxes['source_light'][0] * filter_ratios
    }

    return amplitudes
