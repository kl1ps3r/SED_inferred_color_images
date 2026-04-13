import os
import numpy as np
from argparse import ArgumentParser

from astropy import units as u
from astropy.constants import c

from scipy.integrate import trapezoid
import photometry

import pandas as pd

from skypy.galaxies.luminosity import schechter_lf_magnitude
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

distributions = {
    'source': {
        'redshift': np.array([[1.2, 2.5]]),
        'effective_radius': 
            np.array([
                [8.44, 3.28],
                [5.39, 1.77],
                [4.91, 2.04],
                [4.81, 1.90],
                [3.88, 1.51],
                [2.55, 1.18]
            ]),
        'sersic_index':         
            np.array([
                [2.71, 1.19],
                [2.65, 1.28],
                [1.86, 0.98],
                [1.53, 0.87],
                [1.20, 0.73],
                [1.38, 0.62]
            ]),
        'axis_ratio': np.array([[0.7, 0.15]]),
        'AB_magnitude': np.array([[23.5, 0.25]]),
        'pos_x_offset': np.array([[0.0, 0.05]]),
        'pos_y_offset': np.array([[0.0, 0.05]]),
        'angle_offset': np.array([[0.0, np.pi/32]]),
        },
        
    'deflector': {
        'redshift': np.array([[0.3, 0.9]]),
        'effective_radius': 
            np.array([
                [7.15, 1.56],
                [4.77, 2.05],
                [3.52, 1.79],
                [2.06, 1.04],
                [1.31, 0.73],
                [1.30, 0.55]
                ]),
        'sersic_index':
            np.array([
                [4.83, 1.19],
                [5.57, 1.46],
                [5.13, 1.41],
                [4.39, 1.32],
                [3.97, 1.38],
                [2.73, 0.96]
            ]),
        'axis_ratio':
            np.array([
                [0.74, 0.13],
                [0.71, 0.15],
                [0.67, 0.19],
                [0.63, 0.19],
                [0.65, 0.17],
                [0.68, 0.11]
            ]),
        'angle_offset': np.array([[0.0, 2*np.pi]]),
        'M_star': -23.5,
        'alpha': -1.36,
        'sigma_star': 200,
        'Re_star': 2.43,
        'Re_alpha': 0.83,
        'Re_beta': 1.0
    }
}

units = {
    'effective_radius': 'kpc',
    'sersic_index': '',
    'axis_ratio': '',
    'AB_magnitude': 'mag',
    'pos_x_offset': 'arcsec',
    'pos_y_offset': 'arcsec',
    'angle_offset': 'rad'
}

def generate_galaxy_parameters(num_galaxies):

    '''
    Rejection sampling of parameters for  source deflector galaxy pairs given a VIS apparent magnitude cut of 22.5.
    '''

    batch_size = int(num_galaxies * 10)

    source_keys = ['effective_radius','sersic_index','AB_magnitude','pos_x_offset','pos_y_offset','angle_offset','axis_ratio','redshift']
    deflector_keys = ['effective_radius','sersic_index','axis_ratio','AB_magnitude','pos_x_offset','pos_y_offset','angle_offset','redshift', 'einstein_radius']

    source_parameters = pd.DataFrame(columns=source_keys)
    deflector_parameters = pd.DataFrame(columns=deflector_keys)

    rng = np.random.default_rng()

    number_remaining = num_galaxies

    while number_remaining > 0:
        
        source_redshift, source_redshift_counts = generate_redshift_distribution(
            batch_size, distribution='uniform', z_min=distributions['source']['redshift'][0,0], z_max=distributions['source']['redshift'][0,1])
        deflector_redshift, deflector_redshift_counts = generate_redshift_distribution(
            batch_size, distribution='uniform', z_min=distributions['deflector']['redshift'][0,0], z_max=distributions['deflector']['redshift'][0,1])

        # sample intermediate deflector parameters

        deflector_abs_mag = schechter_lf_magnitude(redshift=deflector_redshift, M_star=distributions['deflector']['M_star'], alpha=distributions['deflector']['alpha'], m_lim=22.5, cosmology=cosmo)
        deflector_luminosity = 10**(-0.4*(deflector_abs_mag - distributions['deflector']['M_star']))
        deflector_velocity_dispersion = (deflector_luminosity)**(1/4) * distributions['deflector']['sigma_star'] * u.km / u.s

        # sample final deflector parameters

        deflector_einstein_radius = (4 * np.pi * (deflector_velocity_dispersion/c.to(u.km/u.s))**2 * cosmo.angular_diameter_distance_z1z2(deflector_redshift, source_redshift) / cosmo.angular_diameter_distance(source_redshift) * u.rad).to(u.arcsec)
        deflector_effective_radius = distributions['deflector']['Re_star'] * (deflector_luminosity)**distributions['deflector']['Re_alpha'] * (1 + deflector_redshift)**(-distributions['deflector']['Re_beta']) * u.kpc
        deflector_apparent_mag = apparent_magnitude(deflector_abs_mag, deflector_redshift)
        deflector_rotation_angle = rng.uniform(distributions['deflector']['angle_offset'][0,0], distributions['deflector']['angle_offset'][0,1], size=deflector_redshift.shape[0])
        

        # sample redshift binned deflector parameters
        deflector_sersic_in_bins = [list(rng.normal(loc=distributions['deflector']['sersic_index'][i,0], scale=distributions['deflector']['sersic_index'][i,1], size=count)) for i, count in enumerate(deflector_redshift_counts)]
        deflector_axis_ratio_in_bins = [list(rng.normal(loc=distributions['deflector']['axis_ratio'][i,0], scale=distributions['deflector']['axis_ratio'][i,1], size=count)) for i, count in enumerate(deflector_redshift_counts)]

        deflector_sersic = []
        deflector_axis_ratio = []

        for z in deflector_redshift:
            match z:
                case z if z >= 0.2 and z < 0.6:
                    bin_index = 0
                case z if 0.6 <= z < 1.0:
                    bin_index = 1
                case z if 1.0 <= z < 1.5:
                    bin_index = 2
                case z if 1.5 <= z < 2.0:
                    bin_index = 3
                case z if 2.0 <= z < 3.0:
                    bin_index = 4
                case _:
                    continue

            deflector_sersic.append(deflector_sersic_in_bins[bin_index].pop(0))
            deflector_axis_ratio.append(deflector_axis_ratio_in_bins[bin_index].pop(0))

        # clip sersic index and axis ratio to physical values
        deflector_sersic = np.clip(deflector_sersic, 1.5, 8.0)
        deflector_axis_ratio = np.clip(deflector_axis_ratio, 0.1, 1.0)

        # sample source constant parameters 

        source_ab_mag = rng.normal(loc=distributions['source']['AB_magnitude'][0,0], scale=distributions['source']['AB_magnitude'][0,1], size=source_redshift.shape[0])
        source_axis_ratio = rng.normal(loc=distributions['source']['axis_ratio'][0,0], scale=distributions['source']['axis_ratio'][0,1], size=source_redshift.shape[0])
        source_rotation_angle = rng.uniform(distributions['source']['angle_offset'][0,0], distributions['source']['angle_offset'][0,1], size=source_redshift.shape[0])
        source_x_offset = rng.normal(loc=distributions['source']['pos_x_offset'][0,0], scale=distributions['source']['pos_x_offset'][0,1], size=source_redshift.shape[0])
        source_y_offset = rng.normal(loc=distributions['source']['pos_y_offset'][0,0], scale=distributions['source']['pos_y_offset'][0,1], size=source_redshift.shape[0])


        # sample redshift binned source parameters
        source_effective_radius_in_bins = [list(rng.normal(loc=distributions['source']['effective_radius'][i,0], scale=distributions['source']['effective_radius'][i,1], size=count)) for i, count in enumerate(source_redshift_counts)]
        source_sersic_in_bins = [list(rng.normal(loc=distributions['source']['sersic_index'][i,0], scale=distributions['source']['sersic_index'][i,1], size=count)) for i, count in enumerate(source_redshift_counts)]

        source_effective_radius = []
        source_sersic = []

        for z in source_redshift:
            match z:
                case z if z >= 0.2 and z < 0.6:
                    bin_index = 0
                case z if 0.6 <= z < 1.0:
                    bin_index = 1
                case z if 1.0 <= z < 1.5:
                    bin_index = 2
                case z if 1.5 <= z < 2.0:
                    bin_index = 3
                case z if 2.0 <= z < 3.0:
                    bin_index = 4
                case _:
                    continue

            source_effective_radius.append(source_effective_radius_in_bins[bin_index].pop(0))
            source_sersic.append(source_sersic_in_bins[bin_index].pop(0))

        # clip sersic index and effective radius to physical values
        source_sersic = np.clip(source_sersic, 0.5, 8.0)
        source_effective_radius = np.clip(source_effective_radius, 0.1, 20.0)

        # create dataframes for source and deflector parameters
        source_df = pd.DataFrame({
            
            'effective_radius': source_effective_radius,
            'sersic_index': source_sersic,
            'AB_magnitude': source_ab_mag,
            'pos_x_offset': source_x_offset,
            'pos_y_offset': source_y_offset,   
            'angle_offset': source_rotation_angle,
            'axis_ratio': source_axis_ratio,
            'redshift': source_redshift,
        })

        deflector_df = pd.DataFrame({
            'effective_radius': deflector_effective_radius.value,
            'sersic_index': deflector_sersic,
            'axis_ratio': deflector_axis_ratio,
            'AB_magnitude': deflector_apparent_mag,
            'pos_x_offset': np.zeros(deflector_redshift.shape[0]),
            'pos_y_offset': np.zeros(deflector_redshift.shape[0]),
            'angle_offset': deflector_rotation_angle,
            'redshift': deflector_redshift,
            'einstein_radius': deflector_einstein_radius.value
        })

        # create mask for source-deflector pairs that satisfy the apparent magnitude cut

        detect = deflector_apparent_mag < 22.5
        strong = deflector_einstein_radius.value > 0.5

        mask = detect & strong

        weights = deflector_einstein_radius.value**2

        # optional detectability term
        weights *= 10**(-0.4 * (deflector_apparent_mag - 22.5))

        # normalise safely
        p_acc = weights / weights.max()

        # 4. rejection step
        random_draws = np.random.rand(batch_size)
        accepted = mask & (random_draws < p_acc)

        # apply mask to source and deflector dataframes
        source_df = source_df.loc[accepted].copy()
        deflector_df = deflector_df.loc[accepted].copy()

        # append accepted pairs to the running parameter tables
        source_parameters = pd.concat([source_parameters, source_df], ignore_index=True)
        deflector_parameters = pd.concat([deflector_parameters, deflector_df], ignore_index=True)

        number_remaining = num_galaxies - deflector_parameters.shape[0]
        print(number_remaining)
        
    source_parameters = source_parameters.head(num_galaxies)
    deflector_parameters = deflector_parameters.head(num_galaxies)
    return {
        'source': source_parameters,
        'deflector': deflector_parameters
    }

def apparent_magnitude(absolute_magnitude, redshift):
    mu = 5 * np.log10(cosmo.luminosity_distance(redshift).value*1e6) - 5

    k_correction = vis_k_correction(redshift)

    return absolute_magnitude + mu + k_correction


def vis_k_correction(z):
    def redshift_sed(SED, z):
        return np.array([SED[0] * (1 + z), SED[1] / (1 + z)])
    
    def get_weighted_mean_flux(SED, filter_throughput):
        interp_filter = np.interp(
            SED[0],
            filter_throughput[0],
            filter_throughput[1],
            left=0.0,
            right=0.0,
        )
        numerator = trapezoid(SED[1] * interp_filter * SED[0], SED[0])
        denominator = trapezoid(interp_filter * SED[0], SED[0])
        return numerator / denominator

    def get_ab_magnitude(SED, filter_throughput):
        mean_flux = get_weighted_mean_flux(SED, filter_throughput)
        effective_wavelength = trapezoid(
            filter_throughput[0] * filter_throughput[1],
            filter_throughput[0],
        ) / trapezoid(filter_throughput[1], filter_throughput[0])
        f_nu = mean_flux * effective_wavelength ** 2 / c.to(u.AA / u.s).value
        return -2.5 * np.log10(f_nu) - 48.6


    VIS_filter_passband = photometry.Passband(file='VIS.Euclid.pb')
    VIS_filter = np.array([VIS_filter_passband.lam(unit=u.AA).value, VIS_filter_passband.y])

    elliptical_SED = np.loadtxt('./inputs/SEDs/Ell13_template_norm.csv', unpack=True)

    rest_vis_ab_mag = get_ab_magnitude(elliptical_SED, VIS_filter)

    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    k_vals = np.array([
        get_ab_magnitude(redshift_sed(elliptical_SED, zi), VIS_filter) - rest_vis_ab_mag
        for zi in z_arr
    ])
    return k_vals[0] if np.ndim(z) == 0 else k_vals

def generate_redshift_distribution(num_galaxies, distribution='uniform', bins=np.array([0.2, 0.6, 1.0, 1.5, 2.0, 3.0]), rng=np.random.default_rng(), **kwargs):
    """
    Generate a redshift distribution for a specified number of galaxies and calculates the counts given a set of redshift bins.

    Parameters:
    num_galaxies (int): The number of galaxies to generate redshifts for.
    distribution (str): The type of distribution to use ('uniform' or 'custom').
    **kwargs: Additional parameters for custom distributions.

    Returns:
    np.ndarray: An array of redshift values.
    """
    if distribution == 'uniform':
        z_min = kwargs.get('z_min', 0.1)
        z_max = kwargs.get('z_max', 3.0)
        redshifts = rng.uniform(z_min, z_max, num_galaxies)
    elif distribution == 'number_density':
        # Not sure what to do
        pass
    else:
        raise ValueError("Unsupported distribution type.")

    # Calculate the histogram of redshifts
    redshift_counts, _ = np.histogram(redshifts, bins=bins)

    return redshifts, redshift_counts

def save_to_csv(galaxy_parameters, filename_prefix, **kwargs):
    """
    Save galaxy parameters to CSV files.

    Parameters:
    galaxy_parameters (dict): A dictionary containing galaxy parameters.
    filename_prefix (str): The prefix for the output CSV files.
    """
    verbose = kwargs.get('verbose', False)
    os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)

    for galaxy_type, params in galaxy_parameters.items():
        filename = f"{filename_prefix}_{galaxy_type}.csv"
        params.to_csv(filename, index=False)
        if verbose:
            print(f"Saved {galaxy_type} parameters to {filename}")


def running_mean_in_z_space(params, parameter, z_bin_size, step_size=None, z_min=None, z_max=None, min_count=1):
    """
    Compute running statistics for a parameter in redshift-space windows.

    Parameters:
    params (pd.DataFrame): DataFrame containing at least `redshift` and `parameter`.
    parameter (str): Column name to average.
    z_bin_size (float): Width of each redshift window.
    step_size (float): Distance between adjacent window centers. Defaults to z_bin_size / 2.
    z_min (float): Minimum redshift to evaluate. Defaults to data minimum.
    z_max (float): Maximum redshift to evaluate. Defaults to data maximum.
    min_count (int): Minimum number of points required to report stats.

    Returns:
    pd.DataFrame: Columns are z_left, z_center, z_right, running_mean, running_std, and count.
    """
    if parameter not in params.columns:
        raise ValueError(f"Column '{parameter}' not found in input DataFrame.")
    if 'redshift' not in params.columns:
        raise ValueError("Column 'redshift' not found in input DataFrame.")
    if z_bin_size <= 0:
        raise ValueError("z_bin_size must be > 0.")

    clean = params[['redshift', parameter]].dropna()
    if clean.empty:
        return pd.DataFrame(columns=['z_left', 'z_center', 'z_right', 'running_mean', 'running_std', 'count'])

    z_vals = clean['redshift'].to_numpy()
    p_vals = clean[parameter].to_numpy()

    if z_min is None:
        z_min = float(z_vals.min())
    if z_max is None:
        z_max = float(z_vals.max())
    if z_max <= z_min:
        raise ValueError("z_max must be greater than z_min.")

    if step_size is None:
        step_size = z_bin_size / 2.0
    if step_size <= 0:
        raise ValueError("step_size must be > 0.")

    half_width = z_bin_size / 2.0
    centers = np.arange(z_min + half_width, z_max - half_width + step_size, step_size)

    rows = []
    for center in centers:
        left = center - half_width
        right = center + half_width
        in_window = (z_vals >= left) & (z_vals < right)

        count = int(np.count_nonzero(in_window))
        if count >= min_count:
            window_vals = p_vals[in_window]
            mean_val = float(np.mean(window_vals))
            std_val = float(np.std(window_vals, ddof=1)) if count > 1 else np.nan
        else:
            mean_val = np.nan
            std_val = np.nan
        rows.append({
            'z_left': left,
            'z_center': center,
            'z_right': right,
            'running_mean': mean_val,
            'running_std': std_val,
            'count': count,
        })

    return pd.DataFrame(rows)




def plot_params(source_params, deflector_params):
    from matplotlib import pyplot as plt
    import glob

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams.update({'figure.autolayout': True})


    plt.rc('text', usetex=False)
    plt.rc('font', family='serif',size=15)
    plt.rc('font',size=15)
    plt.rc('axes', linewidth=1.5) # change back to 1.5
    plt.rc('axes', labelsize=20) # change back to 10
    plt.rc('xtick', labelsize=18, direction='in')
    plt.rc('ytick', labelsize=18, direction='in')
    plt.rc('legend', fontsize=15) # change back to 7

    # setting xtick parameters:

    plt.rc('xtick.major',size=10,pad=4)
    plt.rc('xtick.minor',size=5,pad=4)

    plt.rc('ytick.major',size=10)
    plt.rc('ytick.minor',size=5)

    colours={'source':'blue', 'deflector':'red'}
    to_do_running_mean = ['effective_radius', 'AB_magnitude']

    running_mean_counts = {}
    z_bin_size = 0.2

    for i, param in enumerate(to_do_running_mean):
        running_mean_counts[param] = {'source': [], 'deflector': []}
        for galaxy_type, params in zip(['source', 'deflector'], [source_params, deflector_params]):
            running_mean_counts[param][galaxy_type] = running_mean_in_z_space(
                params,
                parameter=param,
                z_bin_size=z_bin_size,
            )

    '''fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, param in enumerate(to_do_running_mean):
        for galaxy_type, params in zip(['source', 'deflector'], [source_params, deflector_params]):
            #axes[i].scatter(params['redshift'], params[param], color=colours[galaxy_type], alpha=0.5, label=galaxy_type)
            rm = running_mean_counts[param][galaxy_type]
            axes[i].plot(rm['z_center'], rm['running_mean'], color=colours[galaxy_type], label=f'{galaxy_type} mean')
            axes[i].fill_between(
                rm['z_center'],
                rm['running_mean'] - rm['running_std'],
                rm['running_mean'] + rm['running_std'],
                color=colours[galaxy_type],
                alpha=0.15,
                linewidth=0,
            )
        axes[i].set_xlabel('redshift')
        axes[i].set_ylabel(f'{param.replace("_", " ").title()} ({units[param]})')
        
    axes[1].legend()
    plt.savefig('./plots/effective_radius_ab_mag.png', dpi=300)'''
    #plt.show()

    z_vals = pd.to_numeric(deflector_params['redshift'], errors='coerce').to_numpy(dtype=float)
    r_eff_kpc = pd.to_numeric(deflector_params['effective_radius'], errors='coerce').to_numpy(dtype=float)
    einstein_arcsec = pd.to_numeric(deflector_params['einstein_radius'], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(z_vals) & np.isfinite(r_eff_kpc) & np.isfinite(einstein_arcsec)

    d_a_kpc = cosmo.angular_diameter_distance(z_vals[valid]).to(u.kpc).value
    eff_rad_arcsec = ((r_eff_kpc[valid] / d_a_kpc) * u.rad).to(u.arcsec).value

    '''plt.figure(figsize=(6, 5))
    plt.scatter(einstein_arcsec[valid], eff_rad_arcsec, color='red', marker='.')
    plt.xlabel(r'Einstein Radius $[arcsec]$')
    plt.ylabel(r'Effective Radius $[arcsec]$')
    #plt.legend()
    plt.savefig('./plots/einstein_radius_effective_radius_scatter.png', dpi=300)'''

    plt.figure(figsize=(6, 5))

    bins=np.linspace(0, 3.5, 8)
    plt.hist(eff_rad_arcsec, bins=bins, color='red', alpha=0.7, label='Effective Radius')
    plt.hist(einstein_arcsec[valid], bins=bins, color='blue', alpha=0.7, label='Einstein Radius')
    plt.xlabel(r'Einstein Radius $[arcsec]$')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig('./plots/einstein_radius_histogram.png', dpi=300)

    plt.xlim(0, 3.5)

    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('num_galaxies', type=int, default=1000, help='Number of galaxies to generate parameters for.')
    parser.add_argument('-o', '--output', type=str, help='Output filename prefix', default='./galaxy_parameters/galaxy_parameters')
    args = parser.parse_args()


    output = generate_galaxy_parameters(args.num_galaxies)
    save_to_csv(output, args.output, verbose=False)

    plot_params(output['source'], output['deflector'])