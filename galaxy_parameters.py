import numpy as np
import glob

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
        'AB_magnitude': np.array([[23.5, 0.5]]),
        'pos_x_offset': np.array([[0.0, 0.05]]),
        'pos_y_offset': np.array([[0.0, 0.05]]),
        'angle_offset': np.array([[0.0, np.pi/16]]),
        },
        
    'deflector': {
        'redshift': np.array([[0.3, 0.7]]),
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
        'AB_magnitude': np.array([[21.0, 0.5]]),
        'pos_x_offset': np.array([[0.0, 0.1]]),
        'pos_y_offset': np.array([[0.0, 0.1]]),
        'angle_offset': np.array([[0.0, np.pi/4]]),
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

def generate_galaxy_parameters(num_galaxies, source=True, deflector=True, rng=np.random.default_rng(), **kwargs):
    """
    Generate random parameters for a specified number of galaxies.

    Parameters:
    num_galaxies (int): The number of galaxies to generate parameters for.
    source (bool): Whether to generate parameters for source galaxies.
    deflector (bool): Whether to generate parameters for deflector galaxies.

    Returns:
    dict: A dictionary containing arrays of galaxy parameters.
    """
    
    verbose = kwargs.get('verbose', False)

    # generate redshift distribution
    redshifts = {}
    redshift_bins = np.array([0.2, 0.6, 1.0, 1.5, 2.0, 3.0])

    output_parameters = {}
    if source:
        same = ['redshift', 'AB_magnitude', 'pos_x_offset', 'pos_y_offset', 'angle_offset', 'axis_ratio']
        # Source galaxies: redshift between 0.5 and 3.0
        redshifts, redshift_counts = generate_redshift_distribution(
            num_galaxies, distribution='uniform', z_min=distributions['source']['redshift'][0,0], z_max=distributions['source']['redshift'][0,1], 
            bins=redshift_bins, rng=rng
        )

        if verbose:
            print('Redshift Counts for Source Galaxies:', redshift_counts)
        parameters = {}

        for param, dist in distributions['source'].items():
            if param in same:
                continue

            parameters[param] = []
            for i, count in enumerate(redshift_counts):

                mean, std = dist[i+1]

                if verbose:
                    print(f"Sampling {count} values for {param} in redshift bin {i} with mean {mean} and std {std}")

                parameters[param].append(list(rng.normal(mean, std, count)))


        combined_parameters = {}

        # Combine all sampled values into a single array by iterating over list of redshifts
        for z in redshifts:
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

            for param in parameters.keys():
                if param not in combined_parameters:
                    combined_parameters[param] = []
                combined_parameters[param].append(parameters[param][bin_index].pop(0))

        for param, values in combined_parameters.items():
                combined_parameters[param] = np.array(values)

        for param in same:
            if param == 'redshift':
                continue
            mean, std = distributions['source'][param][0]
            sampled_values = np.random.normal(mean, std, num_galaxies)
            combined_parameters[param] = sampled_values
        combined_parameters['redshift'] = redshifts

        if verbose:
            print("Combined Parameters:")
            for param, values in combined_parameters.items():
                print(f"Parameter {param}: {combined_parameters[param].shape}")

        output_parameters['source'] = combined_parameters

    if deflector:
        same = ['redshift', 'AB_magnitude', 'pos_x_offset', 'pos_y_offset', 'angle_offset']
        # Deflector galaxies: redshift between 0.1 and 1.0
        redshifts, redshift_counts = generate_redshift_distribution(
            num_galaxies, distribution='uniform', z_min=distributions['deflector']['redshift'][0,0], z_max=distributions['deflector']['redshift'][0,1], 
            bins=redshift_bins, rng=rng
        )

        if verbose:
            print('Redshift Counts for Deflector Galaxies:', redshift_counts)
        parameters = {}

        for param, dist in distributions['deflector'].items():
            if param in same:
                continue

            parameters[param] = []
            for i, count in enumerate(redshift_counts):

                mean, std = dist[i+1]
                
                if verbose:
                    print(f"Sampling {count} values for {param} in redshift bin {i} with mean {mean} and std {std}")

                parameters[param].append(list(rng.normal(mean, std, count)))


        combined_parameters = {}

        # Combine all sampled values into a single array by iterating over list of redshifts
        for z in redshifts:
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
                    print("Redshift out of bounds:", z)
                    continue

            for param in parameters.keys():
                if param not in combined_parameters:
                    combined_parameters[param] = []
                combined_parameters[param].append(parameters[param][bin_index].pop(0))

        for param, values in combined_parameters.items():
                combined_parameters[param] = np.array(values)
        
        for param in same:
            if param == 'redshift':
                continue
            mean, std = distributions['deflector'][param][0]
            sampled_values = np.random.normal(mean, std, num_galaxies)
            combined_parameters[param] = sampled_values
        combined_parameters['redshift'] = redshifts

        if verbose:
            print("Combined Parameters:")
            for param, values in combined_parameters.items():
                print(f"Parameter {param}: {combined_parameters[param].shape}")

        output_parameters['deflector'] = combined_parameters

    return output_parameters

def physical_to_angular_size(effective_radius_kpc, redshift, cosmo):
    """
    Convert physical size in kpc to angular size in arcseconds.

    Parameters:
    effective_radius_kpc (np.ndarray): Effective radius in kpc.
    redshift (np.ndarray): Redshift of the galaxy.
    cosmo: Cosmology object from astropy.cosmology.

    Returns:
    np.ndarray: Effective radius in arcseconds.
    """
    from astropy import units as u
    from astropy.cosmology import Planck18 as cosmo

    angular_diameter_distance = cosmo.angular_diameter_distance(redshift)  # in Mpc
    angular_size_rad = (effective_radius_kpc * u.kpc) / (angular_diameter_distance.to(u.kpc))  # in radians
    angular_size_arcsec = angular_size_rad.to(u.arcsec).value  # convert to arcseconds

    return angular_size_arcsec

def save_to_csv(galaxy_parameters, filename_prefix):
    """
    Save galaxy parameters to CSV files.

    Parameters:
    galaxy_parameters (dict): A dictionary containing galaxy parameters.
    filename_prefix (str): The prefix for the output CSV files.
    """
    for galaxy_type, params in galaxy_parameters.items():
        filename = f"{filename_prefix}_{galaxy_type}.csv"
        header = ','.join(params.keys())
        data = np.column_stack([params[key] for key in params.keys()])
        np.savetxt(filename, data, delimiter=',', header=header, comments='')
        print(f"Saved {galaxy_type} parameters to {filename}")

if __name__ == "__main__":
    from matplotlib import pyplot as plt
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

    output = generate_galaxy_parameters(10000, source=True, deflector=True, verbose=False)

    #save_to_csv(output, "galaxy_parameters")

    colours={'source':'blue', 'deflector':'red'}
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, (galaxy_type, params) in enumerate(output.items()):
        for ax, name in zip(axes.flatten(), sorted(params.keys())):

            if name == 'redshift':
                ax.hist(params[name], bins=10, color=colours[galaxy_type], alpha=0.5, label=galaxy_type, density=True)
                #ax.set_title(f"{galaxy_type.capitalize()} Galaxy Parameter: {name}")
                ax.set_xlabel('z')
                ax.set_ylabel('Density')
                continue

            #print(f"Plotting {name} for {galaxy_type}")
            # make the plot running mean of parameter vs redshift with standard deviation as confidence interval
            running_mean = []
            running_std = []
            z_bins = np.linspace(np.min(params['redshift']), np.max(params['redshift']), 30)
            z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
            z_centers[0] = z_bins[0]
            z_centers[-1] = z_bins[-1]
            for j in range(len(z_bins)-1):
                bin_mask = (params['redshift'] >= z_bins[j]) & (params['redshift'] < z_bins[j+1])
                if np.sum(bin_mask) > 0:
                    running_mean.append(np.mean(params[name][bin_mask]))
                    running_std.append(np.std(params[name][bin_mask]))
                else:
                    running_mean.append(np.nan)
                    running_std.append(np.nan)
            ax.plot(z_centers, running_mean, '--', color=colours[galaxy_type], alpha=0.9, label=galaxy_type)
            #ax.plot(params['redshift'], params[name], '.', color=colours[galaxy_type], alpha=0.1, label=galaxy_type)
            ax.fill_between(z_centers, np.array(running_mean) - np.array(running_std), np.array(running_mean) + np.array(running_std), color=colours[galaxy_type], alpha=0.5)
            #ax.fill_between(z_centers, np.array(running_mean) - 2*np.array(running_std), np.array(running_mean) + 2*np.array(running_std), color=colours[galaxy_type], alpha=0.3)
            #ax.set_title(f"{name.replace('_', ' ').capitalize()}")
            ax.set_xlabel('z')
            y_unit = units.get(name, '')
            #print(name, y_unit)
            if y_unit != '':
                ax.set_ylabel(f"{name.replace('_', ' ').capitalize()} [{y_unit}]")
            else:
                ax.set_ylabel(name.replace('_', ' ').capitalize())
            #ax.set_ylabel(name.replace('_', ' ').capitalize() + f" [{units.get(name, '')}]")

    axes[0, 1].legend()
    previous_filenames = glob.glob("./plots/galaxy_parameters_v*.png")
    if previous_filenames:
        versions = [int(filename.split('_v')[-1].split('.png')[0]) for filename in previous_filenames]
        previous_version = max(versions)
    else:
        previous_version = 0
    print(f"Plot version: {previous_version + 1}")
    filename = f'./plots/galaxy_parameters_v{previous_version + 1}.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()