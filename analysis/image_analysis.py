"""Module that performs different kinds of analysis on the concentration profiles
"""

import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import utils.file_operations as file_operations
import utils.simulation_helper as simulation_helper
import sys
sys.path.append('../')


def plot_concentration(path, figure_parameters, time_point=-1,
                       spatial_variables_file='spatial_variables.hdf5',
                       input_parameters_file='input_params.txt'):
    """ Function that plots the protein and RNA concentration profiles given a path

    Args:
        path(string): path to the directory that contains the spatial_variables.hdf5 file

        figure_parameters(dict): a dictionary that contains parameters that will be used to plot the figure. Example:
            {num_components: 2
            component_indices: [0, 1]
            color_map: ["Blues", "Reds"]
            titles: ["Protein", "RNA"]
            c0_range: [-2.5, 2.5]
            c1_range: [-2.5, 2.5]
            figure_size: [15, 6]
            keep_axes_labels: True
            }

        time_point(int): Index in the hdf5 file that corresponds to the time at which we want to plot the concentrations
        If this is -1, then it automatically plots the concentration profile at the last time point.

        spatial_variables_file(string): Name of the hdf5 file that stores the concentration profile within /path

        input_parameters_file(string): Name of the file containing simulation input parameters within /path

    Returns:
        fig(matplotlib.figure): Figure object of the plot

        axs(matplotlib.axes): Axes object of the plot

    """

    input_params = file_operations.input_parse(os.path.join(path, input_parameters_file))
    sim_geometry = simulation_helper.set_mesh_geometry(input_params)
    mesh = sim_geometry.mesh

    if os.path.exists(os.path.join(path, spatial_variables_file)):

        with h5py.File(os.path.join(path, spatial_variables_file), mode="r") as concentration_dynamics:
            # Read plotting range for the concentrations and the concentration profile data from files
            concentration_profile = []
            for i in range(int(figure_parameters['num_components'])):
                concentration_profile.append(concentration_dynamics['c_{index}'.format(index=i)])

            if time_point == -1:
                flag = False
                for t in range(concentration_profile[0].shape[0]):
                    # Check if we have reached the end of the time series
                    zero_component_counter = 0
                    for i in range(int(figure_parameters['num_components'])):
                        if np.all(concentration_profile[i][t] == 0):
                            zero_component_counter = zero_component_counter + 1
                    # The default values of the concentrations of all species in the hdf5 files is 0.0, unless this is
                    # overwritten by simulation data. By checking if all the concentration variables at this time point
                    # are 0.0, we are checking if we have reached the end of the simulation as the default values are no
                    # longer overwritten by the simulation data
                    if zero_component_counter == int(figure_parameters['num_components']):
                        flag = True
                    if flag:
                        t = t - 1
                        break
            else:
                t = time_point

            # Get upper and lower limits of the concentration values from the concentration profile data
            plotting_range = []
            for i in range(int((figure_parameters['num_components']))):
                # check if plotting range is explicitly specified in movie_parameters
                if 'c{index}_range'.format(index=i) in figure_parameters.keys():
                    plotting_range.append(figure_parameters['c{index}_range'.format(index=i)])
                else:
                    min_value = np.min(concentration_profile[i][t])
                    max_value = np.max(concentration_profile[i][t])
                    plotting_range.append([min_value, max_value])

            # Generate and save plots
            fig, axs = plt.subplots(1, len(list(figure_parameters['component_indices'])),
                                    figsize=figure_parameters['figure_size'])

            # This if condition is to check if we are plotting a single panel or multi panel plot. For multi panel
            # concentration profile plots, the axs object needs to be indexed
            if len(list(figure_parameters['component_indices'])) > 1:
                for i in list(figure_parameters['component_indices']):
                    cs = axs[i].tricontourf(mesh.x, mesh.y, concentration_profile[i][t],
                                            levels=np.linspace(plotting_range[i][0], plotting_range[i][1] + 0.01, 256),
                                            cmap=figure_parameters['color_map'][i])

                    axs[i].xaxis.set_tick_params(labelbottom=False)
                    axs[i].yaxis.set_tick_params(labelleft=False)
                    if not figure_parameters['keep_axes_labels']:
                        axs[i].axis('off')
                    else:
                        cbar = fig.colorbar(cs, ax=axs[i], ticks=np.linspace(plotting_range[i][0],
                                                                             plotting_range[i][1], 3))
                        cbar.ax.tick_params(labelsize=30)
                        axs[i].set_title(figure_parameters['titles'][i], fontsize=40)
            else:
                for i in list(figure_parameters['component_indices']):
                    cs = axs.tricontourf(mesh.x, mesh.y, concentration_profile[i][t],
                                         levels=np.linspace(plotting_range[i][0], plotting_range[i][1] + 0.01, 256),
                                         cmap=figure_parameters['color_map'][i])

                    axs.xaxis.set_tick_params(labelbottom=False)
                    axs.yaxis.set_tick_params(labelleft=False)
                    if not figure_parameters['keep_axes_labels']:
                        axs.axis('off')
                    else:
                        cbar = fig.colorbar(cs, ax=axs, ticks=np.linspace(plotting_range[i][0],
                                                                             plotting_range[i][1], 3))
                        cbar.ax.tick_params(labelsize=30)
                        axs.set_title(figure_parameters['titles'][i], fontsize=40)

            return fig, axs

    else:
        print(os.path.join(path, spatial_variables_file) + ' does not exist')
        return None, None


def radial_distribution_function(image, center=None, num_bins=100):
    """
    Calculate the radial distribution function (RDF) from a 2D grayscale image.

    Args:
        image (numpy.ndarray): 2D grayscale image as a NumPy array.
        center (tuple): Center coordinates (x, y) of the image. If None, it is set to the image center.
        num_bins (int): Number of radial bins for the RDF calculation.

    Returns:
        r (numpy.ndarray): Radial distances from the center.
        rdf (numpy.ndarray): Radial distribution function values.
    """
    # Ensure image is a NumPy array and is 2D
    image = np.asarray(image, dtype=float)

    # Check if image is a 2D grayscale image
    if image.ndim != 2:
        raise ValueError("Input image should be 2D grayscale")

    # Set the center to the image center if not provided
    if center is None:
        center = (image.shape[1] // 2, image.shape[0] // 2)

    # Calculate the distance from each pixel to the center
    y, x = np.indices(image.shape)
    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Calculate the maximum distance from the center to use as the outer radius
    max_distance = min(center[0], center[1], image.shape[1] - center[0], image.shape[0] - center[1])

    # Compute the RDF
    # g(r) = image_intensity/(2*pi*r)
    rdf_weights = np.divide(image, 2 * np.pi * distances, out=np.zeros_like(image), where=distances != 0.0)

    bins = np.linspace(0, max_distance, num_bins + 1)
    rdf, _ = np.histogram(distances, bins=bins, weights=rdf_weights)
    r = (bins[1:] + bins[:-1]) / 2

    # Remove the value in the first index of the rdf (which is anyway 0 due to division by 0 distance) and return the
    # rest

    return r[1:], rdf[1:]


def smooth_radial_distribution_function(data, window_size=3):
    """
    Smooth a vector containing the radial distribution function using a simple moving average.

    Parameters:
        data (numpy.ndarray): Input time series data as a 1D NumPy array.
        window_size (int): Size of the smoothing window. Default is 3.

    Returns:
        numpy.ndarray: Smoothed time series data as a 1D NumPy array.
    """
    if window_size < 2:
        raise ValueError("Window size should be at least 2 for smoothing.")

    # Check if data has enough points to perform smoothing
    if len(data) < window_size:
        raise ValueError("Data length should be greater than or equal to the window size.")

    # Initialize the smoothed data array with zeros
    smoothed_data = np.zeros_like(data)

    # Perform the moving average smoothing
    for i in range(window_size // 2, len(data) - window_size // 2):
        smoothed_data[i] = np.mean(data[i - window_size // 2:i + window_size // 2 + 1])

    # Copy boundary points directly (padding with zeros)
    smoothed_data[:window_size // 2] = data[:window_size // 2]
    smoothed_data[-window_size // 2:] = data[-window_size // 2:]

    return smoothed_data


def find_local_maxima_indices(data):
    """
    Find the indices of local maxima of the radial distribution function to which can be supplied to the distance vector
    to identify dominant wavelength.

    Parameters:
        data (numpy.ndarray): Input time series data as a 1D NumPy array.

    Returns:
        local_maxima_indices (list): List of indices corresponding to local maxima.
    """
    # Initialize the list to store the indices of local maxima
    local_maxima_indices = []

    # Loop through the data points to find local maxima
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            local_maxima_indices.append(i)

    return local_maxima_indices


def find_local_minima_indices(data):
    """
    Find the indices of local minima of the radial distribution function to which can be supplied to the distance vector
    to identify the radius of the droplet.

    Parameters:
        data (numpy.ndarray): Input time series data as a 1D NumPy array.

    Returns:
        local_minima_indices (list): List of indices corresponding to local maxima.
    """
    # Initialize the list to store the indices of local maxima
    local_minima_indices = []

    # Loop through the data points to find local maxima
    for i in range(1, len(data) - 1):
        if data[i] < data[i - 1] and data[i] < data[i + 1]:
            local_minima_indices.append(i)

    return local_minima_indices


def calculate_autocorrelation(image, window_size, overlap_fraction):
    """
    Calculate the 2D auto correlation map using a sliding square window.

    Parameters:
        image (numpy.ndarray): Input 2D grayscale image as a NumPy array.

        window_size (int): Size of the square window for calculating the auto correlation.

        overlap_fraction (float): A number in (0.0-1.0) that describes the overlap area fraction between sliding windows

    Returns:
        autocorr_map(numpy.ndarray): A 2D auto correlation map as a window_size x window_size NumPy array.
    """
    # Ensure the window size is odd for symmetric windowing
    if window_size % 2 == 0:
        raise ValueError("Window size should be an odd number for symmetric windowing.")

    # Calculate the mean of the entire image
    mean = np.mean(image)

    # Pad the image to handle edge effects
    padded_image = np.pad(image, pad_width=window_size // 2, mode='constant', constant_values=mean)

    # Center the image around 0 and make the values in the range -1 to 1
    padded_image = padded_image - np.mean(padded_image)
    padded_image = 2 * (padded_image - np.min(padded_image)) / (np.max(padded_image) - np.min(padded_image)) - 1

    # Calculate the auto correlation map
    autocorr_map = np.zeros([window_size, window_size], dtype=float)

    counter = 0
    for y in range(window_size // 2, padded_image.shape[0] - window_size // 2, int(overlap_fraction * window_size)):
        for x in range(window_size // 2, padded_image.shape[1] - window_size // 2, int(overlap_fraction * window_size)):
            window = padded_image[y - window_size // 2:y + window_size // 2 + 1,
                                  x - window_size // 2:x + window_size // 2 + 1]
            autocorr_map = autocorr_map + window[window_size // 2, window_size // 2] * window
            counter = counter + 1

    autocorr_map = autocorr_map/counter

    return autocorr_map

