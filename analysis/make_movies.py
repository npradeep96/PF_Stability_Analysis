"""Main script to generate movies of concentration profiles from hdf5 files
"""

import sys; sys.path.append('../')
import utils.file_operations as file_operations
import utils.simulation_helper as simulation_helper
import argparse
import re
import h5py
import os
import moviepy.editor as mp
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc

# Suppress any outputs to an interactive interface
matplotlib.use('Agg')

# Add current directory to system path


# Settings to make pretty plots using pyplot
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['font.size'] = 20
plt.rcParams["text.usetex"] = True


def write_movies_two_component_2d(path, hdf5_file, movie_parameters, mesh, fps=3):
    """Function that writes out movies of concentration profiles for 2 component simulations in 2D

    Args:
        path (string): Directory that contains the hdf5_file and input_parameters_file
        hdf5_file (string): Name of the hdf5 file that contains concentration profiles of the 2 components in 2D
        mesh (fipy.mesh): A fipy mesh object that contains mesh.x and mesh.y coordinates
        movie_parameters (dict): A dictionary that contains information on how to make the plots. This is read from
                                 the file movie_parameters.txt
        fps (int): Frame per second to stitch together to make the movie. Default value is 5.
    """

    # make directory to store the movies
    movies_directory = os.path.join(path, 'movies')
    try:
        os.mkdir(movies_directory)
        print("Successfully made the directory " + movies_directory + " ...")
    except OSError:
        print(movies_directory + " directory already exists")

    with h5py.File(os.path.join(path, hdf5_file), mode="r") as concentration_dynamics:
        # Read concentration profile data from files
        concentration_profile = []
        for i in range(int(movie_parameters['num_components'])):
            concentration_profile.append(concentration_dynamics['c_{index}'.format(index=i)])

        # Get upper and lower limits of the concentration values from the concentration profile data
        plotting_range = []
        for i in range(int(movie_parameters['num_components'])):
            # check if plotting range is explicitly specified in movie_parameters
            if 'c{index}_range'.format(index=i) in movie_parameters.keys():
                plotting_range.append(movie_parameters['c{index}_range'.format(index=i)])
            else:
                min_value = np.min(concentration_profile[i][0])
                max_value = np.max(concentration_profile[i][0])
                for t in range(1, concentration_profile[0].shape[0]):
                    if min_value > np.min(concentration_profile[i][t]):
                        min_value = np.min(concentration_profile[i][t])
                    if max_value < np.max(concentration_profile[i][t]):
                        max_value = np.max(concentration_profile[i][t])
                plotting_range.append([min_value, max_value])

        for t in range(concentration_profile[0].shape[0]):
            # Plot and save plots at each time point before stitching them together into a movie

            # Check if we have reached the end of the movie
            flag = False
            zero_component_counter = 0
            for i in range(int(movie_parameters['num_components'])):
                if np.all(concentration_profile[i][t] == 0):
                    zero_component_counter = zero_component_counter + 1
            if zero_component_counter == int(movie_parameters['num_components']):
                flag = True
            if flag:
                break

            # Generate and save plots
            fig, ax = plt.subplots(1, int(movie_parameters['num_components']), figsize=movie_parameters['figure_size'])
            for i in range(int(movie_parameters['num_components'])):
                cs = ax[i].tricontourf(mesh.x, mesh.y, concentration_profile[i][t],
                                       levels=np.linspace(int(plotting_range[i][0]*100)*0.01,
                                                          int(plotting_range[i][1]*100)*0.01,
                                                          256),
                                       cmap=movie_parameters['color_map'][i])
                # ax[i].tick_params(axis='both', which='major', labelsize=20)
                ax[i].xaxis.set_tick_params(labelbottom=False)
                ax[i].yaxis.set_tick_params(labelleft=False)
                cbar = fig.colorbar(cs, ax=ax[i], ticks=np.linspace(int(plotting_range[i][0]*100)*0.01,
                                                                    int(plotting_range[i][1]*100)*0.01,
                                                                    3))
                cbar.ax.tick_params(labelsize=30)
                ax[i].set_title(movie_parameters['titles'][i], fontsize=40)
            fig.savefig(fname=movies_directory + '/Movie_step_{step}.png'.format(step=t), dpi=300, format='png')
            plt.close(fig)

    # Stitch together images to make a movie:

    def key_funct(x):
        return int(x.split('_')[-1].rstrip('.png'))

    file_names = sorted(list((file_name for file_name in os.listdir(movies_directory) if file_name.endswith('.png'))),
                        key=key_funct)
    # print(file_names)
    file_paths = [os.path.join(movies_directory, f) for f in file_names]
    # print(file_paths)
    clip = mp.ImageSequenceClip(file_paths, fps=fps)
    clip.write_videofile(os.path.join(path, 'movies', 'Movie.mp4'), fps=fps)
    clip.close()

    # delete individual images
    for f in file_paths:
        os.remove(f)


def get_movie_maker(input_parameters, movie_parameters):
    """Function that picks out the correct function to make movies

        Args:
            input_parameters (dict): A dictionary that contains (key,value) pairs of (parameter name, parameter value)
                                     associated with the simulation
            movie_parameters (dict): A dictionary that contains information on how to make the plots. This is read from
                                     the file movie_parameters.txt
        Returns:
            movie_maker_function (function): The appropriate function within this script to call and make movies
    """

    movie_maker_function = None

    if input_parameters['dimension'] == 2:
        if movie_parameters['num_components'] == 2:
            movie_maker_function = write_movies_two_component_2d

    return movie_maker_function


if __name__ == "__main__":
    """
        Generates movies of concentration profiles from information stores in hdf5 files.      
        This script assumes that each directory containing a hdf5 file and an input_parameters.txt file with the 
        necessary information to construct a mesh.
    """
    # Define and parse command line arguments
    parser = argparse.ArgumentParser(description='Directory name to search for hdf5 files and generate movies')
    parser.add_argument('--i', help="Directory containing the hdf5 files", required=True)
    parser.add_argument('--h', help="Name of hdf5 file within this directory", required=True)
    parser.add_argument('--p', help="Name of input parameters file for simulations within this directory",
                        required=True)
    parser.add_argument('--r', help="Directory name pattern to match and pick specific directories out",
                        required=True)
    parser.add_argument('--m', help="Path to movie parameters file", required=True)
    args = parser.parse_args()

    # Directory to search
    base_path = args.i
    # Regular expression describing the name of the directory to search for hdf5 files
    regex_1 = str(args.r)
    # Regular expression describing the hdf5 file to search within subdirectories of the base_path
    regex_2 = r'.*' + str(args.h)
    found_at_least_one = 0

    # Loop through all subdirectories in the directory "base_path"
    for root, dirs, files in os.walk(base_path):
        # Loop through all files in each subdirectory to see if we have a hdf5 file with the name given in args.f
        match_1 = re.search(regex_1, root)
        for fi in files:
            match_2 = re.search(regex_2, fi)
        if match_1 is not None and match_2 is not None:
            # Found a hdf5 file!
            found_at_least_one = 1
            # Read the input_parameters.txt file from the directory:
            input_parameters_file = os.path.join(root, args.p)
            input_params = file_operations.input_parse(input_parameters_file)
            print('Successfully parsed input parameters ...')
            # Make a mesh of that geometry
            sim_geometry = simulation_helper.set_mesh_geometry(input_params)
            print('Successfully set up mesh geometry ...')
            # Read input parameter file for movies
            movie_parameters_file = os.path.join(args.m)
            movie_params = file_operations.input_parse(movie_parameters_file)
            # Find the correct function from the menu available at the top of this file to make movies
            movie_maker = get_movie_maker(input_params, movie_params)
            if movie_maker is None:
                print("Could not find an appropriate function to make movies ...")
            else:
                movie_maker(root, fi, movie_params, sim_geometry.mesh)

    if not found_at_least_one:
        print('Could not find any hdf5 files in the supplied directory!')
