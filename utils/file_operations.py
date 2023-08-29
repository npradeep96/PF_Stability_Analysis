"""Module that contains helper functions to read and write files during simulations

"""

import ast
import numpy as np
import h5py


def input_parse(filename):
    """Parse input parameters from file and return them as a dictionary

    Args:
        filename (string): Name of file that contains the input parameters for simulations

    Returns:
        input_parameters (dict): A dictionary that contains (key,value) pairs of (parameter name, parameter value)
    """

    # Initialize dictionary
    input_parameters = {}

    # Check if input parameter file exists
    try:
        open(filename, 'r')
    except IOError:
        print("Could not open input parameter file: " + filename)
        exit()

    # Open and read input parameter file
    with open(filename, 'r') as f:
        for line in f:
            # Remove the \n character from the string
            line = line.strip()
            # If the line is not an empty string, process the line
            if line:
                # If the line is not a comment i.e. does not start with '#', then process the line
                if line.find('#') == -1:
                    # handle lines with more than 1 comma
                    var_name, var_value = line.split(',')[0], ",".join(line.split(',')[1:])
                    # Remove any white spaces in the variable value
                    var_value = var_value.strip()
                    try:
                        input_parameters[var_name] = float(var_value)
                    except ValueError:
                        # This occurs when python cannot convert a string into a float.
                        # Evaluate the python expression as a list
                        input_parameters[var_name] = ast.literal_eval(var_value)

    return input_parameters


def write_input_params_from_dict(input_parameters, target_filename):
    """Write input parameters to a target file from a dictionary storing input parameter values

    Args:
        input_parameters (dict): A dictionary that contains (key,value) pairs of (parameter name, parameter value)

        target_filename (string): Name of target file to write these parameters to
    """

    with open(target_filename, 'w') as f:
        for key, value in input_parameters.items():
            line = str(key) + ', ' + str(value) + '\n'
            f.write(line)


def write_input_params_from_file(input_filename, target_filename):
    """Write input parameters to a target file from a source file

    Args:
        input_filename (string): Name of source file that contains the input parameters for simulations

        target_filename (string): Name of target file to write these parameters to
    """

    # Check if input parameter file exists
    try:
        open(input_filename, 'r')
    except IOError:
        print("Could not open input parameter file: " + input_filename)
        exit()

    with open(input_filename, 'r') as fi:
        with open(target_filename, 'w+') as fo:
            for line in fi:
                fo.write(line)


def write_stats(t, dt, steps, c_vector, geometry, free_energy, residuals, max_change, target_file):
    """Writes out simulation statistics

    Args:
        t (float): Current time

        dt (float): Size of current time step

        steps (int): Number of time steps taken

        c_vector (numpy.ndarray): An nx1 vector of species concentrations that looks like :math:`[c_1, c_2, ... c_n]`.
        The concentration variables :math:`c_i` must be instances of the class :class:`fipy.CellVariable` or equivalent.

        geometry (Geometry): An instance of class :class:`utils.geometry.Geometry` that contains mesh description
        free_energy (utils.free_energy): An instance of one of the free energy classes present in
        :mod:`utils.free_energy`

        residuals (float): Largest value of residual when solving the dynamical equations at this current time step

        max_change (float): Maximum rate of change of concentration fields at any position

        target_file (string): Target file to write out the statistics
    """

    # Write out header of the stats file
    if steps == 0:
        # Header of the stats file
        stats_list = ['step', 't', 'dt']
        for i in range(len(c_vector)):
            stats_list.append('c_{index}_avg'.format(index=i))
            stats_list.append('c_{index}_min'.format(index=i))
            stats_list.append('c_{index}_max'.format(index=i))
        stats_list += ['residuals', 'max_rate_of_change', 'free_energy']
        # Write out the header to the file
        with open(target_file, 'w+') as stats:
            stats.write("\t".join(stats_list) + "\n")

    # Write out simulation statistics to the stats file
    stats_simulation = ["{}".format(int(steps)),
                        "{:.8f}".format(t),
                        "{:.3e}".format(dt)]
    for i in range(len(c_vector)):
        stats_simulation.append("{:.8f}".format(c_vector[i].cellVolumeAverage.value))
        stats_simulation.append("{:.8f}".format(min(c_vector[i].value)))
        stats_simulation.append("{:.8f}".format(max(c_vector[i].value)))

    stats_simulation.append("{:.8f}".format(float(residuals)))
    stats_simulation.append("{:.8f}".format(float(max_change)))
    stats_simulation.append("{:.8f}".format(np.sum((free_energy.calculate_fe(c_vector)
                                                    * geometry.mesh.cellVolumes).value)))

    with open(target_file, 'a') as stats:
        stats.write("\t".join(stats_simulation) + "\n")


def write_spatial_variables_to_hdf5_file(step, total_steps, c_vector, geometry, free_energy, target_file):
    """Function to write out the concentration fields and chemical potentials to a hdf5 file

    Args:
        step (int): The step number to write out the spatial variables data

        total_steps (int): Total number of snapshots at which we need to store the concentration fields

        c_vector (numpy.ndarray): An nx1 vector of species concentrations that looks like :math:`[c_1, c_2, ... c_n]`.
        The concentration variables :math:`c_i` must be instances of the class :class:`fipy.CellVariable` or equivalent.

        geometry (Geometry): An instance of class :class:`utils.geometry.Geometry` that contains mesh description

        free_energy (utils.free_energy): An instance of one of the free energy classes present in
        :mod:`utils.free_energy`

        target_file (string): Target file to write out the statistics
    """

    # Create the list of variable names to store. We are going to store the concentration fields and the chemical
    # potentials
    list_of_spatial_variables = []
    for i in range(len(c_vector)):
        list_of_spatial_variables.append("c_{index}".format(index=i))
        list_of_spatial_variables.append("mu_{index}".format(index=i))

    # Create the HDF5 file if it doesn't exist
    if step == 0:
        number_of_mesh_points = np.shape(c_vector)[1]
        with h5py.File(target_file, 'w') as f:
            for sv in list_of_spatial_variables:
                f.create_dataset(sv, (total_steps, number_of_mesh_points))

    # Write out simulation data to the HDF5 file
    with h5py.File(target_file, 'a') as f:
        mu_vector = free_energy.calculate_mu(c_vector)
        for i in range(len(c_vector)):
            f["c_{index}".format(index=i)][step, :] = c_vector[i].value
            f["mu_{index}".format(index=i)][step, :] = mu_vector[i].value
