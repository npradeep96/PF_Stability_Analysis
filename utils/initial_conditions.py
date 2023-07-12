"""Module that contains helper functions to initialize concentration fields

"""

import fipy as fp
import numpy as np


def initialize_uniform_profile(c_vector, values):
    """Function that initializes a spatially uniform value for concentration variables in c_vector

    Args:
        c_vector (numpy.ndarray): An nx1 vector of species concentrations that looks like :math:`[c_1, c_2, ... c_n]`.
        The concentration variables :math:`c_i` must be instances of the class :class:`fipy.CellVariable` or equivalent.

        values (numpy.ndarray): An nx1 vector of values to set the concentration fields to
    """

    for i in range(len(c_vector)):
        if type(c_vector[i]) == fp.variables.cellVariable.CellVariable:
            c_vector[i].value = values[i]


def add_noise_to_initial_conditions(c_vector, sigmas, random_seed):
    """Function that initializes a spatially uniform value for concentration variables in c_vector

    Args:
        c_vector (numpy.ndarray): An nx1 vector of species concentrations that looks like :math:`[c_1, c_2, ... c_n]`.
        The concentration variables :math:`c_i` must be instances of the class :class:`fipy.CellVariable` or equivalent.

        sigmas (numpy.ndarray): An nx1 vector of values that captures the variance of the noise term added to the
        initial condition.

        random_seed (int): An integer to seed the random number generator to add noise
    """

    # Seed random number generator
    np.random.seed(random_seed)

    for i in range(len(c_vector)):
        if type(c_vector[i]) == fp.variables.cellVariable.CellVariable:
            number_of_mesh_elements = np.size(c_vector[i].value)
            c_vector[i].value += sigmas[i] * np.random.randn(number_of_mesh_elements)


def nucleate_spherical_seed(concentration, value, dimension, geometry, nucleus_size, location):
    """Function that nucleates a circular or spherical region of high concentration

    Args:
         concentration (fipy.CellVariable or equivalent): A concentration variable

         value (float): The value of concentration within the nucleus

         dimension (int): Can be 1, 2, or 3 corresponding to a 1D, 2D or 3D mesh respectively

         geometry (Geometry): An instance of class :class:`utils.geometry.Geometry` that contains mesh description

         nucleus_size (float): Radius of the circular or spherical nucleus

         location (numpy.ndarray): A vector containing the coordinates of the center of the nucleus relative to origin
    """
    # Ensure that the dimensions are the same as the number of coordinates that describe the center of the nucleus
    assert np.size(location) == dimension, "The location coordinates does not match with the dimensions of the mesh"
    coordinates_of_cells = geometry.mesh.cellCenters.value

    if dimension == 1:
        x_centroid = 0.5 * (min(geometry.mesh.x) + max(geometry.mesh.x))
        x_centroid += location[0]
        distance = np.abs(coordinates_of_cells[0] - x_centroid)
        concentration[distance < nucleus_size] = value
    elif dimension == 2:
        x_centroid = 0.5 * (min(geometry.mesh.x) + max(geometry.mesh.x))
        x_centroid += location[0]
        y_centroid = 0.5 * (min(geometry.mesh.y) + max(geometry.mesh.y))
        y_centroid += location[1]
        distance = np.sqrt((coordinates_of_cells[0] - x_centroid) ** 2
                           + (coordinates_of_cells[1] - y_centroid) ** 2)
        concentration[distance < nucleus_size] = value
    elif dimension == 3:
        x_centroid = 0.5 * (min(geometry.mesh.x) + max(geometry.mesh.x))
        x_centroid += location[0]
        y_centroid = 0.5 * (min(geometry.mesh.y) + max(geometry.mesh.y))
        y_centroid += location[1]
        z_centroid = 0.5 * (min(geometry.mesh.z) + max(geometry.mesh.z))
        z_centroid += location[2]
        distance = np.sqrt((coordinates_of_cells[0] - x_centroid) ** 2
                           + (coordinates_of_cells[1] - y_centroid) ** 2
                           + (coordinates_of_cells[2] - z_centroid) ** 2)
        concentration[distance < nucleus_size] = value
    else:
        print("Dimensions greater than 3 are not supported by the function nucleate_spherical_seed()")
        exit()
