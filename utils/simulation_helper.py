"""Module that contains helper functions to run simulations that can be used by run_simulation.py
"""

import geometry
import initial_conditions
import free_energy
import dynamical_equations
import fipy as fp


def set_mesh_geometry(input_params):
    """Set the mesh geometry depending on the options in input_parameters

    Args:
        input_params (dict): Dictionary that contains input parameters. We are only interested in the key,value
                             pairs that describe the mesh geometry

    Returns:
         simulation_geometry (Geometry): Instance of class :class:`utils.geometry.Geometry`
    """

    simulation_geometry = geometry.Geometry()

    # Geometry in 2 dimensions
    if input_params['dimension'] == 2:
        # Circular geometry
        if input_params['circ_flag'] == 1:
            assert 'radius' in input_params.keys() and 'dx' in input_params.keys(), \
                "input_params dictionary doesn't have values corresponding to the domain radius and mesh size"
            simulation_geometry.circular_mesh_2d(radius=input_params['radius'], cell_size=input_params['dx'])
        # Square geometry
        else:
            assert 'length' in input_params.keys() and 'dx' in input_params.keys(), \
                "input_params dictionary doesn't have values corresponding to the domain length and mesh size"
            simulation_geometry.square_mesh_2d(length=input_params['length'], dx=input_params['dx'])
    # Geometry in 3 dimensions
    elif input_params['dimension'] == 3:
        assert 'length' in input_params.keys() and 'dx' in input_params.keys(), \
            "input_params dictionary doesn't have values corresponding to the domain length and mesh size"
        simulation_geometry.cube_mesh_3d(length=input_params['length'], dx=input_params['dx'])

    return simulation_geometry


def initialize_concentrations(input_params, simulation_geometry):
    """Set initial conditions for the concentration profiles

    Args:
        input_params (dict): Dictionary that contains input parameters. We are only interested in the key,value
                             pairs that describe the initial conditions
        simulation_geometry (Geometry): Instance of class :class:`utils.geometry.Geometry` that describes the mesh
                                        geometry

    Returns:
        concentration_vector (numpy.ndarray): An nx1 vector of species concentrations that looks like
                                             :math:`[c_1, c_2, ... c_n]`. The concentration variables :math:`c_i` must
                                             be instances of the class :class:`fipy.CellVariable` or equivalent.
    """

    # Initialize concentration_vector
    concentration_vector = []

    for i in range(int(input_params['n_concentrations'])):
        # Initialize fipy.CellVariable
        concentration_variable = fp.CellVariable(mesh=simulation_geometry.mesh, name='c_{index}'.format(index=i),
                                                 hasOld=True, value=input_params['initial_values'][i])
        # Nucleate a seed of dense concentrations if necessary
        if input_params['nucleate_seed'][i] == 1:
            initial_conditions.nucleate_spherical_seed(concentration=concentration_variable,
                                                       value=input_params['seed_value'][i],
                                                       dimension=input_params['dimension'],
                                                       geometry=simulation_geometry,
                                                       nucleus_size=input_params['nucleus_size'][i],
                                                       location=input_params['location'][i])
        # Append the concentration variable to the
        concentration_vector.append(concentration_variable)
    return concentration_vector


def set_free_energy(input_params):
    """Set free energy of interactions

    Args:
        input_params (dict): Dictionary that contains input parameters. We are only interested in the key,value
                             pairs that describe the free energy

    Returns:
        free_en (utils.free_energy): An instance of one of the classes in mod:`utils.free_energy`
    """

    if input_params['free_energy_type'] == 1:
        free_en = free_energy.TwoCompDoubleWellFHCrossQuadratic(alpha=input_params['alpha'],
                                                                beta=input_params['beta'],
                                                                gamma=input_params['gamma'],
                                                                lamda=input_params['lamda'],
                                                                kappa=input_params['kappa'],
                                                                c_bar_1=input_params['c_bar'])
        return free_en


def set_model_equations(input_params, concentration_vector, free_en):
    """Set dynamical equations for the model

    Args:
        input_params (dict): Dictionary that contains input parameters. We are only interested in the key,value
                             pairs that describe the parameters associated with the dynamical model
        concentration_vector (numpy.ndarray): An nx1 vector of species concentrations that looks like
                                             :math:`[c_1, c_2, ... c_n]`. The concentration variables :math:`c_i` must
                                             be instances of the class :class:`fipy.CellVariable` or equivalent.
        free_en (utils.free_energy): An instance of one of the classes in mod:`utils.free_energy`

    Returns:
        equations (utils.dynamical_equations): An instance of one of the classes in mod:`utils.dynamical_equations`
    """

    if input_params['dynamical_model_type'] == 1:
        equations = dynamical_equations.TwoComponentModelBRD(mobility_1=input_params['M1'],
                                                             mobility_2=input_params['M2'],
                                                             rate_constant_1=input_params['k_production'],
                                                             rate_constant_2=input_params['k_degradation'],
                                                             free_energy=free_en,
                                                             c_vector=concentration_vector)
        return equations
