"""Main script to assemble and run phase field simulations
"""

import sys
sys.path.append('../')
import argparse
import utils.file_operations as file_operations
import utils.simulation_helper as simulation_helper
import os
import os.path
import numpy as np
from tqdm import tqdm


def get_output_dir_name(input_params):
    """Set dynamical equations for the model

    Args:
        input_params (dict): Dictionary that contains input parameters

    Returns:
        output_dir (string): Name of the output directory including the important parameter names
    """

    output_dir = 'M_1_' + str(input_params['M1']) + '_beta_' + str(input_params['beta_tilde']) \
                 + '_gamma_' + str(input_params['gamma_tilde']) + '_kappa_' + str(input_params['kappa_tilde']) \
                 + '_K_' + str(input_params['basal_k_production']) \
                 + '_K_spatial_' + str(input_params['k_production']) \
                 + '_c_initial_' + str(input_params['initial_values'][0]) + '_noise_variance_' \
                 + str(input_params['initial_condition_noise_variance'][0]) \
                 + '_reaction_sigma_' + str(input_params['reaction_sigma'])
    # + '_well_depth_' + str(input_params['well_depth'])
    # + '_reaction_sigma_' + str(input_params['reaction_sigma'])
    return output_dir


def run_simulation(input_params, concentration_vector, simulation_geometry, free_en, equations, out_directory):
    """Integrate the dynamical equations for concentrations and write to files

    Args:
        input_params (dict): Dictionary that contains input parameters. We are interested in the parameters associated
        with the numerical method for integration

        concentration_vector (numpy.ndarray): An nx1 vector of species concentrations that looks like
        :math:`[c_1, c_2, ... c_n]`. The concentration variables :math:`c_i` must be instances of the class
        :class:`fipy.CellVariable` or equivalent.

        simulation_geometry (Geometry): Instance of one of the classes in :mod:`utils.geometry` that describes the
        mesh geometry.

        free_en (utils.free_energy): An instance of one of the classes in mod:`utils.free_energy`

        equations (utils.dynamical_equations): An instance of one of the classes in mod:`utils.dynamical_equations`
        out_directory (string): The directory to output simulation data

    Returns:
        err_flag (boolean): Whether the simulation has run successfully without any errors
    """

    # Simple time stepping over time interval dt
    # We increase the time step size upto a value of dt_max if the maximum change in the concentration variables is
    # small enough

    # Initial time step parameters for simulation
    dt = input_params['dt']
    dt_max = input_params['dt_max']
    dt_min = input_params['dt_min']
    max_change_allowed = input_params['max_change_allowed']
    duration = int(input_params['duration'])
    total_steps = int(input_params['total_steps'])
    max_sweeps = int(input_params['max_sweeps'])
    max_residual = float(input_params['max_residual'])
    data_log_frequency = int(input_params['data_log'])
    pbar = tqdm(total=total_steps)

    # Start time stepping
    step = 0
    t = 0
    elapsed = 0
    err_flag = 0

    # Check if we have a time profile of parameters to implement
    time_profile_flag = 0
    if len(input_params['time_profile']) != 0:
        time_profile_flag = 1
        number_of_transitions_in_profile = len(input_params['time_profile'])
        transition_counter = 0

    while (elapsed <= duration) and (step <= total_steps):

        # Check if we need to change parameters to implement the time profile of parameters
        if time_profile_flag:
            # Reset parameters when the threshold time is reached
            if elapsed > input_params['time_profile'][transition_counter]['transition_time']:
                # Update the input parameter values
                for key, val in input_params['time_profile'][transition_counter].items():
                    input_params[key] = float(val)
                # Update model equations
                free_en = simulation_helper.set_free_energy(input_parameters)
                equations = simulation_helper.set_model_equations(input_params=input_params,
                                                                  concentration_vector=concentration_vector,
                                                                  free_en=free_en,
                                                                  simulation_geometry=simulation_geometry)
                # Update transition counter to reflect that a transition has happened
                transition_counter = transition_counter + 1
                # If we have completed all parameter transitions, stop implementing the time profile
                if transition_counter == number_of_transitions_in_profile:
                    time_profile_flag = 0

        # Update the old values of concentrations
        equations.update_old(concentration_vector)

        has_converged = False
        # Step over a time step dt and solve the equations
        while dt > dt_min:
            has_converged, residuals, max_change = equations.step_once(c_vector=concentration_vector, dt=dt,
                                                                       max_residual=max_residual, max_sweeps=max_sweeps)
            if not has_converged:
                dt *= 0.5
                continue
            else:
                break

        if dt <= dt_min:
            err_flag = 1
            break

        # Write simulation output to files
        if step % data_log_frequency == 0:
            file_operations.write_stats(t=t, dt=dt, steps=step, c_vector=concentration_vector,
                                        geometry=simulation_geometry, free_energy=free_en,
                                        residuals=np.max(residuals), max_change=np.max(max_change),
                                        target_file=os.path.join(out_directory, 'stats.txt'))
            file_operations.write_spatial_variables_to_hdf5_file(step=int(step / data_log_frequency),
                                                                 total_steps=int(total_steps / data_log_frequency) + 1,
                                                                 c_vector=concentration_vector,
                                                                 geometry=simulation_geometry,
                                                                 free_energy=free_en,
                                                                 target_file=os.path.join(out_directory,
                                                                                          'spatial_variables.hdf5'))
        # Update all the variables that keep track of time
        step += 1
        elapsed += dt
        t += dt
        pbar.update(n=1)

        # Increase time step if converged
        dt *= 1.1
        dt = min(dt, dt_max)

    return err_flag


if __name__ == "__main__":
    """This script assembles and runs phase field simulations using helper functions defined in this file
    """

    # Read command line arguments that describe file containing input parameters and folder to output simulation results
    parser = argparse.ArgumentParser(description='Input parameter file and output directory are command line arguments')
    parser.add_argument('--i', help="Name of input parameter file", required=True)
    parser.add_argument('--o', help="Name of output directory", required=True)
    args = parser.parse_args()
    input_parameter_file = args.i

    # Read input parameters from file
    input_parameters = file_operations.input_parse(filename=input_parameter_file)
    print('Successfully parsed input parameters ...')

    # Set mesh geometry
    sim_geometry = simulation_helper.set_mesh_geometry(input_params=input_parameters)
    print('Successfully set up mesh geometry ...')

    # Initialize concentration variables and initial conditions
    c_vector = simulation_helper.initialize_concentrations(input_params=input_parameters,
                                                           simulation_geometry=sim_geometry)
    print('Successfully initialized concentration vectors ...')

    # Set free energy
    fe = simulation_helper.set_free_energy(input_parameters)
    print('Successfully set up the free energy ...')

    # Choose the model equations
    model_equations = simulation_helper.set_model_equations(input_params=input_parameters,
                                                            concentration_vector=c_vector,
                                                            free_en=fe,
                                                            simulation_geometry=sim_geometry)
    print('Successfully set up model equations ...')

    # Create the output directory
    output_directory = os.path.join(args.o, get_output_dir_name(input_parameters))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Write the input parameters file to the output directory
    file_operations.write_input_params_from_file(input_filename=input_parameter_file,
                                                 target_filename=os.path.join(output_directory, 'input_params.txt'))
    print('Successfully created the output directory to write simulation data ...')

    # Run simulation
    print('Running simulation ...')
    error_flag = run_simulation(input_params=input_parameters,
                                concentration_vector=c_vector,
                                simulation_geometry=sim_geometry,
                                free_en=fe,
                                equations=model_equations,
                                out_directory=output_directory)

    if error_flag:
        print("There were some numerical issues in the simulations. Try reducing the minimum step size in time, or " +
              "try for a different range of parameters")
