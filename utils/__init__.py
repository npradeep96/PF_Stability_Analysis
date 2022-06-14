""" Module :mod:`utils` contains helper modules to perform phase field simulations

:mod:`utils.geometry`
    Contains functions to help set up the mesh geometry for phase field simulations
:mod:`utils.initial_conditions`
    Contains functions to help set up initial condition of concentration fields for the phase
    field simulations
:mod:`utils.free_energy`
    Contains class definitions of different interaction free energies between molecular species in the
    system
:mod:`utils.reaction_rates`
    Contains class definitions of different forms of reaction rates that can produce or degrade species in the system
:mod:`utils.dynamical_equations`
    Contains class definitions to (i) implement different kinds of spatio-temporal dynamics of concentrations of the
    different molecular species (ii) numerical methods to solve them
:mod:`utils.file_operations`
    Contains functions to read and write files in the course of phase field simulations
:mod:`utils.simulation_helper`
    Contains helper functions to run simulations that can be used by run_simulation.py
"""