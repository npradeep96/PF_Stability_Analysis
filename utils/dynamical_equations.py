"""Module that assembles the model equations for spatiotemporal dynamics of concentration fields.
"""

import fipy as fp
import numpy as np
import reaction_rates as rates


class TwoComponentModel:
    """Two component system, with Model B for species 1 and Model AB or reaction-diffusion with reactions for species 2

    This class describes the spatiotemporal dynamics of concentration fields two component system given by the below
    expression:

    .. math::

        \\partial c_1 / \\partial t = \\nabla (M_1 \\nabla \\mu_1 (c_1, c_2))

        \\partial c_2 / \\partial t = \\nabla (M_2 \\nabla \\mu_2 (c_1, c_2)) + k_1 c_1 - k_2 c_2

        (or)

        \\partial c_2 / \\partial t = \\nabla (M_2 \\nabla c_2) + k_1 c_1 - k_2 c_2

    Species 1 relaxes via Model B dynamics, with a mobility coefficient :math:`M_1`. It's total amount in the domain is
    conserved.

    Species 2 undergoes a Model AB or reaction-diffusion dynamics. Detailed balance is broken in this equation.
    It's mobility coefficient is :math:`M_2` and is produced by species 1 with a rate constant :math:`k_1` and degrades
    with a rate constant :math:`k_2`
    """

    def __init__(self, mobility_1, mobility_2, modelAB_dynamics_type, degradation_constant, free_energy, c_vector):
        """Initialize an object of :class:`TwoComponentModelBModelAB`.

        Args:
            mobility_1 (float): Mobility of species 1

            mobility_2 (float): Mobility of species 2

            modelAB_dynamics_type (integer): If = 1, Model AB dynamics. If = 2, reaction-diffusion for species 2.

            degradation_constant (float): Rate constant for first-order degradation of species 2

            free_energy: An instance of one of the free energy classes present in :mod:`utils.free_energy`

            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`.

            The concentration variables :math:`c_1` and :math:`c_2` must be instances of the class
            :class:`fipy.CellVariable`
        """

        # Parameters of the dynamical equations
        self._M1 = mobility_1
        self._M2 = mobility_2
        self._modelAB_dynamics_type = modelAB_dynamics_type
        self._free_energy = free_energy
        # Define the reaction terms in the model equations
        self._production_term = None
        self._degradation_term = rates.FirstOrderReaction(k=degradation_constant)
        # Define model equations
        self._equations = None # self._model_equations(c_vector)

    def set_production_term(self, reaction_type, **kwargs):
        """ Sets the nature of the production term of species :math:`c_2` from :math:`c_1`

        If reaction_type == 1: First order reaction with rate constant uniform in space.

        If reaction_type == 2: First order reaction with rate constant Gaussian in space. This requires the parameter
        sigma (width of Gaussian), coordinates of the center point of Gaussian, and the mesh geometry as optional input
        arguments to compute rate constant at every position in space.

        Args:
            reaction_type (integer): An integer value describing what reaction type to consider.
        """
        if reaction_type == 1:
            # Reaction rate constant is uniform in space
            rate_constant = kwargs.get('rate_constant', None)
            self._production_term = rates.FirstOrderReaction(k=rate_constant)
        elif reaction_type == 2:
            # Reaction rate constant is Gaussian in space
            rate_constant = kwargs.get('rate_constant', None)
            sigma = kwargs.get('sigma', None)
            center_point = kwargs.get('center_point', None)
            geometry = kwargs.get('geometry', None)
            self._production_term = rates.LocalizedFirstOrderReaction(k=rate_constant, sigma=sigma, x0=center_point,
                                                                      simulation_geometry=geometry)

    def set_model_equations(self, c_vector):
        """Assemble the model equations given a mesh and concentrations

        This functions assembles the model equations necessary

        Args:
            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`.
            The concentration variables :math:`c_1` and :math:`c_2` must be instances of the class
            :class:`fipy.CellVariable`

        Assigns:
            self._equations (list): List that would go to 0 if the concentrations in c_vector satisfy the model equation
        """

        # Get Jacobian matrix associated with the free energy. This gives us the coefficients that multiply the
        # gradients of the concentration fields in the Model B dynamics.
        assert hasattr(self._free_energy, 'calculate_jacobian'), \
            "self._free_energy instance does not have a function calculate_jacobian()"
        assert hasattr(self._free_energy, '_kappa'), \
            "self._free_energy instance does not have an attribute kappa describing the surface energy"
        jacobian = self._free_energy.calculate_jacobian(c_vector)

        # Model B dynamics for species 1
        eqn_1 = (fp.TransientTerm(coeff=1.0, var=c_vector[0])
                 == fp.DiffusionTerm(coeff=self._M1 * jacobian[0, 0], var=c_vector[0])
                 + fp.DiffusionTerm(coeff=self._M1 * jacobian[0, 1], var=c_vector[1])
                 - fp.DiffusionTerm(coeff=(self._M1, self._free_energy.kappa), var=c_vector[0])
                 - self._M1 * (self._free_energy.get_gaussian_function(c_vector[0].mesh)).faceGrad.divergence
                 )
        # Model AB dynamics or reaction-diffusion dynamics for species 2 with production and degradation reactions
        if self._modelAB_dynamics_type == 1:
            # Model AB dynamics for species 2
            eqn_2 = (fp.TransientTerm(coeff=1.0, var=c_vector[1])
                     == fp.DiffusionTerm(coeff=self._M2 * jacobian[1, 0], var=c_vector[0])
                     + fp.DiffusionTerm(coeff=self._M2 * jacobian[1, 1], var=c_vector[1])
                     + self._production_term.rate(c_vector[0])
                     - self._degradation_term.rate(c_vector[1])
                     )
        elif self._modelAB_dynamics_type == 2:
            # Reaction-diffusion dynamics for species 2
            eqn_2 = (fp.TransientTerm(coeff=1.0, var=c_vector[1])
                     == fp.DiffusionTerm(coeff=self._M2 * jacobian[1, 1], var=c_vector[1])
                     + self._production_term.rate(c_vector[0])
                     - self._degradation_term.rate(c_vector[1])
                     )

        self._equations = [eqn_1, eqn_2]

    def step_once(self, c_vector, dt, max_sweeps):
        """Function that solves the model equations over a time step of dt to get the concentration profiles.

        Args:
            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`.
            The concentration variables :math:`c_1` and :math:`c_2` must be instances of the class
            :class:`fipy.CellVariable`

            dt (float): Size of time step to solve the model equations over once
            max_sweeps (int): Number of times to sweep using the function sweep() in the fipy package

        Returns:
            residuals (numpy.ndarray): A 2x1 numpy array containing residuals after solving the equations

            max_change (float): Maximum change in the concentration fields at any given position for the time interval
            dt

        """

        # Solve the model equations for a time step of dt by sweeping max_sweeps times
        sweeps = 0
        residual_1 = 1e6
        residual_2 = 1e6
        while sweeps < max_sweeps:
            residual_1 = self._equations[0].sweep(dt=dt)
            residual_2 = self._equations[1].sweep(dt=dt)
            sweeps += 1
        residuals = np.array([residual_1, residual_2])

        max_change_c_1 = np.max(np.abs((c_vector[0] - c_vector[0].old).value))
        max_change_c_2 = np.max(np.abs((c_vector[1] - c_vector[1].old).value))
        max_change = np.max([max_change_c_1, max_change_c_2])

        return residuals, max_change
