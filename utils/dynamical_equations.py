"""Module that assembles the model equations for spatiotemporal dynamics of concentration fields.
"""

import fipy as fp
import numpy as np
import reaction_rates as rates


class TwoComponentModelBRD:
    """Two component system, with Model B for species 1 and Model AB with reactions for species 2

    This class describes the spatiotemporal dynamics of concentration fields two component system given by the below
    expression:

    .. math::

        \\partial c_1 / \\partial t = \\nabla (M_1 \\nabla \\mu_1 (c_1, c_2))

        \\partial c_2 / \\partial t = \\nabla (M_2 \\nabla \\mu_2 (c_1, c_2)) + k_1 c_1 - k_2 c_2

    Species 1 relaxes via Model B dynamics, with a mobility coefficient :math:`M_1`. It's total amount in the domain is
    conserved.

    Species 2 undergoes a Model AB dynamics. Detailed balance is broken in this equation. It's mobility coefficient is
    :math:`M_2` and is produced by species 1 with a rate constant :math:`k_1` and degrades with a rate constant
    :math:`k_2`
    """

    def __init__(self, mobility_1, mobility_2, rate_constant_1, rate_constant_2, free_energy, c_vector):
        """Initialize an object of :class:`TwoComponentModelBModelAB`.

        Args:
            mobility_1 (float): Mobility of species 1

            mobility_2 (float): Mobility of species 2

            rate_constant_1 (float): Rate constant of production of species 2 by species 1

            rate_constant_2 (float): Rate constant for first-order degradation of species 2

            free_energy: An instance of one of the free energy classes present in :mod:`utils.free_energy`

            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`.

            The concentration variables :math:`c_1` and :math:`c_2` must be instances of the class
            :class:`fipy.CellVariable`
        """

        # Parameters of the dynamical equations
        self._M1 = mobility_1
        self._M2 = mobility_2
        self._free_energy = free_energy
        # Define the reaction terms in the model equations
        self._production_term = rates.FirstOrderReaction(rate_constant_1)
        self._degradation_term = rates.FirstOrderReaction(rate_constant_2)
        # Define model equations
        self._equations = self._model_equations(c_vector)

    def _model_equations(self, c_vector):
        """Assemble the model equations given a mesh and concentrations

        This functions assembles the model equations necessary

        Args:
            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`.
            The concentration variables :math:`c_1` and :math:`c_2` must be instances of the class
            :class:`fipy.CellVariable`

        Returns:
            equations (list): List that would go to 0 if the concentrations in c_vector satisfy the model equations
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
                 )
        # Model AB dynamics for species 2 with production and degradation reactions
        eqn_2 = (fp.TransientTerm(coeff=1.0, var=c_vector[1])
                 == fp.DiffusionTerm(coeff=self._M2 * jacobian[1, 1], var=c_vector[1])
                 + self._production_term.rate(c_vector[0])
                 - self._degradation_term.rate(c_vector[1])
                 )

        equations = [eqn_1, eqn_2]
        return equations

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
