"""Module that contains classes describing different free energies.
"""

import numpy as np


class TwoCompDoubleWellFHCrossQuadratic:
    """Free energy of two component system with a quartic well and quadratic well self, and FH cross interactions.

    This class describes the free energy of a two component system given by the below expression:

    .. math::

       f[c_1, c_2] = 0.25 \\alpha (c_1-\\bar{c}_1)^4 + 0.5 \\beta (c_1-\\bar{c}_1)^2 + \\gamma c_1 c_2
                     + 0.5 \\lambda c^2_2 + 0.5 \\kappa |\\nabla c_1|^2

    Interactions between molecules of species 1 are described by a quartic-well potential. If :math:`\\beta < 0`, then
    we get a double-well and species 1 can phase separate by itself.

    Interactions between molecules of species 2 are described by a quadratic potential. For this term, :math:`\\lambda`
    has to be > 0. Otherwise, the self diffusion of species 2 will cause all molecules to collapse into a point.

    The cross interactions between the species are described by a mean-field product of concentrations with the
    interaction strength captured by a Flory parameter :math:`\\gamma`
    """

    def __init__(self, alpha, beta, gamma, lamda, kappa, c_bar_1):
        """Initialize an object of :class:`TwoCompDoubleWellFHCrossQuadratic`.

        Args:
            alpha (float): Parameter associated with the quartic term :math:`\\alpha (c_1-\\bar{c}_1)^4` of species 1
            beta (float): Parameter associated with the quadratic term :math:`\\beta (c_1-\\bar{c}_1)^2` of species 1
            gamma (float): Parameter that describes the cross-interactions between the species :math:`\\gamma c_1 c_2`
            lamda (float): Parameter that describes the self interaction of species 2 using :math:`\\lambda c^2_2`
            kappa (float): Parameter that describes the surface tension associated with species 1
                           :math:`\\kappa/2 |\\nabla c_1|^2`
            c_bar_1 (float): Critical concentration of species 1 at the onset of phase separation
        """

        # Ensure that the parameter lambda is always positive
        # Otherwise, we will get nonsense results in the simulations
        assert lamda > 0, "The parameter lambda is negative. Please supply a positive value"

        # Assign all free energy parameters to private variables
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._lambda = lamda
        self._kappa = kappa
        self._c_bar_1 = c_bar_1

    @property
    def kappa(self):
        """Getter for the private variable self._kappa.
        This is used to set up the surface tension term in the dynamical equations"""
        return self._kappa

    def calculate_fe(self, c_vector):
        """Calculate free energy according to the expression in class description.

        Args:
            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`.
                                      The concentration variables :math:`c_1` and :math:`c_2` must be instances of the
                                      class :class:`fipy.CellVariable` or equivalent. These instances should have an
                                      attribute called :attr:`.grad.mag` that returns the magnitude of gradient of the
                                      concentration field for every position in the mesh to compute the surface tension
                                      contribution of the free energy

        Returns:
            free_energy (float): Free energy value
        """

        # Check that c_vector satisfies the necessary conditions
        assert len(c_vector) == 2, \
            "The shape of c_vector passed to TwoCompDoubleWellFHCrossQuadratic.calculate_fe() is not 2x1"
        assert hasattr(c_vector[0], "grad"), \
            "The instance c_vector[0] has no attribute grad associated with it"
        assert hasattr(c_vector[1], "grad"), \
            "The instance c_vector[1] has no function grad associated with it"
        assert hasattr(c_vector[0].grad, 'mag'), \
            "The instance c_vector[0].grad has no attribute mag associated with it"
        assert hasattr(c_vector[1].grad, 'mag'), \
            "The instance c_vector[1].grad has no attribute mag associated with it"

        # Calculate the free energy
        fe = self._alpha / 4.0 * (c_vector[0] - self._c_bar_1) ** 4 \
             + self._beta / 2.0 * (c_vector[0] - self._c_bar_1) ** 2 \
             + self._gamma * c_vector[0] * c_vector[1] \
             + self._lambda * c_vector[1] ** 2 \
             + 0.5 * self._kappa * c_vector[1].grad.mag ** 2

        return fe

    def calculate_mu(self, c_vector):
        """Calculate chemical potential of the species.

        Chemical potential of species 1:

        .. math::

            \\mu_1[c_1, c_2] = \\delta F / \\delta c_1 = \\alpha (c_1-\\bar{c}_1)^3 + \\beta (c_1-\\bar{c}_1)
                               + \\gamma c_2 - \\kappa \\nabla^2 c_1

        Chemical potential of species 2:

        .. math::

            \\mu_2[c_1, c_2] = \\gamma c_1 + \\lambda c_2


        Args:
            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`.
                                      The concentration variables :math:`c_1` and :math:`c_2` must be instances of the
                                      class :class:`fipy.CellVariable` or equivalent. These instances should have an
                                      attribute called :attr:`.faceGrad.divergence` that returns the Laplacian of the
                                      concentration field for every position in the mesh to compute the surface tension
                                      contribution to the chemical potential of species 1

        Returns:
            mu (list): A 2x1 vector of chemical potentials that looks like :math:`[mu_1, mu_2]`
        """

        # Check that c_vector satisfies the necessary conditions
        assert len(c_vector) == 2, \
            "The shape of c_vector passed to TwoCompDoubleWellFHCrossQuadratic.calculate_mu() is not 2x1"
        assert hasattr(c_vector[0], "faceGrad"), \
            "The instance c_vector[0] has no attribute faceGrad associated with it"
        assert hasattr(c_vector[1], "faceGrad"), \
            "The instance c_vector[1] has no attribute faceGrad associated with it"
        assert hasattr(c_vector[0].faceGrad, "divergence"), \
            "The instance c_vector[0].faceGrad has no attribute divergence associated with it"
        assert hasattr(c_vector[1].faceGrad, "divergence"), \
            "The instance c_vector[1].faceGrad has no attribute divergence associated with it"

        # Calculate the chemical potentials
        mu_1 = self._alpha * (c_vector[0] - self._c_bar_1) ** 3 \
               + self._beta * (c_vector[0] - self._c_bar_1) \
               + self._gamma * c_vector[1] \
               - self._kappa * c_vector[0].faceGrad.divergence
        mu_2 = self._gamma * c_vector[0] + self._lambda * c_vector[1]
        mu = [mu_1, mu_2]

        return mu

    def calculate_jacobian(self, c_vector):
        """Calculate the Jacobian matrix of coefficients to feed to the transport equations.

        In calculating the Jacobian, we ignore the surface tension and any spatially dependent terms and only take the
        bulk part of the free energy that depends on the concentration fields:

        .. math::

            J[c_1, c_2] = \\begin{bmatrix}
                          \\delta F_{bulk} / \\delta c^2_1 & \\delta F_{bulk} / \\delta c_1 \\delta c_2 \\\
                          \\delta F_{bulk} / \\delta c_1 \\delta c_2 & \\delta F_{bulk} / \\delta c^2_2
                          \\end{bmatrix}
                        = \\begin{bmatrix}
                          3 \\alpha (c_1 - \\bar{c}_1)^2 + \\beta & \\gamma \\\
                          \\gamma & \\lambda
                          \\end{bmatrix}

        Args:
            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`.
                                      The concentration variables :math:`c_1` and :math:`c_2` must be instances of the
                                      class :class:`fipy.CellVariable` or equivalent
        Returns:
            jacobian (numpy.ndarray): A 2x2 Jacobian matrix, with each entry itself being a vector of the same size as
                                      c_vector[0]
        """

        # Check that c_vector satisfies the necessary conditions
        assert len(c_vector) == 2, \
            "The shape of c_vector passed to TwoCompDoubleWellFHCrossQuadratic.calculate_mu() is not 2x1"

        # Calculate the Jacobian matrix
        jacobian = np.array([[3 * self._alpha * (c_vector[0] - self._c_bar_1) ** 2 + self._beta, self._gamma],
                            [self._gamma, self._lambda]])
        return jacobian
