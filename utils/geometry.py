"""Module that contains functions to set up discrete spatial mesh for simulations
"""

import fipy as fp
from fipy import Gmsh2D
import numpy as np


class Geometry(object):
    """Class that describes a mesh geometry and some associated operations with the mesh grid points.

    This is a base class for the different kinds of mesh geometries used in the simulations.
    """

    def __init__(self, mesh=None):
        """Initialize the Geometry object, which initializes an attribute called mesh if available.

        Args:
            mesh (fipy.meshes.mesh): A fipy mesh variable. Default value is None.
        """
        self.mesh = mesh

    def get_mesh_distances_squared_from_point(self, reference_point):
        """ Function that calculates the squared distance of each mesh point from a reference point.

        Args:
            reference_point (numpy.ndarray): An nx1 vector containing coordinates of the reference point

        Returns:
             squared_distances (fipy.variable): A fipy variable that stores the distances of each mesh point from the
             reference point.
        """
        # check if the reference point actually has 2 dimensions

        assert np.shape(reference_point)[0] == np.shape(self.mesh.cellCenters)[0], \
            "The reference point isn't in the same dimensional space as the mesh"
        try:
            squared_distances = fp.CellVariable(mesh=self.mesh,
                                                value=np.sum(((self.mesh.cellCenters - reference_point) ** 2).value, 0))
            return squared_distances
        except AttributeError:
            print('self.mesh is expected to be a fipy.meshes.mesh variable. It does not have an attribute cellCenters')


class CircularMesh2d(Geometry):
    """Class to create a 2D circular mesh derived from the base class Geometry.

    This class is defined by two parameters - radius of the circle and cell size.
    """

    def __init__(self, radius, cell_size):
        """Initialize a circular 2D mesh object depending on the radius and cell size. This uses the function Gmsh2D()

        Args:
            radius (float): Radius of the total domain

            cell_size (float): Side length of a discrete mesh element
        """
        # Initialize base class Geometry
        Geometry.__init__(self)
        # Construct a circular mesh
        self.mesh = Gmsh2D('''   cell_size = %g;
                                 radius = %g;
                                 Point(1) = {0, 0, 0, cell_size};
                                 Point(2) = {-radius, 0, 0, cell_size};
                                 Point(3) = {0, radius, 0, cell_size};
                                 Point(4) = {radius, 0, 0, cell_size};
                                 Point(5) = {0, -radius, 0, cell_size};
                                 Circle(6) = {2, 1, 3};
                                 Circle(7) = {3, 1, 4};
                                 Circle(8) = {4, 1, 5};
                                 Circle(9) = {5, 1, 2};
                                 Line Loop(10) = {6, 7, 8, 9};
                                 Plane Surface(11) = {10};
                              ''' % (cell_size, radius))


class SquareMesh2d(Geometry):
    """Class to create a 2D square mesh derived from the base class Geometry.

    This class is defined by two parameters - edge length of the square and size of mesh element.
    """

    def __init__(self, length, dx):
        """Initialize a square 2D mesh object depending on the length of the square and the size of the mesh element.

        This function uses the Grid2D function from the fipy package to create a 2d square mesh

        Args:
            length (float): Length of the square domain

            dx (float): Side length of a discrete mesh element in the square domain
        """
        # Initialize base class Geometry
        Geometry.__init__(self)
        # Construct a square mesh
        nx = int(length / dx)
        self.mesh = fp.Grid2D(nx=nx, ny=nx, dx=dx, dy=dx)
        # Center the mesh at (0,0)
        self.mesh = self.mesh - float(nx) * dx * 0.5


class CubeMesh3d(Geometry):
    """Class to create a 3D cubical mesh derived from the base class Geometry.

    This class is defined by two parameters - edge length of the cube and size of mesh element.
    """

    def __init__(self, length, dx):
        """Initialize a 3D cubical mesh depending on the length and size of mesh element

        This function uses the Grid3D function from the fipy package to create a 3d cubical mesh

        Args:
            length (float): Length of the cubical domain

            dx (float): Side length of a discrete mesh element in the cubical domain

        Returns:
            mesh (Grid3D): A 3d cubical mesh that is an instance of fipy.Grid3D
        """
        # Initialize base class Geometry
        Geometry.__init__(self)
        # Construct a square mesh
        nx = int(length / dx)
        self.mesh = fp.Grid3D(nx=nx, ny=nx, nz=nx, dx=dx, dy=dx, dz=dx)
        # Center the mesh at (0,0)
        self.mesh = self.mesh - float(nx) * dx * 0.5
