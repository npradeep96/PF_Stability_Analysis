"""Main script to generate Fourier spectrum at each time point for the conserved and non-conserved species
concentration profiles
"""

# plot a spectrum. Smallest largest values of kx and ky are 2*pi/dx
# Input:
# 1. A discrete mesh of coordinates
# 2. A discrete set of concentration points
# 3. The correct basis function is used to weight the fourier transform coefficients
# 4. Create a discrete space of kx and ky
