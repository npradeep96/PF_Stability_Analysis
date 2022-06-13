# PF_Stability_Analysis

Use the environment.yml file provided to setup your python environment. 

To run the simulations:

1. Create an input parameter text file under /inputs to define your system. An example file is provided. Please go through the code documentation to understand what each parameter in this file means.
2. Run simulations on the command line using: ``python run_simulation.py --i path/to/input/parameter/file --o path/to/output/directory/to/write/simulation/data``
3. Make movies of your simulations using the script analysis/make_movies.py
4. To make movies, run the following command on the command line: ``python make_movies.py --i path/to/directory/containing/simulation/data``
