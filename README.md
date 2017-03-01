ssipative Particle Dynamics
Dissipative Particle Dynamics (DPD) is a mesoscale coarse-grained fluid dynamics simulation technique.  Instead of simulating fluids with atomistic detail, which would place needless restrictions on the time and length scales one can probe, DPD represents many molecules of the fluid with a single point particle. For a detailed description of the method and its parameters, please see the seminal work of [Hoogerbrugge and Koelman (1992)](http://iopscience.iop.org/article/10.1209/0295-5075/19/3/001) and [Groot and Warren (1997)](http://aip.scitation.org/doi/abs/10.1063/1.474784). This software implementation uses MPI to parallelize the calculations, and was used for the parameter optimization project: [Krafnick and García, *Efficient Schmidt number scaling in dissipative particle dynamics*, J. Chem. Phys. **143** (2016)](http://dx.doi.org/10.1063/1.4930921).

### Compiling
This code runs on linux and uses MPI.  To compile the code into a production executable, simply run `make` in the project directory, which will use mpic++ to create an MPI application.  Since the goal of the above project was to examine and optimize system parameters, there are 5 additional utility executables available.

Use `make torque_test` to compile an executable for testing the sheer friction coefficient γ<sup>S</sup>.  This will rotate a fixed-position test particle at a fixed angular velocity and measure the torque that the fluid exerts on it in response.  This can be compared with the theoretical torque on a rotating sphere to derive a value for the effective radius of a DPD particle with the given friction coefficient.

Use `make force_test` to compile an executable for testing the central friction coefficient γ<sup>C</sup>.  This will simulate the DPD system with a fixed-position test particle in a constant-force fluid flow.  The force driving the fluid will increase the average fluid velocity until the friction (which is based on γ<sup>C</sup> and γ<sup>S</sup>) balances it out.  By comparing the force on the test particle to the Stokes' drag experienced by a spherical particle, we can come up with another expression for the effective particle hydrodynamic radius.  Since the force here depends on both the central and sheer friction, this step should be performed after the torque test.

Temperature reproduction can be verified via the `make temperature_test` executable.  The measured value of the temperature depends on the time step and the velocity prediction parameter λ.  This outputs temperature values derived from both the translational and rotational kinetic energies, which may not be equivalent.

The viscosity of the system can be measured using the result of `make viscosity`, which uses the periodic Poiseuille flow method (see [J. A. Backer et al. 2005](http://aip.scitation.org/doi/abs/10.1063/1.1883163)).  With this method, the velocity profile of a system is measured under a separate pressure drop for each half of the simulation box, where the peak and average fluid velocity values have theoretical relationships with the fluid viscosity itself.

Optimization of the system requires the use of as short a cutoff as possible on neighbor searching.  To determine the lower limit on the neighbor search cutoff, use `make ncutoff_test`.  This program prints out the maximum distances traveled between neighbor searching phases, and can be compared with data from different values of the neighbor cutoff length to determine at what point the cutoff affects the results.

An example input file is included.  As mentioned, the primary purpose of this program was to test and refine parameters for adequate representation of target fluids, and does not have the same features as something designed for more generic purposes.

### Running
An example input file is included.  As mentioned, the primary purpose of this program was to test and refine parameters for adequate representation of target fluids, and does not have the same features as something designed for more generic purposes.  The basic format of a run command is:
    
    dpd.x -input INPUT_FILE -output OUTPUT_FILE
Place `mpirun -np NP` before the command to run `NP` MPI processes in parallel, and use `-x NX -y NY -z NZ` to set the distribution of processes along each axis.  In order to set up a deterministic simulation with a fixed random seed, use `-seed N`.  Timing statistics (which were fundamental to this work) can be extracted via the `-show-cycles` option, which prints out the raw cycle count spent on a variety of tasks.

