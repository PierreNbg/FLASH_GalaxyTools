# Scripts to determine an inital pressure to stabilise the disk a galaxy.

This script consist of two separate routines which use a slightly different approach for the integration step.

**Uniform grid**
The initial approach in the *read_fortran_chk.py* routine uses uniform grids.
All dataset are thus reprojected from the AMR grid onto a uniform grid of the highest included resolution of the grid.
This becomes very computational expensive for higher resolved datasets.

**AMR grid**
The newer approach does all calculations on the AMR structure provided by FLASH.
We create our own guard cells to propagate the integration from one block to the next.
The routine is called *get_pressure_amr.py*.
