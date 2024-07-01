from os import system
import numpy as np
from partition_functions import molecular_hydrogen_energy


def test_c_vs_python():
    """Verifies that the C and python implementations of the partition function
    agree to machine precision.
    """
    system("gcc partition_function.c -o ZROT -lm -Ofast")
    system("./ZROT")
    c_output = np.loadtxt("partition_function_c.dat")
    system("rm partition_function_c.dat ZROT")

    Tgrid = np.logspace(1, 5, 41)
    python_values = np.array(molecular_hydrogen_energy(Tgrid)).T
    assert np.isclose(np.c_[Tgrid, python_values], c_output).all()
