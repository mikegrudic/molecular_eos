#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const double BOLTZMANN = 1.380649e-16; // in erg/K
const double EPSILON = 2.220446049250313e-16;

double molecular_hydrogen_zrot_mixture(double temp, double result[3])
{
    /*
    Rotational partition function of hydrogen molecule and derived quantities,
    considering a 3:1 mixture of ortho- and parahydrogen that cannot efficiently
    come into equilibrium.

    Parameters
    ----------
    temp: double
        Temperature in K
    ortho_frac: double
        Fraction of ortho-H2 (default is 3:1 ortho:para mixture)
    result: double[3]
        Stores the partition function value, the average rotational energy per molecule,
        and the heat capacity per molecule at constant volume.
    */

    const double THETA_ROT = 85.4;         // in K
    const double ortho_frac = 0.75; // 3:1 mixture
    const double para_frac = 1 - ortho_frac;
    const double x = THETA_ROT / temp;
    const double expmx = exp(-x);
    const double expmx4 = pow(expmx, 4);

    double error = 1e100;
    double z[2] = {0}; // index 0 for para, 1 for ortho
    double dz_dtemp[2] = {0};
    double d2z_dtemp2[2] = {0};
    double zterm[2] = {0};

    z[0] = zterm[0] = 1.0;
    z[1] = zterm[1] = 9.0;
    double expterm = expmx4 * expmx * expmx;

    // Summing over rotational levels
    int j = 2;
    double dzterm, d2zterm;
    while (error > EPSILON)
    {
        int s = j % 2;
        zterm[s] *= (2 * j + 1) * expterm / (2 * j - 3);
        int jjplusone = j * (j + 1);
        if (s == 1)
        { // ortho
            dzterm = (jjplusone - 2) * x * zterm[1];
            d2zterm = ((jjplusone - 2) * x - 2) * dzterm;
        }
        else
        { // para
            dzterm = jjplusone * x * zterm[0];
            d2zterm = (jjplusone * x - 2) * dzterm;
        }
        z[s] += zterm[s];
        dz_dtemp[s] += dzterm;
        d2z_dtemp2[s] += d2zterm;
        double err0 = zterm[0]/z[0];
        double err1 = zterm[1]/z[1];
        if (err1 > err0)
        {
            error = err1;
        }
        else
        {
            error = err0;
        }
        expterm *= expmx4;
        j++;
    } 

    result[0] = exp(para_frac * log(z[0]) + ortho_frac * log(z[1]));                                                                                                                                              // partition function
    result[1] = BOLTZMANN * temp * (para_frac * dz_dtemp[0] / z[0] + ortho_frac * dz_dtemp[1] / z[1]);                                                                                                            // mean energy per molecule
    result[2] = BOLTZMANN * (ortho_frac * (2 * dz_dtemp[1] + d2z_dtemp2[1] - dz_dtemp[1] * dz_dtemp[1] / z[1]) / z[1] + para_frac * (2 * dz_dtemp[0] + d2z_dtemp2[0] - dz_dtemp[0] * dz_dtemp[0] / z[0]) / z[0]); // heat capacity
}

double molecular_hydrogen_zvib(double temp, double result[3])
{
    /*
    Vibrational partition function of hydrogen molecule and derived quantities.

    Parameters
    ----------
    temp: double
        Temperature in K
    result: double[3]
        Stores the partition function value, the average rotational energy per molecule,
        and the heat capacity per molecule at constant volume.
    */
    const double THETA_VIB = 6140;
    const double x = THETA_VIB / temp;
    result[0] = -1.0 / expm1(-x);
    result[1] = BOLTZMANN * THETA_VIB / expm1(x);
    result[2] = THETA_VIB * result[0] * result[1] / (temp * temp);
}

double molecular_hydrogen_partition(double temp, double result[3]){
    /*
    Partition function of hydrogen molecule and derived quantities.

    Parameters
    ----------
    temp: double
        Temperature in K
    result: double[3]
        Stores the rotational partition function value, the average rotational energy per molecule,
        the heat capacity per molecule at constant volume, and the adiabatic index
    */

    double zrot[4], zvib[4];
    molecular_hydrogen_zrot_mixture(temp, zrot);
    molecular_hydrogen_zvib(temp,zvib);
    double etot = 1.5 * BOLTZMANN * temp;  // translation
    double cv = 1.5 * BOLTZMANN;
    etot += zrot[1];  // rotation
    cv += zrot[2];
    etot += zvib[1];  // vibration
    cv += zvib[2];
    double gamma = (cv / BOLTZMANN + 1) / (cv / BOLTZMANN);
    result[0] = etot;
    result[1] = cv;
    result[2] = gamma;
}

void main()
{
    double logT = 1;
    double *Tgrid = malloc(sizeof(double)*41);
    int i = 0;
    double result[3];
    // Open a file in writing mode
    FILE *fptr;
    fptr = fopen("partition_function_c.dat", "w");
    fprintf(fptr, "# (0) Temperature (K) (1) Partition function (2) Average energy (erg) (3) Heat capacity at constant vol (erg/K)\n");
    while (logT <= 5)
    {
        Tgrid[i] = pow(10., logT);
        molecular_hydrogen_partition(Tgrid[i], result);
        fprintf(fptr, "%g %g %g %g\n",Tgrid[i],result[0],result[1],result[2]);
        logT += 1e-1;
        i++;
    }
    // Close the file
    fclose(fptr); 
}