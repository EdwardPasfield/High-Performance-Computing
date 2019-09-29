#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include "alloc.h"
#include "boundary.h"
#include "datadef.h"
#include "init.h"
#include "simulation.h"
#include <mpi.h>

void write_bin(float **u, float **v, float **p, char **flag,
     int imax, int jmax, float xlength, float ylength, char *file);

int read_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char *file);

static void print_usage(void);
static void print_version(void);
static void print_help(void);

static char *progname;

int proc = 0;                       /* Rank of the current process */
int nprocs = 0;                /* Number of processes in communicator */

int ileft, iright;           /* Array bounds for each processor */

double startt, endt;
double totalt = 0;

#define PACKAGE "karman"
#define VERSION "1.0"

/* Command line options */
static struct option long_opts[] = {
    { "del-t",   1, NULL, 'd' },
    { "help",    0, NULL, 'h' },
    { "imax",    1, NULL, 'x' },
    { "infile",  1, NULL, 'i' },
    { "jmax",    1, NULL, 'y' },
    { "outfile", 1, NULL, 'o' },
    { "t-end",   1, NULL, 't' },
    { "verbose", 1, NULL, 'v' },
    { "version", 1, NULL, 'V' },
    { 0,         0, 0,    0   }
};
#define GETOPTS "d:hi:o:t:v:Vx:y:"

int main(int argc, char *argv[])
{

//initialsation of communication world, size and rank
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&proc);

 // const int numCPUs = atoi(getenv("OMP_NUM_THREADS"));
  // printf("%d,",numCPUs);
    int verbose = 1;          /* Verbosity level */
    float xlength = 22.0;     /* Width of simulated domain */
    float ylength = 4.1;      /* Height of simulated domain */
  //*2 for the bigger cfd computation
    int imax = 660 * 2;           /* Number of cells horizontally */
    int jmax = 120 * 2;           /* Number of cells vertically */

    char *infile;             /* Input raw initial conditions */
    char *outfile;            /* Output raw simulation results */

    float t_end = 2.1;        /* Simulation runtime */
    float del_t = 0.003;      /* Duration of each timestep */
    float tau = 0.5;          /* Safety factor for timestep control */

    int itermax = 100;        /* Maximum number of iterations in SOR */
    float eps = 0.001;        /* Stopping error threshold for SOR */
    float omega = 1.7;        /* Relaxation parameter for SOR */
    float gamma = 0.9;        /* Upwind differencing factor in PDE
                                 discretisation */

    float Re = 150.0;         /* Reynolds number */
    float ui = 1.0;           /* Initial X velocity */
    float vi = 0.0;           /* Initial Y velocity */

    float t, delx, dely;
    int  i, j, itersor = 0, ifluid = 0, ibound = 0;
    float res;
    float **u, **v, **p, **rhs, **f, **g;
    char  **flag;
    int init_case, iters = 0;
    int show_help = 0, show_usage = 0, show_version = 0;

    progname = argv[0];
    infile = strdup("karman.bin");
    outfile = strdup("karman.bin");

    int optc;
    while ((optc = getopt_long(argc, argv, GETOPTS, long_opts, NULL)) != -1) {
        switch (optc) {
            case 'h':
                show_help = 1;
                break;
            case 'V':
                show_version = 1;
                break;
            case 'v':
                verbose = atoi(optarg);
                break;
            case 'x':
                imax = atoi(optarg);
                break;
            case 'y':
                jmax = atoi(optarg);
                break;
            case 'i':
                free(infile);
                infile = strdup(optarg);
                break;
            case 'o':
                free(outfile);
                outfile = strdup(optarg);
                break;
            case 'd':
                del_t = atof(optarg);
                break;
            case 't':
                t_end = atof(optarg);
                break;
            default:
                show_usage = 1;
        }
    }
    if (show_usage || optind < argc) {
        print_usage();
        return 1;
    }

    if (show_version) {
        print_version();
        if (!show_help) {
            return 0;
        }
    }

    if (show_help) {
        print_help();
        return 0;
    }

    delx = xlength/imax;
    dely = ylength/jmax;

    /* Allocate arrays */
    u    = alloc_floatmatrix(imax+2, jmax+2);
    v    = alloc_floatmatrix(imax+2, jmax+2);
    f    = alloc_floatmatrix(imax+2, jmax+2);
    g    = alloc_floatmatrix(imax+2, jmax+2);
    p    = alloc_floatmatrix(imax+2, jmax+2);
    rhs  = alloc_floatmatrix(imax+2, jmax+2);
    flag = alloc_charmatrix(imax+2, jmax+2);

    if (!u || !v || !f || !g || !p || !rhs || !flag) {
        fprintf(stderr, "Couldn't allocate memory for matrices.\n");
        return 1;
    }

    /* Read in initial values from a file if it exists */
    init_case = read_bin(u, v, p, flag, imax, jmax, xlength, ylength, infile);

    if (init_case > 0) {
        /* Error while reading file */
        return 1;
    }

    if (init_case < 0) {
        /* Set initial values if file doesn't exist */
        for (i=0;i<=imax+1;i++) {
            for (j=0;j<=jmax+1;j++) {
                u[i][j] = ui;
                v[i][j] = vi;
                p[i][j] = 0.0;
            }
        }
        init_flag(flag, imax, jmax, delx, dely, &ibound);
        apply_boundary_conditions(u, v, flag, imax, jmax, ui, vi);
    }
//create p2 matrix in order to re populate the grid with MPI_ALLgather  
    float **p2;

    p2 = alloc_floatmatrix(imax + 2, jmax + 2);
//loop through the grid
    for(i=0; i<=imax + 1;i++){

      for(j=0; j<=jmax + 1;j++){

        p2[i][j] = 0;

      }

    }

    /* Main loop */
//Define the values of ileft and iright 
    ileft = proc * imax/nprocs + 1;
    iright = (proc + 1)* imax/nprocs;
//Define Timers
    double mainStart, mainEnd;
    double mainTotal = 0;
    //main loop start time-stamp
    mainStart = MPI_Wtime();
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        set_timestep_interval(&del_t, imax, jmax, delx, dely, u, v, Re, tau);
        //printf("proc: %d, iteration %d, t: %f \n",proc, iters, t);
        ifluid = (imax * jmax) - ibound;

        compute_tentative_velocity(u, v, f, g, flag, imax, jmax,
            del_t, delx, dely, gamma, Re);

        compute_rhs(f, g, rhs, flag, imax, jmax, del_t, delx, dely);
        //start poisson time-stamp
        startt = MPI_Wtime();

        if (ifluid > 0) {
            itersor = poisson(p, rhs, flag, imax, jmax, delx, dely,
                        eps, itermax, omega, &res, ifluid);
        } else {
            itersor = 0;
        }
        //gather the ileft values and populate p2 with them. 
        MPI_Allgather(p[ileft], (imax/nprocs) * (jmax + 2), MPI_FLOAT, p2[1], (imax/nprocs) * (jmax + 2), MPI_FLOAT, MPI_COMM_WORLD);

        for(i=1; i<=imax;i++){
          for(j=0; j<=jmax + 1;j++){
            p[i][j] = p2[i][j];
          }
        }	
        //poisson loop end time-stamp
        endt = MPI_Wtime();

        if (proc == 0 && verbose > 1) {
            printf("%d t:%g, del_t:%g, SOR iters:%3d, res:%e, bcells:%d\n",
                iters, t+del_t, del_t, itersor, res, ibound);
        }

        update_velocity(u, v, f, g, p, flag, imax, jmax, del_t, delx, dely);

        apply_boundary_conditions(u, v, flag, imax, jmax, ui, vi);
        //calculate total poisson time.
        totalt += (endt-startt);

    } /* End of main loop */
    //end main loop time-stamp
    mainEnd = MPI_Wtime();
    //calculate main loop time total.
    mainTotal += mainEnd - mainStart;

    if (outfile != NULL && strcmp(outfile, "") != 0 && proc == 0) {
        write_bin(u, v, p, flag, imax, jmax, xlength, ylength, outfile);
    }
    //define a double variable that reduce can populate
    double global;
    //reduce totalt by summing it and setting it to global.
    MPI_Reduce(&totalt, &global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(proc == 0 ){
      printf("%g,%g,%g,%d\n",(global/(iters*nprocs)),((mainTotal)/iters), (mainTotal), nprocs);
    //  printf("Average Poisson Loop Time: %g \n", global/(iters*nprocs));
    //  printf("Average Total Main Loop Time: %g \n", (mainEnd-mainStart)/iters);

    }
    //printf("");

    free_matrix(u);
    free_matrix(v);
    free_matrix(f);
    free_matrix(g);
    free_matrix(p);
    free_matrix(rhs);
    free_matrix(flag);

    return 0;
}

/* Save the simulation state to a file */
void write_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char* file)
{
    int i;
    FILE *fp;

    fp = fopen(file, "wb");

    if (fp == NULL) {
        fprintf(stderr, "Could not open file '%s': %s\n", file,
            strerror(errno));
        return;
    }

    fwrite(&imax, sizeof(int), 1, fp);
    fwrite(&jmax, sizeof(int), 1, fp);
    fwrite(&xlength, sizeof(float), 1, fp);
    fwrite(&ylength, sizeof(float), 1, fp);

    for (i=0;i<imax+2;i++) {
        fwrite(u[i], sizeof(float), jmax+2, fp);
        fwrite(v[i], sizeof(float), jmax+2, fp);
        fwrite(p[i], sizeof(float), jmax+2, fp);
        fwrite(flag[i], sizeof(char), jmax+2, fp);
    }
    fclose(fp);
}

/* Read the simulation state from a file */
int read_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char* file)
{
    int i,j;
    FILE *fp;

    if (file == NULL) return -1;

    if ((fp = fopen(file, "rb")) == NULL) {
        fprintf(stderr, "Could not open file '%s': %s\n", file,
            strerror(errno));
        fprintf(stderr, "Generating default state instead.\n");
        return -1;
    }

    fread(&i, sizeof(int), 1, fp);
    fread(&j, sizeof(int), 1, fp);
    float xl, yl;
    fread(&xl, sizeof(float), 1, fp);
    fread(&yl, sizeof(float), 1, fp);

    if (i!=imax || j!=jmax) {
        fprintf(stderr, "Warning: imax/jmax have wrong values in %s\n", file);
        fprintf(stderr, "%s's imax = %d, jmax = %d\n", file, i, j);
        fprintf(stderr, "Program's imax = %d, jmax = %d\n", imax, jmax);
        return 1;
    }
    if (xl!=xlength || yl!=ylength) {
        fprintf(stderr, "Warning: xlength/ylength have wrong values in %s\n", file);
        fprintf(stderr, "%s's xlength = %g,  ylength = %g\n", file, xl, yl);
        fprintf(stderr, "Program's xlength = %g, ylength = %g\n", xlength,
            ylength);
        return 1;
    }

    for (i=0; i<imax+2; i++) {
        fread(u[i], sizeof(float), jmax+2, fp);
        fread(v[i], sizeof(float), jmax+2, fp);
        fread(p[i], sizeof(float), jmax+2, fp);
        fread(flag[i], sizeof(char), jmax+2, fp);
    }
    fclose(fp);
    return 0;
}

static void print_usage(void)
{
    fprintf(stderr, "Try '%s --help' for more information.\n", progname);
}

static void print_version(void)
{
    fprintf(stderr, "%s %s\n", PACKAGE, VERSION);
}

static void print_help(void)
{
    fprintf(stderr, "%s. A simple computational fluid dynamics tutorial.\n\n",
        PACKAGE);
    fprintf(stderr, "Usage: %s [OPTIONS]...\n\n", progname);
    fprintf(stderr, "  -h, --help            Print a summary of the options\n");
    fprintf(stderr, "  -V, --version         Print the version number\n");
    fprintf(stderr, "  -v, --verbose=LEVEL   Set the verbosity level. 0 is silent\n");
    fprintf(stderr, "  -x, --imax=IMAX       Set the number of interior cells in the X direction\n");
    fprintf(stderr, "  -y, --jmax=JMAX       Set the number of interior cells in the Y direction\n");
    fprintf(stderr, "  -t, --t-end=TEND      Set the simulation end time\n");
    fprintf(stderr, "  -d, --del-t=DELT      Set the simulation timestep size\n");
    fprintf(stderr, "  -i, --infile=FILE     Read the initial simulation state from this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
    fprintf(stderr, "  -o, --outfile=FILE    Write the final simulation state to this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
}
