#include <stdio.h>
#include "mpi.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
    int       p;         /* Total number of processes    */
    MPI_Comm  comm;      /* Communicator for entire grid */
    MPI_Comm  row_comm;  /* Communicator for my row      */
    MPI_Comm  col_comm;  /* Communicator for my col      */
    int       q;         /* Order of grid                */
    int       my_row;    /* My row number                */
    int       my_col;    /* My column number             */
    int       my_rank;   /* My rank in the grid comm     */
} GRID_INFO_T;


#define MAX 65536
#define MAX_ORDER 1000
typedef struct {
    int     n_bar;
#define Order(A) ((A)->n_bar)
    float  entries[MAX];
#define Entry(A,i,j) (*(((A)->entries) + ((A)->n_bar)*(i) + (j)))
} LOCAL_MATRIX_T;

/* Function Declarations */
LOCAL_MATRIX_T*  Local_matrix_allocate(int n_bar);
void             Free_local_matrix(LOCAL_MATRIX_T** local_A);
void             Read_matrix(char* prompt,char* filename, LOCAL_MATRIX_T* local_A,
                     GRID_INFO_T* grid, int n);
void             Print_matrix(char* title,char* filename, LOCAL_MATRIX_T* local_A,
                     GRID_INFO_T* grid, int n);
void             Set_to_zero(LOCAL_MATRIX_T* local_A);
void             Local_matrix_multiply(LOCAL_MATRIX_T* local_A,
                     LOCAL_MATRIX_T* local_B, LOCAL_MATRIX_T* local_C);
void             Build_matrix_type(LOCAL_MATRIX_T* local_A);
MPI_Datatype     local_matrix_mpi_t;

LOCAL_MATRIX_T*  temp_mat;
void             Print_local_matrices(char* title, LOCAL_MATRIX_T* local_A,
                     GRID_INFO_T* grid);

/*********************************************************/
main(int argc, char* argv[]) {
    int              p;
    int              my_rank;
    GRID_INFO_T      grid;
    LOCAL_MATRIX_T*  local_A;
    LOCAL_MATRIX_T*  local_B;
    LOCAL_MATRIX_T*  local_C;
    int              n_bar;
    char*            filename;
    double           Tstart =0,Tend =0,Twall = 0,t;

    int n = atoi(argv[1]);
    float f = atof(argv[2]);
    float g = atof(argv[3]);
    float s = atof(argv[4]);
    char* Matrix_A = argv[5];
    char* Matrix_B = argv[6];
    char* Matrix_C = argv[7];

    if (my_rank == 0)
        Tstart = MPI_Wtime();
    void Setup_grid(GRID_INFO_T*  grid);
    void Fox(int n, GRID_INFO_T* grid, LOCAL_MATRIX_T* local_A,
             LOCAL_MATRIX_T* local_B, LOCAL_MATRIX_T* local_C);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

// checking for the command line arguments
    if(my_rank == 0){
        if (argc != 8 ){
        printf("need to give order, value of f, value of g, spanning point value s, Matrix A file name,Matrix B file name,Matrix C file name");
        return 1;
    }
    }

    Setup_grid(&grid);
    if (my_rank == 0) {
    printf("The order of the matrices is:%d\n",n);
    printf("Given F is:%f\n",f);
    printf("Given G is:%f\n",g);
    printf("Spanning point S is:%1.0f\n",s);

    int i,k;
    float j,r;
    float x[MAX],y[MAX];
    float A[MAX_ORDER][MAX_ORDER];
    float B[MAX_ORDER][MAX_ORDER];
    float a;
    FILE *fa,*fb;
    // linspace function code

    a = (2*s)/(n-1);
    for (i = 0,j = -s ;i < n; i++, j += a )
    {
        x[i] = j;
        y[i] = j;

        }
    fa = fopen(Matrix_A,"w");
    // generating Matrix A and B
    for (i=0;i<n;i++){
    for (k=0;k<n;k++){
    r=sqrt((x[i]*x[i]) + y[k]*y[k]);
    A[i][k] = f*(cos(r)*cos(r));
    B[i][k] = exp(-g*r);
    fprintf(fa,"%1.3f\t",A[i][k]);
    }
    fprintf(fa,"\n");
    }
    fclose(fa);
    fb = fopen(Matrix_B,"w");
    for (i=0;i<n;i++){
    for (k=0;k<n;k++){
    fprintf(fb,"%1.3f\t",B[i][k]);
    }
    fprintf(fb,"\n");
}
fclose(fb);
}

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n_bar = n/grid.q;

    local_A = Local_matrix_allocate(n_bar);
    Order(local_A) = n_bar;
    Read_matrix("Reading Matrix A",Matrix_A, local_A, &grid, n);
    Print_matrix("We read A =",Matrix_A, local_A, &grid, n);

    local_B = Local_matrix_allocate(n_bar);
    Order(local_B) = n_bar;
    Read_matrix("Reading Matrix B",Matrix_B, local_B, &grid, n);
    Print_matrix("We read B =",Matrix_B, local_B, &grid, n);

    Build_matrix_type(local_A);
    temp_mat = Local_matrix_allocate(n_bar);

    local_C = Local_matrix_allocate(n_bar);
    Order(local_C) = n_bar;
    Fox(n, &grid, local_A, local_B, local_C);

    Print_matrix("The product is",Matrix_C, local_C, &grid, n);

    Free_local_matrix(&local_A);
    Free_local_matrix(&local_B);
    Free_local_matrix(&local_C);

    if(my_rank ==0){
    Tend = MPI_Wtime();
    Twall = Tend - Tstart;
    printf("Time for execution:%f\n",Twall);
    }
    MPI_Finalize();
}  /* main */


/*********************************************************/
void Setup_grid(
         GRID_INFO_T*  grid  /* out */) {
    int old_rank;
    int dimensions[2];
    int wrap_around[2];
    int coordinates[2];
    int free_coords[2];

    /* Set up Global Grid Information */
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

    /* We assume p is a perfect square */
    grid->q = (int) sqrt((double) grid->p);
    dimensions[0] = dimensions[1] = grid->q;

    /* We want a circular shift in second dimension. */
    /* Don't care about first                        */
    wrap_around[0] = wrap_around[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions,
        wrap_around, 1, &(grid->comm));
    MPI_Comm_rank(grid->comm, &(grid->my_rank));
    MPI_Cart_coords(grid->comm, grid->my_rank, 2,
        coordinates);
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];

    /* Set up row communicators */
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(grid->comm, free_coords,
        &(grid->row_comm));

    /* Set up column communicators */
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(grid->comm, free_coords,
        &(grid->col_comm));
} /* Setup_grid */


/*********************************************************/
void Fox(
        int              n         /* in  */,
        GRID_INFO_T*     grid      /* in  */,
        LOCAL_MATRIX_T*  local_A   /* in  */,
        LOCAL_MATRIX_T*  local_B   /* in  */,
        LOCAL_MATRIX_T*  local_C   /* out */) {

    LOCAL_MATRIX_T*  temp_A; /* Storage for the sub-    */
                             /* matrix of A used during */
                             /* the current stage       */
    int              stage;
    int              bcast_root;
    int              n_bar;  /* n/sqrt(p)               */
    int              source;
    int              dest;
    MPI_Status       status;

    n_bar = n/grid->q;
    Set_to_zero(local_C);

    /* Calculate addresses for circular shift of B */
    source = (grid->my_row + 1) % grid->q;
    dest = (grid->my_row + grid->q - 1) % grid->q;

    /* Set aside storage for the broadcast block of A */
    temp_A = Local_matrix_allocate(n_bar);

    for (stage = 0; stage < grid->q; stage++) {
        bcast_root = (grid->my_row + stage) % grid->q;
        if (bcast_root == grid->my_col) {
            MPI_Bcast(local_A, 1, local_matrix_mpi_t,
                bcast_root, grid->row_comm);
            Local_matrix_multiply(local_A, local_B,
                local_C);
        } else {
            MPI_Bcast(temp_A, 1, local_matrix_mpi_t,
                bcast_root, grid->row_comm);
            Local_matrix_multiply(temp_A, local_B,
                local_C);
        }
        MPI_Sendrecv_replace(local_B, 1, local_matrix_mpi_t,
            dest, 0, source, 0, grid->col_comm, &status);
    } /* for */

} /* Fox */


/*********************************************************/
LOCAL_MATRIX_T* Local_matrix_allocate(int local_order) {
    LOCAL_MATRIX_T* temp;

    temp = (LOCAL_MATRIX_T*) malloc(sizeof(LOCAL_MATRIX_T));
    return temp;
}  /* Local_matrix_allocate */


/*********************************************************/
void Free_local_matrix(
         LOCAL_MATRIX_T** local_A_ptr  /* in/out */) {
    free(*local_A_ptr);
}  /* Free_local_matrix */


/*********************************************************/
/* Read and distribute matrix:
 *     for each global row of the matrix,
 *         for each grid column
 *             read a block of n_bar floats on process 0
 *             and send them to the appropriate process.
 */
void Read_matrix(
         char*            prompt   /* in  */,
         char*            filename,
         LOCAL_MATRIX_T*  local_A  /* out */,
         GRID_INFO_T*     grid     /* in  */,
         int              n        /* in  */) {

    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        dest;
    int        coords[2];
    float*     temp;
    FILE        *fr;
    MPI_Status status;
    fr = fopen(filename,"r");

    if (grid->my_rank == 0) {
        temp = (float*) malloc(Order(local_A)*sizeof(float));
        printf("%s\n", prompt);
        fflush(stdout);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_A);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                if (dest == 0) {
                    for (mat_col = 0; mat_col < Order(local_A); mat_col++){
                         if(fscanf(fr,"%f",(local_A->entries)+mat_row*Order(local_A)+mat_col)!= EOF){
                          }
                    }
                } else {
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++){
                        if( fscanf(fr,"%f",temp + mat_col)!= EOF){
                        }
                        }
                    MPI_Send(temp, Order(local_A), MPI_FLOAT, dest, 0,
                        grid->comm);
                }
            }
        }
        free(temp);
        fclose(fr);
    } else {
        for (mat_row = 0; mat_row < Order(local_A); mat_row++)
            MPI_Recv(&Entry(local_A, mat_row, 0), Order(local_A),
                MPI_FLOAT, 0, 0, grid->comm, &status);
    }

}  /* Read_matrix */


/*********************************************************/
void Print_matrix(
         char*            title    /* in  */,
         char*            filename,
         LOCAL_MATRIX_T*  local_A  /* out */,
         GRID_INFO_T*     grid     /* in  */,
         int              n        /* in  */) {
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        source;
    int        coords[2];
    float*     temp;
    FILE       *fp;
    MPI_Status status;

    if (grid->my_rank == 0) {
        fp=fopen(filename,"w");

        temp = (float*) malloc(Order(local_A)*sizeof(float));
        printf("%s\n", title);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_A);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &source);
                if (source == 0) {
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++){
                        fprintf(fp,"%4.3f ", Entry(local_A, mat_row, mat_col));
                        printf("%4.3f ", Entry(local_A, mat_row, mat_col));
                    }
                } else {
                    MPI_Recv(temp, Order(local_A), MPI_FLOAT, source, 0,
                        grid->comm, &status);
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++){
                        fprintf(fp,"%4.3f ", temp[mat_col]);
                        printf("%4.3f ", temp[mat_col]);
                    }
                }
            }
            printf("\n");
            fprintf(fp,"\n");
        }
        fclose(fp);
        free(temp);
    } else {
        for (mat_row = 0; mat_row < Order(local_A); mat_row++)
            MPI_Send(&Entry(local_A, mat_row, 0), Order(local_A),
                MPI_FLOAT, 0, 0, grid->comm);
    }

}  /* Print_matrix */


/*********************************************************/
void Set_to_zero(
         LOCAL_MATRIX_T*  local_A  /* out */) {

    int i, j;

    for (i = 0; i < Order(local_A); i++)
        for (j = 0; j < Order(local_A); j++)
            Entry(local_A,i,j) = 0.0;

}  /* Set_to_zero */


/*********************************************************/
void Build_matrix_type(
         LOCAL_MATRIX_T*  local_A  /* in */) {
    MPI_Datatype  temp_mpi_t;
    int           block_lengths[2];
    MPI_Aint      displacements[2];
    MPI_Datatype  typelist[2];
    MPI_Aint      start_address;
    MPI_Aint      address;

    MPI_Type_contiguous(Order(local_A)*Order(local_A),
        MPI_FLOAT, &temp_mpi_t);

    block_lengths[0] = block_lengths[1] = 1;

    typelist[0] = MPI_INT;
    typelist[1] = temp_mpi_t;

    MPI_Address(local_A, &start_address);
    MPI_Address(&(local_A->n_bar), &address);
    displacements[0] = address - start_address;

    MPI_Address(local_A->entries, &address);
    displacements[1] = address - start_address;

    MPI_Type_struct(2, block_lengths, displacements,
        typelist, &local_matrix_mpi_t);
    MPI_Type_commit(&local_matrix_mpi_t);
}  /* Build_matrix_type */


/*********************************************************/
void Local_matrix_multiply(
         LOCAL_MATRIX_T*  local_A  /* in  */,
         LOCAL_MATRIX_T*  local_B  /* in  */,
         LOCAL_MATRIX_T*  local_C  /* out */) {
    int i, j, k;

    for (i = 0; i < Order(local_A); i++)
        for (j = 0; j < Order(local_A); j++)
            for (k = 0; k < Order(local_B); k++)
                Entry(local_C,i,j) = Entry(local_C,i,j)
                    + Entry(local_A,i,k)*Entry(local_B,k,j);

}  /* Local_matrix_multiply */
