/*Sorce Code From Peter Pacheco Prof. Mary Thomas SDSU 2014 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#define MAX_ORDER 100

typedef float LOCAL_MATRIX_T[MAX_ORDER][MAX_ORDER];

main(int argc, char* argv[]) {
    int             my_rank;
    int             p;
    LOCAL_MATRIX_T  local_A;
    float           global_x[MAX_ORDER];
    float           local_x[MAX_ORDER];
    float           local_y[MAX_ORDER];
    //int             m, n;
    int             local_m, local_n;
    double tws,twe;
    tws = MPI_Wtime();
    FILE *fp;
    int count =0;
    int ch;

    void Read_matrix(char* file, LOCAL_MATRIX_T local_A, int local_m, int n,int my_rank, int p);
    void Read_vector(char*file, char* prompt, float local_x[], int local_n, int my_rank,int p);
    void Parallel_matrix_vector_prod_AccInt( LOCAL_MATRIX_T local_A, int m,int n, float local_x[], float global_x[], float local_y[],
                        int local_m, int local_n, int my_rank, int p);
    void Print_matrix(char* title, LOCAL_MATRIX_T local_A, int local_m, int n, int my_rank, int p);
    void Total_Balance(char* file, LOCAL_MATRIX_T local_A, float local_y[], int local_m, int local_n, int my_rank,int p,int n, float local_x[]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    char* balance = argv[3];
    char* interest = argv[4];
    char* newBalance = argv[5];


    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_m = m/p;
    local_n = n/p;

    Read_matrix(balance, local_A, local_m, n, my_rank, p);
    Print_matrix("\nStarting account balances:\n", local_A, local_m, n, my_rank, p);

    Read_vector(interest, "\nAccount Interests:\n", local_x, local_n, my_rank, p);
    Parallel_matrix_vector_prod_AccInt(local_A, m, n, local_x, global_x,
        local_y, local_m, local_n, my_rank,p);
    Total_Balance(newBalance,local_A, local_y, local_m, local_n, my_rank, p, n, local_x);

     MPI_Finalize();
     twe = MPI_Wtime();
     if(my_rank == 0)
     printf("\nTWall For The program is: %f\n", (twe-tws));

}  /* main */


/**********************************************************************/
void Read_matrix( char* file,
         LOCAL_MATRIX_T  local_A  /* out */,
         int             local_m  /* in  */,
         int             n        /* in  */,
         int             my_rank  /* in  */,
         int             p        /* in  */) {
         int             i, j;
    LOCAL_MATRIX_T  temp;
    FILE *fv;
    float ch;

    for (i = 0; i < p*local_m; i++)
        for (j = n; j < MAX_ORDER; j++)
            temp[i][j] = 0.0;

   if (my_rank == 0) {

   fv=fopen(file,"r");
   if( fv == NULL )
   {
      perror("Error while opening the file.\n");
      exit(EXIT_FAILURE);
   }
            i = 0; j = 0;
            while(fscanf(fv,"%f",&ch)!= EOF){
            temp[i][j] = ch;
              j++;
               if(j>=n)
            {
              i++;
              j = 0;
              if(i>=p*local_m)
               break;
            }
            }
    fclose(fv);
   }

   MPI_Scatter(temp, local_m*MAX_ORDER, MPI_FLOAT, local_A,local_m*MAX_ORDER, MPI_FLOAT, 0, MPI_COMM_WORLD);
}  /* Read_matrix */


/**********************************************************************/
void Read_vector(
                 char* file,
         char*  prompt     /* in  */,
         float  local_x[]  /* out */,
         int    local_n    /* in  */,
         int    my_rank    /* in  */,
         int    p          /* in  */)
          {
         int   i =0;
         float temp[MAX_ORDER];
         FILE *fv;
         float ch;

    if (my_rank == 0) {
        printf("%s\n", prompt);

    fv=fopen(file,"r");
   if( fv == NULL )
   {
      perror("Error while opening the file.\n");
      exit(EXIT_FAILURE);
   }
            while(fscanf(fv,"%f",&ch)!= EOF){
            temp[i] = ch;
            printf("AI:%d ", i+1);
            printf(" %f\n",temp[i]);
            i++;
            if(i>p*local_n)
            break;
            }
    }
    MPI_Scatter(temp, local_n, MPI_FLOAT, local_x, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

}  /* Read_vector */


/**********************************************************************/
void Parallel_matrix_vector_prod_AccInt(
         LOCAL_MATRIX_T  local_A     /* in  */,
         int             m           /* in  */,
         int             n           /* in  */,
         float           local_x[]   /* in  */,
         float           global_x[]  /* in  */,
         float           local_y[]   /* out */,
         int             local_m     /* in  */,
         int             local_n     /* in  */,
         int             my_rank,
         int p) {

    float Account_interest[MAX_ORDER][MAX_ORDER];
    float total = 0.0;
    float temp[MAX_ORDER][MAX_ORDER];
    int i, j;
    MPI_Gather(local_x, local_n, MPI_FLOAT, global_x, local_n , MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_A, local_m*MAX_ORDER, MPI_FLOAT, temp, local_m*MAX_ORDER, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(my_rank == 0)
    {
        printf("Interest for individual accounts:\n");
        for (i = 0; i < p*local_m; i++)
        {
            printf("UID:%d",i+1);
            total = 0.0;
            for (j = 0; j < n; j++)
            {
                total += temp[i][j]*global_x[j];
                Account_interest[i][j] = temp[i][j]*global_x[j];
                printf("%4.1f\t", Account_interest[i][j]);
            }
            printf("=  %f\t",total);
            printf("\n");
        }
    }
}/* Parallel_matrix_vector_prod */


/**********************************************************************/
void Print_matrix(
         char*           title      /* in */,
         LOCAL_MATRIX_T  local_A    /* in */,
         int             local_m    /* in */,
         int             n          /* in */,
         int             my_rank    /* in */,
         int             p          /* in */) {

    int   i, j;
    float temp[MAX_ORDER][MAX_ORDER];

    MPI_Gather(local_A, local_m*MAX_ORDER, MPI_FLOAT, temp,
         local_m*MAX_ORDER, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("\n%s\n", title);
        for (i = 0; i < p*local_m; i++)
        {
            printf("UID:%d ", i+1);
            for (j = 0; j < n; j++)
                printf("%4.1f ", temp[i][j]);
            printf("\n");
        }
    }
}  /* Print_matrix */

/*********************************************************************/
void Total_Balance(
         char* file,
         LOCAL_MATRIX_T local_A,
         float  local_y[],
         int local_m,
         int local_n,
         int my_rank,
         int p,
         int n, float local_x[]){
         float OldTotal[MAX_ORDER];
         int i,j;
         float temp[MAX_ORDER][MAX_ORDER];
        float global_x[MAX_ORDER];
        FILE *ft;

    MPI_Gather(local_A, local_m*MAX_ORDER, MPI_FLOAT, temp, local_m*MAX_ORDER, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_x, local_n, MPI_FLOAT, global_x, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

   if(my_rank ==0){
 for(i=0;i<local_m*p;i++)
{
    OldTotal[i] = 0;
    for(j=0;j<n;j++)
    {
        OldTotal[i]+=temp[i][j];
    }
}

    ft =fopen(file, "w");
    fprintf(ft,"Account Balances Without Interest\n\n");

    printf("\nTotal Balance without interest:\n\n");
    for(i=0;i<p*local_m;i++)
    {
        fprintf(ft,"UID:%d ", i+1);
        printf("UID:%d ", i+1);
        for(j=0;j<n;j++)
        {
            fprintf(ft,"%4.1f\t",temp[i][j]);
            printf("%4.1f\t",temp[i][j]);
        }
        fprintf(ft,"  =  %4.1f ",OldTotal[i]);
        printf("  =  %4.1f ",OldTotal[i]);
        fprintf(ft,"\n");
        printf("\n");
    }

    fprintf(ft,"Account Balances With Interest\n\n");
    printf("\nAccount balances with interest:\n");

    float NewTotal[MAX_ORDER];

    for(i=0;i<p*local_m;i++)
            {
        for (j = 0; j < p*local_n; j++)
    {
        local_y[i]= local_y[i] +temp[i][j]*global_x[j];

    }

    }

    for (i = 0; i <local_m*p; i++)
    {
        fprintf(ft,"UID:%d ",i+1);
        printf("UID:%d ",i+1);
        for(j=0;j<n;j++)
        {
            fprintf(ft,"%4.1f\t ", temp[i][j]);
            printf("%4.1f\t ", temp[i][j]);
        }
        NewTotal[i]= OldTotal[i]+local_y[i];
        fprintf(ft,"  =  %4.1f",NewTotal[i]);
        printf("  =  %4.1f",NewTotal[i]);
        fprintf(ft,"\n");
        printf("\n");
    }

}
}
