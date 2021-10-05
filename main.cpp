#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#define N 40
#define NS 10

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name( name, &len );

    if(rank == 3 || rank == 35){
        printf("[rank %d] the world is %d ranks wide and my name is %s\n",rank,comm_size,name); fflush(stdout);
    }

    // allocate the array of size [10x10x10]
    int size = N*N*N;
    int* array = (int*) malloc(sizeof(int)*size);
    for(int i=0; i<size; ++i){
        array[i] = rank;
    }

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Win  window = MPI_WIN_NULL;
    MPI_Win_create(array,size*sizeof(int), sizeof(int), info, MPI_COMM_WORLD, &window);
    MPI_Info_free(&info);

    // create the group
    //int next_rank = (comm_size + rank + 1)%comm_size;
    //int prev_rank = (comm_size + rank - 1)%comm_size;
    int next_rank = 0 + (rank == 3) * 35 + (rank == 35) * 3;
    int prev_rank = 0 + (rank == 3) * 35 + (rank == 35) * 3;
    if(rank == 3 || rank == 35){
        printf("[rank %d] I am between rank %d and rank %d\n",rank,prev_rank,next_rank); fflush(stdout);
    }
    MPI_Group prev_group, next_group, global_group;
    MPI_Comm_group(MPI_COMM_WORLD, &global_group);
    MPI_Group_incl(global_group,1,&prev_rank,&prev_group);
    MPI_Group_incl(global_group,1,&next_rank,&next_group);
    MPI_Group_free(&global_group);


    // get the datatype to get the rank in an array of size [2,2,2]
    MPI_Datatype type1, type2;
    MPI_Type_create_hvector(NS,NS,N*sizeof(int),MPI_INT,&type1);
    MPI_Type_create_hvector(NS,1,N*N*sizeof(int),type1,&type2);
    MPI_Type_commit(&type2);
    MPI_Type_free(&type1);

    // start
    MPI_Win_post(prev_group,0,window);
    MPI_Win_start(next_group,0,window);
    
    int* other = (int*) malloc(sizeof(int)*size);
    
    int offset1 = 0 + N * ( 0 + N * 0);
    int offset2 = 10 + N * ( 10 + N * 10);

    if(rank == 3 || rank == 35){
        printf("starting the MPI_Gets\n"); fflush(stdout);
    }
    if(rank == 35 || rank == 3){
        MPI_Get(other+offset1,1,type2,next_rank,offset1,1,type2,window);
        
        MPI_Get(other+offset2,1,type2,next_rank,offset2,1,type2,window);

        //int offset3 = 5 + N * ( 2 + N * 7);
        //MPI_Get(other+offset,1,type2,next_rank,offset,1,type2,window);

        //int offset4 = 4 + N * ( 5 + N * 1);
        //MPI_Get(other+offset,1,type2,next_rank,offset,1,type2,window);

        MPI_Type_free(&type2);
    }
    if(rank == 3 || rank == 35){
        printf("starting the MPI_Win_complete\n"); fflush(stdout);
    }

    MPI_Win_complete(window);
    if(rank == 3 || rank == 35){
        printf("starting the MPI_Win_wait\n"); fflush(stdout);
    }
    MPI_Win_wait(window);


    if(rank == 35 || rank == 3){
        printf("I am rank %d and the next one is %d (answer = %d)\n",rank,other[offset1],next_rank); fflush(stdout);
        printf("I am rank %d and the next one is %d (answer = %d)\n",rank,other[offset2],next_rank); fflush(stdout);
    }


    // free
    if(rank == 3 || rank == 35){
        printf("free the groups\n"); fflush(stdout);
    }
    MPI_Group_free(&next_group);
    MPI_Group_free(&prev_group);

    if(rank == 3 || rank == 35){
        printf("free the window\n"); fflush(stdout);
    }
    MPI_Win_free(&window);
    
    if(rank == 3 || rank == 35){
        printf("free the arrays\n"); fflush(stdout);
    }
    free(array);
    free(other);


    if(rank == 3 || rank == 35){
        printf("Finalize\n"); fflush(stdout);
    }


    MPI_Finalize();
}
