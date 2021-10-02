#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>


int main(int argc, char** argv){
    MPI_Init(&argc,&argv);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name( name, &len );

    printf("[rank %d] the world is %d ranks wide and my name is %s\n",rank,comm_size,name);

    // allocate the array
    int* array = (int*) malloc(sizeof(int)*1);
    array[0] = rank;


    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Win  window = MPI_WIN_NULL;
    MPI_Win_create(array,sizeof(int), sizeof(int), info, MPI_COMM_WORLD, &window);
    MPI_Info_free(&info);

    // create the group
    MPI_Group prev_group, next_group, global_group;
    MPI_Comm_group(MPI_COMM_WORLD, &global_group);
    int next_rank = (comm_size + rank + 1)%comm_size;
    int prev_rank = (comm_size + rank - 1)%comm_size;
    printf("[rank %d] I am between rank %d and rank %d\n",rank,prev_rank,next_rank);
    
    MPI_Group_incl(global_group,1,&prev_rank,&prev_group);
    MPI_Group_incl(global_group,1,&next_rank,&next_group);
    MPI_Group_free(&global_group);



    // start
    MPI_Win_post(prev_group,0,window);
    MPI_Win_start(next_group,0,window);
    
    int other;
    MPI_Get(&other,1,MPI_INT,next_rank,0,1,MPI_INT,window);

    MPI_Win_complete(window);
    MPI_Win_wait(window);


    printf("I am rank %d and the next one is %d (answer = %d)\n",rank,other,next_rank);


    // free
    MPI_Group_free(&next_group);
    MPI_Group_free(&prev_group);
    MPI_Win_free(&window);
    free(array);



    MPI_Finalize();
}
