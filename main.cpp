#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#define N 40
#define NS 10

static int rank_comm[2] = {3, 35};

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);

    // get usefull MPI information
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

    // only those two ranks actually perform communication
    const bool is_comm = (rank == rank_comm[0]) || (rank == rank_comm[1]);

    // get the name and check if we are on the same node or not
    char name[MPI_MAX_PROCESSOR_NAME];
    int  len;
    MPI_Get_processor_name(name, &len);
    if (is_comm) {
        printf("[rank %d] the world is %d ranks wide and my name is %s\n", rank, comm_size, name);
        fflush(stdout);
    }

    // allocate the array of size [N x N x N]
    int  size  = N * N * N;
    int* array = (int*)malloc(sizeof(int) * size);
    int* other = (int*)malloc(sizeof(int) * size);
    for (int i = 0; i < size; ++i) {
        array[i] = rank;
        other    = -1;
    }

    // associate the window with that array
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Win window = MPI_WIN_NULL;
    MPI_Win_create(array, size * sizeof(int), sizeof(int), info, MPI_COMM_WORLD, &window);
    MPI_Info_free(&info);

    // create the communication groups
    int n_in_group = (is_comm);
    int next_rank  = (rank == rank_comm[0]) * rank_comm[1] + (rank == rank_comm[1]) * rank_comm[0];
    int prev_rank  = (rank == rank_comm[0]) * rank_comm[1] + (rank == rank_comm[1]) * rank_comm[0];
    if (is_comm) {
        printf("[rank %d] I will access rank %d and be accessed by rank %d\n", rank, next_rank, prev_rank);
        fflush(stdout);
    }
    MPI_Group prev_group, next_group, global_group;
    MPI_Comm_group(MPI_COMM_WORLD, &global_group);
    MPI_Group_incl(global_group,n_in_group,&prev_rank,&prev_group);
    MPI_Group_incl(global_group,n_in_group,&next_rank,&next_group);
    MPI_Group_free(&global_group);

    // get the datatype to get the rank in an array of size [2,2,2]
    MPI_Datatype type1, type2;
    MPI_Type_create_hvector(NS, NS, N * sizeof(int), MPI_INT, &type1);
    MPI_Type_create_hvector(NS, 1, N * N * sizeof(int), type1, &type2);
    MPI_Type_commit(&type2);
    MPI_Type_free(&type1);

    // start
    MPI_Win_post(prev_group,0,window);
    MPI_Win_start(next_group,0,window);
    
    
    
    int offset1 = 0 + N * ( 0 + N * 0);
    int offset2 = 10 + N * ( 10 + N * 10);

    if (is_comm) {
        printf("starting the MPI_Gets\n");
        fflush(stdout);

        // get the first sub-block
        MPI_Get(other + offset1, 1, type2, next_rank, offset1, 1, type2, window);
        // and the second one
        MPI_Get(other + offset2, 1, type2, next_rank, offset2, 1, type2, window);
        // the type is now useless and can be destroyed
        MPI_Type_free(&type2);
    }

    if (is_comm) {
        printf("starting the MPI_Win_complete\n");
        fflush(stdout);
    }
    MPI_Win_complete(window);
    if (is_comm) {
        printf("starting the MPI_Win_wait\n");
        fflush(stdout);
    }
    MPI_Win_wait(window);

    if(is_comm){
        printf("I am rank %d and the other one is %d (answer = %d)\n",rank,other[offset1],next_rank); fflush(stdout);
        printf("I am rank %d and the other one is %d (answer = %d)\n",rank,other[offset2],next_rank); fflush(stdout);
    }

    // free
    if (is_comm) {
        printf("free the groups\n");
        fflush(stdout);
    }
    MPI_Group_free(&next_group);
    MPI_Group_free(&prev_group);

    if (is_comm) {
        printf("free the window\n");
        fflush(stdout);
    }
    MPI_Win_free(&window);

    if (is_comm) {
        printf("free the arrays\n");
        fflush(stdout);
    }
    free(array);
    free(other);

    if (is_comm) {
        printf("Finalize\n");
        fflush(stdout);
    }

    MPI_Finalize();
}
