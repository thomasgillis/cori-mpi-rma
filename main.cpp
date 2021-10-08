#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#define SCALABLE_SYNC

#define N 40 // N is the 1D size of the 3D array
#define NS 5 // NS is the size of the subarray (exchanged as one block)

// explicitly specify the two nodes that will communicate
// (it's not needed but convenient to investigate multiple nodes issues)
static int rank_comm[2] = {0, 1};


int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    //--------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------
    // allocate the array of size [N x N x N]
    int  size  = N * N * N;
    int* array = (int*)malloc(sizeof(int) * size);
    int* other = (int*)malloc(sizeof(int) * size);
    for (int i = 0; i < size; ++i) {
        array[i] = rank;
        other[i] = -1;
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
    MPI_Group_incl(global_group, n_in_group, &prev_rank, &prev_group);
    MPI_Group_incl(global_group, n_in_group, &next_rank, &next_group);
    MPI_Group_free(&global_group);

    //--------------------------------------------------------------------------

#ifdef SCALABLE_SYNC
    MPI_Win_post(prev_group, 0, window);
    MPI_Win_start(next_group, 0, window);
#else
    MPI_Win_fence(0, window);
#endif

    if (is_comm) {
        printf("starting the MPI_Gets\n");
        fflush(stdout);

        // for each submatrix of size NSxNSxNS
        for (int i2 = 0; i2 < N; i2 += NS) {
            for (int i1 = 0; i1 < N; i1 += NS) {
                for (int i0 = 0; i0 < N; i0 += NS) {
                    // get the datatype corresponding to the sub-array
                    MPI_Datatype type1, type2;
                    MPI_Type_create_hvector(NS, NS, N * sizeof(int), MPI_INT, &type1);
                    MPI_Type_create_hvector(NS, 1, N * N * sizeof(int), type1, &type2);
                    MPI_Type_commit(&type2);
                    MPI_Type_free(&type1);

                    // do the MPI_Get
                    int offset = i0 + N * (i1 + N * i2);
                    MPI_Get(other + offset, 1, type2, next_rank, offset, 1, type2, window);

                    // free the type
                    MPI_Type_free(&type2);
                }
            }
        }
    }

#ifdef SCALABLE_SYNC
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
#else
    MPI_Win_fence(0, window);
#endif
    //--------------------------------------------------------------------------
    // make sure we have the correct result
    if(is_comm){
        // randomly pick two elements
        printf("I am rank %d and the other one is %d (answer = %d)\n",rank,other[8+N*(3+N*3)],next_rank); fflush(stdout);
        printf("I am rank %d and the other one is %d (answer = %d)\n",rank,other[1+N*(0+N*9)],next_rank); fflush(stdout);
    }


    MPI_Group_free(&next_group);
    MPI_Group_free(&prev_group);
    MPI_Win_free(&window);

    free(array);
    free(other);

    MPI_Finalize();
}
