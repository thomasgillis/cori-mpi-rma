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

    int result;

    // associate the window with that array
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Win window = MPI_WIN_NULL;
    MPI_Win_create(&result, sizeof(int), sizeof(int), info, MPI_COMM_WORLD, &window);
    MPI_Info_free(&info);

    // create the communication groups
    MPI_Group group = MPI_GROUP_EMPTY;

    //--------------------------------------------------------------------------
    MPI_Win_post(group, 0, window);
    MPI_Win_start(group, 0, window);
    // do nothing, the group is empty

    MPI_Win_complete(window);
    MPI_Win_wait(window);

    MPI_Group_free(&group);
    MPI_Win_free(&window);


    MPI_Finalize();
}
