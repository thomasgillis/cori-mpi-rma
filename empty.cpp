#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    //--------------------------------------------------------------------------
    // get usefull MPI information
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

    // get a random information to share
    int result = 1729;

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

    // do not free the group
    //MPI_Group_free(&group);
    MPI_Win_free(&window);


    MPI_Finalize();
}
