#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

// communication modes
#define COMM_SENDRECV 0
#define COMM_RMA_ACTV 1
#define COMM_RMA_FENC 2

#ifdef COMM_MODE
#define M_COMM COMM_MODE
#else
#define M_COMM COMM_SENDRECV
#endif

// the total communication size will be N^3 and split-up in NS^3 chunks, each chunk will be sent out 
// independently
#define N 100 // N is the 1D size of the 3D array
#define NS 5 // NS is the size of the subarray (exchanged as one block)
#define IMAX 100

static int get_next_rank(const int rank, const int comm_size) {
  return (rank + comm_size / 2) % comm_size;
}
static int get_prev_rank(const int rank, const int comm_size) {
  return (rank + comm_size / 2) % comm_size;
}

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    //--------------------------------------------------------------------------
    // get usefull MPI information
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

    //// get the name and check if we are on the same node or not
    //char name[MPI_MAX_PROCESSOR_NAME];
    //int  len;
    //MPI_Get_processor_name(name, &len);
    //printf("[rank %d] the world is %d ranks wide and my name is %s\n", rank, comm_size, name);
    //fflush(stdout);

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
    //int n_in_group = (is_comm);
    //int next_rank  = (rank == rank_comm[0]) * rank_comm[1] + (rank == rank_comm[1]) * rank_comm[0];
    //int prev_rank  = (rank == rank_comm[0]) * rank_comm[1] + (rank == rank_comm[1]) * rank_comm[0];
    //if (is_comm) {
    //    printf("[rank %d] I will access rank %d and be accessed by rank %d\n", rank, next_rank, prev_rank);
    //    fflush(stdout);
    //}
    int n_in_group = 1;
    int prev_rank = get_prev_rank(rank, comm_size);
    int next_rank = get_next_rank(rank, comm_size);
    MPI_Group prev_group, next_group, global_group;
    MPI_Comm_group(MPI_COMM_WORLD, &global_group);
    MPI_Group_incl(global_group, n_in_group, &prev_rank, &prev_group);
    MPI_Group_incl(global_group, n_in_group, &next_rank, &next_group);
    MPI_Group_free(&global_group);

#if (M_COMM == COMM_SENDRECV)
        const int nreq = (N / NS) * (N / NS) * (N / NS);
        MPI_Request *sreq, *rreq;
        sreq = (MPI_Request *)malloc(sizeof(MPI_Request) * nreq);
        rreq = (MPI_Request *)malloc(sizeof(MPI_Request) * nreq);
#endif

    //--------------------------------------------------------------------------
    double mean_time;
    for (int iter = 0; iter < IMAX; ++iter) {
        double tic = MPI_Wtime();
#if (M_COMM == COMM_RMA_ACTV)
        MPI_Win_post(prev_group, 0, window);
        MPI_Win_start(next_group, 0, window);
#elif (M_COMM == COMM_RMA_FENC)
        MPI_Win_fence(0, window);
#endif

        // for each submatrix of size NSxNSxNS
        int ireq = 0;
        for (int i2 = 0; i2 < N; i2 += NS) {
          for (int i1 = 0; i1 < N; i1 += NS) {
            for (int i0 = 0; i0 < N; i0 += NS) {
              // get the datatype corresponding to the sub-array
              MPI_Datatype type1, type2;
              MPI_Type_create_hvector(NS, NS, N * sizeof(int), MPI_INT, &type1);
              MPI_Type_create_hvector(NS, 1, N * N * sizeof(int), type1,
                                      &type2);
              MPI_Type_commit(&type2);
              MPI_Type_free(&type1);

              int offset = i0 + N * (i1 + N * i2);
#if (M_COMM == COMM_RMA_ACTV || M_COMM == COMM_RMA_FENC)
              // do the MPI_Get
              MPI_Get(other + offset, 1, type2, next_rank, offset, 1, type2,
                      window);
#elif (M_COMM == COMM_SENDRECV)
              MPI_Irecv(other + offset, 1, type2, prev_rank, 0, MPI_COMM_WORLD,
                        rreq + ireq);
              MPI_Isend(array + offset, 1, type2, next_rank, 0, MPI_COMM_WORLD,
                        sreq + ireq);
              ireq++;
#endif

              // free the type
              MPI_Type_free(&type2);
            }
          }
        }

#if (M_COMM == COMM_RMA_ACTV)
        MPI_Win_complete(window);
        MPI_Win_wait(window);
#elif (M_COMM == COMM_RMA_FENC)
        MPI_Win_fence(0, window);
#elif (M_COMM == COMM_SENDRECV)
        MPI_Waitall(nreq, sreq, MPI_STATUSES_IGNORE);
        MPI_Waitall(nreq, rreq, MPI_STATUSES_IGNORE);
#endif
        double toc = MPI_Wtime();
        mean_time += (toc - tic) / IMAX;
    }

    double mtime_global;
    MPI_Reduce(&mean_time,&mtime_global,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    mtime_global /= comm_size;

    if (rank == 0) {
#if (M_COMM == COMM_RMA_ACTV)
        fprintf(stdout,
                "RMA-ACTIVE: mean send time = %e ms > bandwidth = %e GB/s\n",
                mtime_global * 1e+3,
                (N * N * N * sizeof(double)) / mtime_global * 1e+9);
#elif (M_COMM == COMM_RMA_FENC)
        fprintf(stdout,
                "RMA-FENCE: mean send time = %e ms > bandwidth = %e GB/s\n",
                mtime_global * 1e+3,
                (N * N * N * sizeof(double)) / mtime_global * 1e+9);
#elif (M_COMM == COMM_SENDRECV)
        fprintf(stdout,
                "SEND-RECV: mean send time = %e ms > bandwidth = %e GB/s\n",
                mtime_global * 1e+3,
                (N * N * N * sizeof(double)) / mtime_global * 1e+9);
#endif
        fflush(stdout);
    }
    //--------------------------------------------------------------------------
    // make sure we have the correct result
#if (M_COMM == COMM_SENDRECV)
    free(sreq);
    free(rreq);
#endif

    MPI_Group_free(&next_group);
    MPI_Group_free(&prev_group);
    MPI_Win_free(&window);

    free(array);
    free(other);

    MPI_Finalize();
}
