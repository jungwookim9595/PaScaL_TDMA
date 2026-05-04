#include <mpi.h>
#include <vector>

class PaScaL_TDMA {
public:
    struct ptdma_plan_single {
        MPI_Comm  ptdma_world;    // Single dimensional subcommunicator to assemble data for the reduced TDMA
        int n_row_rt;       // Number of rows of a reduced tridiagonal system after MPI_Gather
        
        int gather_rank;    // Destination rank of MPI_Igather
        int myrank;         // Current rank ID in the communicator of ptdma_world
        int nprocs;         // Communicator size of ptdma_world
        
        // Coefficient arrays after reduction, a: lower, b: diagonal, c: upper, d: rhs.
        std::vector<double> A_rd, B_rd, C_rd, D_rd;

        // Coefficient arrays after transpose of a reduced system, a: lower, b: diagonal, c: upper, d: rhs
        std::vector<double> A_rt, B_rt, C_rt, D_rt;

    };

    void PaScaL_TDMA_plan_single_create(ptdma_plan_single& plan,
                                        int myrank, int nprocs,
                                        MPI_Comm mpi_world,
                                        int gather_rank);
                                        
    void PaScaL_TDMA_plan_single_destroy(ptdma_plan_single& plan);

    void PaScaL_TDMA_single_solve(ptdma_plan_single& plan, 
                                  std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& d,
                                  int n_row);

    void PaScaL_TDMA_single_solve_debug(ptdma_plan_single& plan, 
                                  std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& d,
                                  int n_row, std::vector<double>& time_list);
    
    void PaScaL_TDMA_single_solve_cycle(ptdma_plan_single& plan, 
                                std::vector<double>& A, std::vector<double>& B, std::vector<double>& C, std::vector<double>& D,
                                int n_row);

    struct ptdma_plan_many {
        MPI_Comm  ptdma_world;
        int n_sys_rt;
        int n_row_rt;

        int nprocs;

        // Send buffer related variables for MPI_Ialltoallw
        // ddtype_FS: A, B, C, D 계수들을 보내는 buffer
        // count_send: alltoall 에서 i 번째에 보내는 데이터 개수
        // displ_send: alltoall 에서 i 번째에 보내는 데이터 주소
        std::vector<MPI_Datatype> ddtype_Fs;
        std::vector<int> count_send;
        std::vector<int> displ_send;

        // Recv. buffer related variables MPI_Ialltoallw 
        // ddtype_Bs: A, B, C, D 계수들을 받는 buffer
        // count_recv: alltoall 에서 i 번째로 부터 받는 데이터 개수
        // displ_recv: alltoall 에서 i 번째로 부터 받는 데이터 주소
        std::vector<MPI_Datatype> ddtype_Bs;
        std::vector<int> count_recv;
        std::vector<int> displ_recv;

        // Coefficient arrays after reduction, a: lower, b: diagonal, c: upper, d: rhs.
        // The orginal dimension (m:n) is reduced to (m:2)
        // singel 이였으면 m을 순차적으로 풀었지만 여기서는 한번에 모은다.
        std::vector<double> A_rd, B_rd, C_rd, D_rd;

        // Coefficient arrays after reduction, a: lower, b: diagonal, c: upper, d: rhs.
        // The reduced dimension (m:2) changes to (m/np: 2*np) after transpose.
        std::vector<double> A_rt, B_rt, C_rt, D_rt;

    };

    void PaScaL_TDMA_plan_many_create(ptdma_plan_many& plan, int n_sys, int myrank, int nprocs, MPI_Comm mpi_world);

    void PaScaL_TDMA_plan_many_destroy(ptdma_plan_many& plan, int nprocs);

    void PaScaL_TDMA_many_solve(ptdma_plan_many& plan,
                                double* __restrict A,
                                double* __restrict B,
                                double* __restrict C,
                                double* __restrict D,
                                int n_sys, int n_row);

    void PaScaL_TDMA_many_solve_debug(ptdma_plan_many& plan,
                                        double* __restrict A,
                                        double* __restrict B,
                                        double* __restrict C,
                                        double* __restrict D,
                                        int n_sys, int n_row,
                                        std::vector<double>& time_list);

    void PaScaL_TDMA_many_solve_cycle(ptdma_plan_many& plan,
                                    std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& d,
                                    int n_sys, int n_row);
};
