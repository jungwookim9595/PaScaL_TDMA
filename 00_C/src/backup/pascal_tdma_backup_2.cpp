#include <mpi.h>
#include <vector>
#include "iostream"
#include <numeric>
#include <cmath>
#include <omp.h>

#include "pascal_tdma.hpp"
#include "tdmas.hpp"
#include "../examples/mpi_subdomain.hpp"
#include "../examples/mpi_topology.hpp"
#include "para_range.hpp"

// Create a plan for a single tridiagonal system of equations.
void PaScaL_TDMA::PaScaL_TDMA_plan_single_create(ptdma_plan_single& plan, int myrank, int nprocs, MPI_Comm mpi_world, int gather_rank) {

    int nr_rd;      // Number of rows of a reduced tridiagonal system per process, 2
    int nr_rt;      // Number of rows of a reduced tridiagonal system after MPI_Gather
    
    nr_rd = 2;
    nr_rt = nr_rd*nprocs;

    plan.myrank = myrank;
    plan.nprocs = nprocs;
    plan.gather_rank = gather_rank;
    plan.ptdma_world = mpi_world;
    plan.n_row_rt = nr_rt;

    plan.A_rd.resize(nr_rd), plan.B_rd.resize(nr_rd), plan.C_rd.resize(nr_rd), plan.D_rd.resize(nr_rd);
    plan.A_rt.resize(nr_rt), plan.B_rt.resize(nr_rt), plan.C_rt.resize(nr_rt), plan.D_rt.resize(nr_rt);
}

void PaScaL_TDMA::PaScaL_TDMA_plan_single_destroy(ptdma_plan_single& plan) {
    // 원래는 A, B, C, D 를 deallocate를 해야하는데 cpp에서는 굳이 필요가 없음
    // 자동으로 메모리 해제 해준다.
}

void PaScaL_TDMA::PaScaL_TDMA_single_solve(ptdma_plan_single& plan, 
                                std::vector<double>& A, std::vector<double>& B, std::vector<double>& C, std::vector<double>& D, int n_row) {


    if (plan.nprocs==1) {
        tdma_single(A, B, C, D, n_row);
    }
    else {
        // MultiTimer timer;

        // 1) 초기 작업 및 reduced system 만든다.
        // A_rd, ... 여기에 저장
        // timer.start("prepare");

        A[0] = A[0]/B[0];
        D[0] = D[0]/B[0];
        C[0] = C[0]/B[0];

        A[1] = A[1]/B[1];
        D[1] = D[1]/B[1];
        C[1] = C[1]/B[1];
        
        double r;
        for (int i=2; i<n_row; ++i) {
            r = 1.0/(B[i] - A[i]*C[i-1]);
            D[i] = r*(D[i] - A[i]*D[i-1]);
            C[i] = r*C[i];
            A[i] = -r*A[i]*A[i-1];
        }

        // Reduction step : elimination of upper diagonal elements
        for (int i=n_row-3; i>=1; --i) {
            D[i] = D[i] - C[i]*D[i+1];
            A[i] = A[i] - C[i]*A[i+1];
            C[i] = -C[i]*C[i+1];
        }

        r = 1.0/(1.0 - A[1]*C[0]);
        D[0] = r*(D[0]-C[0]*D[1]);
        A[0] = r*A[0];
        C[0] = -r*C[0]*C[1];

        // 여기서 통신해서 푼다.

        // 2) MPI_Igather 로 reduced system 을 gather_rank 로 모은다.
        // A_rd -> A_rt 로 전송
        // A_rt, ... 여기에 저장

        // Construct a reduced tridiagonal system of equations per each rank. Each process has two reduced rows.
        plan.A_rd[0] = A[0];    plan.A_rd[1] = A[n_row-1];
        plan.B_rd[0] = 1.0;     plan.B_rd[1] = 1.0;
        plan.C_rd[0] = C[0];    plan.C_rd[1] = C[n_row-1];
        plan.D_rd[0] = D[0];    plan.D_rd[1] = D[n_row-1];
        // std::cout << "[prepare] elapsed: " << timer.elapsed_ns("prepare") << " ns\n";

        // Gather the coefficients of the reduced tridiagonal system to a defined rank, plan%gather_rank.
        // timer.start("Igather");
        std::vector<MPI_Request> request(4);
        MPI_Igather(plan.A_rd.data(), 2, MPI_DOUBLE, 
                    plan.A_rt.data(), 2, MPI_DOUBLE,
                    plan.gather_rank, plan.ptdma_world, &request[0]);
        MPI_Igather(plan.B_rd.data(), 2, MPI_DOUBLE, 
                    plan.B_rt.data(), 2, MPI_DOUBLE,
                    plan.gather_rank, plan.ptdma_world, &request[1]);
        MPI_Igather(plan.C_rd.data(), 2, MPI_DOUBLE, 
                    plan.C_rt.data(), 2, MPI_DOUBLE,
                    plan.gather_rank, plan.ptdma_world, &request[2]);
        MPI_Igather(plan.D_rd.data(), 2, MPI_DOUBLE, 
                    plan.D_rt.data(), 2, MPI_DOUBLE,
                    plan.gather_rank, plan.ptdma_world, &request[3]);
        MPI_Waitall(4, request.data(), MPI_STATUS_IGNORE);
        // std::cout << "[Igather] elapsed: " << timer.elapsed_ns("Igather") << " ns\n";

        // 3) gather_rank가 reduced system 을 tdma_single 으로 푼다.
        if (plan.myrank==plan.gather_rank) {
            // timer.start("solve_reduced");
            tdma_single(plan.A_rt, plan.B_rt, plan.C_rt, plan.D_rt, plan.n_row_rt);
            // std::cout << "[solve_reduced] elapsed: " << timer.elapsed_ns("solve_reduced") << " ns\n";
        };
        
        // 4) MPI_Iscatter로 각 랭크에 값을 뿌린다.
        // timer.start("Iscatter");
        MPI_Iscatter(plan.D_rt.data(), 2, MPI_DOUBLE,
                    plan.D_rd.data(), 2, MPI_DOUBLE,
                    plan.gather_rank, plan.ptdma_world, &request[0]);
        MPI_Waitall(1, request.data(), MPI_STATUS_IGNORE);
        // std::cout << "[Iscatter] elapsed: " << timer.elapsed_ns("Iscatter") << " ns\n";

        // 5) 이젠 그냥 알아서 푼다.
        // timer.start("solve_remain");
        D[0] = plan.D_rd[0];
        D[n_row-1] = plan.D_rd[1];
        for (int i=1; i<n_row-1; ++i) {
            D[i] = D[i] - A[i]*D[0] - C[i]*D[n_row-1];
        };
        // std::cout << "[solve_remain] elapsed: " << timer.elapsed_ns("solve_remain") << " ns\n";
    }
}

void PaScaL_TDMA::PaScaL_TDMA_single_solve_cycle(ptdma_plan_single& plan, 
                                std::vector<double>& A, std::vector<double>& B, std::vector<double>& C, std::vector<double>& D,
                                int n_row) {

    // The modified Thomas algorithm : elimination of lower diagonal elements.
    A[0] = A[0]/B[0];
    D[0] = D[0]/B[0];
    C[0] = C[0]/B[0];

    A[1] = A[1]/B[1];
    D[1] = D[1]/B[1];
    C[1] = C[1]/B[1];

    double r;
    for (int i=2; i<n_row; ++i) {
        r = 1/(B[i] - A[i]*C[i-1]);
        D[i] = r*(D[i] - A[i]*D[i-1]);
        C[i] = r*C[i];
        A[i] = -r*A[i]*A[i-1];
    }

    // The modified Thomas algorithm : elimination of upper diagonal elements.
    for (int i=n_row-3; i>=1; --i) {
        D[i] = D[i] - C[i]*D[i+1];
        A[i] = A[i] - C[i]*A[i+1];
        C[i] = -C[i]*C[i+1];
    }

    r = 1/(1 - A[1]*C[0]);
    D[0] = r*(D[0]-C[0]*D[1]);
    A[0] = r*A[0];
    C[0] = -r*C[0]*C[1];

    // Construct a reduced tridiagonal system of equations per each rank. Each process has two reduced rows.
    plan.A_rd[0] = A[0];    plan.A_rd[1] = A[n_row-1];
    plan.B_rd[0] = 1;       plan.B_rd[1] = 1;
    plan.C_rd[0] = C[0];    plan.C_rd[1] = C[n_row-1];
    plan.D_rd[0] = D[0];    plan.D_rd[1] = D[n_row-1];

    // Gather the coefficients of the reduced tridiagonal system to a defined rank, plan%gather_rank.
    std::vector<MPI_Request> request(4);

    MPI_Igather(plan.A_rd.data(), 2, MPI_DOUBLE, 
                plan.A_rt.data(), 2, MPI_DOUBLE,
                plan.gather_rank, plan.ptdma_world, &request[0]);
    MPI_Igather(plan.B_rd.data(), 2, MPI_DOUBLE, 
                plan.B_rt.data(), 2, MPI_DOUBLE,
                plan.gather_rank, plan.ptdma_world, &request[1]);
    MPI_Igather(plan.C_rd.data(), 2, MPI_DOUBLE, 
                plan.C_rt.data(), 2, MPI_DOUBLE,
                plan.gather_rank, plan.ptdma_world, &request[2]);
    MPI_Igather(plan.D_rd.data(), 2, MPI_DOUBLE, 
                plan.D_rt.data(), 2, MPI_DOUBLE,
                plan.gather_rank, plan.ptdma_world, &request[3]);
    MPI_Waitall(4, request.data(), MPI_STATUS_IGNORE);

    // Solve the reduced cyclic tridiagonal system on plan%gather_rank.
    if (plan.myrank==plan.gather_rank) {
        tdma_cycl_single(plan.A_rt, plan.B_rt, plan.C_rt, plan.D_rt, plan.n_row_rt);
    };

    // Distribute the solutions to each rank.
    MPI_Iscatter(plan.D_rt.data(), 2, MPI_DOUBLE,
                plan.D_rd.data(), 2, MPI_DOUBLE,
                plan.gather_rank, plan.ptdma_world, &request[0]);
                
    MPI_Waitall(1, request.data(), MPI_STATUS_IGNORE);

    // Update solutions of the modified tridiagonal system with the solutions of the reduced tridiagonal system.
    D[0] = plan.D_rd[0];
    D[n_row-1] = plan.D_rd[1];
    for (int i=1; i<n_row-1; ++i) {
        D[i] = D[i] - A[i]*D[0] - C[i]*D[n_row-1];
    };
}

// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Create a plan for a single tridiagonal system of equations.
void PaScaL_TDMA::PaScaL_TDMA_plan_many_create(ptdma_plan_many& plan, int n_sys, int myrank, int nprocs, MPI_Comm mpi_world) {

    int i;
    int ista, iend;                                         // First and last indices of assigned range in many tridiagonal systems of equations 
    std::vector<int> bigsize(2), subsize(2), start(2);      // Temporary variables of derived data type (DDT)
    int ns_rd, nr_rd;                                       // Dimensions of many reduced tridiagonal systems
    int ns_rt, nr_rt;                                       // Dimensions of many reduced tridiagonal systems after transpose
    std::vector<int> ns_rt_array(nprocs);                   // Array specifying the number of tridiagonal systems for each process after transpose

    plan.nprocs = nprocs;

    // Specify dimensions for reduced systems.
    ns_rd = n_sys;
    nr_rd = 2;

    // Specify dimensions for reduced systems after transpose.
    // ns_rt         : divide the number of tridiagonal systems of equations per each process  
    // ns_rt_array   : save the ns_rt in ns_rt_array for defining the DDT
    // nr_rt         : dimensions of the reduced tridiagonal systems in the solving direction, nr_rd*nprocs
    para_range(1, ns_rd, nprocs, myrank, ista, iend);
    ns_rt = iend - ista + 1;
    MPI_Allgather(&ns_rt, 1, MPI_INT,
                  ns_rt_array.data(), 1, MPI_INT,
                  mpi_world);
    nr_rt = nr_rd*nprocs;

    // Assign plan variables and allocate coefficient arrays.
    plan.n_sys_rt = ns_rt;
    plan.n_row_rt = nr_rt;
    plan.ptdma_world = mpi_world;

    plan.A_rd.resize(nr_rd * ns_rd);
    plan.B_rd.resize(nr_rd * ns_rd);
    plan.C_rd.resize(nr_rd * ns_rd);
    plan.D_rd.resize(nr_rd * ns_rd);

    plan.A_rt.resize(nr_rt * ns_rt);
    plan.B_rt.resize(nr_rt * ns_rt);
    plan.C_rt.resize(nr_rt * ns_rt);
    plan.D_rt.resize(nr_rt * ns_rt);

    // Building the DDTs.
    plan.ddtype_Fs.resize(nprocs), plan.ddtype_Bs.resize(nprocs);
    int sum = 0;
    for (i=0; i<nprocs; ++i) {
        // DDT for sending coefficients of the reduced tridiagonal systems using MPI_Ialltoallw communication.
        bigsize[1] = ns_rd;
        bigsize[0] = nr_rd;
        subsize[1] = ns_rt_array[i];
        subsize[0] = nr_rd;
        start[1] = sum;
        sum += ns_rt_array[i];
        start[0] = 0;
        MPI_Type_create_subarray(2, bigsize.data(), subsize.data(), start.data(),
                                        MPI_ORDER_C, MPI_DOUBLE,
                                        &plan.ddtype_Fs[i]);
        MPI_Type_commit(&plan.ddtype_Fs[i]);
        
        // DDT for receiving coefficients for the transposed systems of reduction using MPI_Ialltoallw communication.
        bigsize[1] = ns_rt;
        bigsize[0] = nr_rt;
        subsize[1] = ns_rt;
        subsize[0] = nr_rd;
        start[1] = 0;
        start[0] = nr_rd*i;
        MPI_Type_create_subarray(2, bigsize.data(), subsize.data(), start.data(), 
                                        MPI_ORDER_C, MPI_DOUBLE,
                                        &plan.ddtype_Bs[i]);
        MPI_Type_commit(&plan.ddtype_Bs[i]);
    }

    // Buffer counts and displacements for MPI_Ialltoallw.
    // All buffer counts are 1 and displacements are 0 due to the defined DDT.
    // resize + init 이랑 re declare 성능을 비교 했는데 후자가 조금 더 빠름 (init도 같이 진행하기 때문)
    plan.count_send = std::vector<int>(nprocs, 1);
    plan.displ_send = std::vector<int>(nprocs, 0);
    plan.count_recv = std::vector<int>(nprocs, 1);
    plan.displ_recv = std::vector<int>(nprocs, 0);     

    // Deallocate local array.
    if (!ns_rt_array.empty()) {
        ns_rt_array.clear();         // 내용 제거
        ns_rt_array.shrink_to_fit(); // capacity까지 제거
    }
}

void PaScaL_TDMA::PaScaL_TDMA_plan_many_destroy(ptdma_plan_many& plan, int nprocs) {

    for (int i=0; i<nprocs; ++i) {
        MPI_Type_free(&plan.ddtype_Fs[i]);
        MPI_Type_free(&plan.ddtype_Bs[i]);
    }

    plan.ddtype_Fs.shrink_to_fit(); plan.ddtype_Fs.shrink_to_fit();
    plan.count_send.shrink_to_fit(); plan.displ_send.shrink_to_fit();
    plan.count_recv.shrink_to_fit(); plan.displ_recv.shrink_to_fit();
    plan.A_rd.shrink_to_fit(); plan.B_rd.shrink_to_fit(); plan.C_rd.shrink_to_fit(); plan.D_rd.shrink_to_fit();
    plan.A_rt.shrink_to_fit(); plan.B_rt.shrink_to_fit(); plan.C_rt.shrink_to_fit(); plan.D_rt.shrink_to_fit();
}

void PaScaL_TDMA::PaScaL_TDMA_many_solve(ptdma_plan_many& plan,
                                         std::vector<double>& A_,
                                         std::vector<double>& B_,
                                         std::vector<double>& C_,
                                         std::vector<double>& D_,
                                         int n_sys, int n_row) {

    std::vector<MPI_Request> request(4);

    if (plan.nprocs == 1) {
        tdma_many(A_, B_, C_, D_, n_sys, n_row);
        return;
    }
    
    // ========================================================================
    // 변수 선언 최적화 적용
    // ========================================================================
    double* __restrict A = A_.data();
    double* __restrict B = B_.data();
    double* __restrict C = C_.data();
    double* __restrict D = D_.data();

    int i, j, idx, idx_jp, idx_jm;
    double r;

    // --- Preprocess ---
    for (i=0; i<n_sys; ++i) {
        
        int i0 = 0 * n_sys + i;
        int i1 = 1 * n_sys + i;
        
        // j=0
        r = 1.0 / B[i0];
        A[i0] *= r; D[i0] *= r; C[i0] *= r;

        // j=1
        r = 1.0 / B[i1];
        A[i1] *= r; D[i1] *= r; C[i1] *= r;

    }

    // --- Forward Elimination ---
    for (j=2; j<n_row; ++j) {
        for (i=0; i<n_sys; ++i) {
            idx    = j*n_sys + i;
            idx_jm = (j-1)*n_sys + i;

            r = 1.0 / (B[idx] - A[idx] * C[idx_jm]);
            D[idx] =  r * (D[idx] - A[idx] * D[idx_jm]);
            C[idx] =  r * C[idx];
            A[idx] = -r * A[idx] * A[idx_jm];
        }
    }

    // --- Backward Substitution ---
    for (j=n_row-3; j>=1; --j) {
        for (i=0; i<n_sys; ++i) {
            idx    = j * n_sys + i;
            idx_jp = (j + 1) * n_sys + i;

            D[idx] -= C[idx] * D[idx_jp];
            A[idx] -= C[idx] * A[idx_jp];
            C[idx] *= -C[idx_jp];
        }
    }

    // --- Pack ---
    for (i = 0; i < n_sys; ++i) {

        int idx0 = 0 * n_sys + i;
        int idx1 = 1 * n_sys + i;
        int idxN = (n_row - 1) * n_sys + i;

        r = 1.0 / (1.0 - A[idx1] * C[idx0]);
        D[idx0] = r * (D[idx0] - C[idx0] * D[idx1]);
        A[idx0] = r * A[idx0];
        C[idx0] = -r * C[idx0] * C[idx1];

        plan.A_rd[idx0] = A[idx0];
        plan.A_rd[idx1] = A[idxN];
        plan.B_rd[idx0] = 1.0;
        plan.B_rd[idx1] = 1.0;
        plan.C_rd[idx0] = C[idx0];
        plan.C_rd[idx1] = C[idxN];
        plan.D_rd[idx0] = D[idx0];
        plan.D_rd[idx1] = D[idxN];
    }
    
    // (MPI 통신)
    MPI_Ialltoallw(plan.A_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                   plan.A_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                   plan.ptdma_world, &request[0]);
    MPI_Ialltoallw(plan.B_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                   plan.B_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                   plan.ptdma_world, &request[1]);
    MPI_Ialltoallw(plan.C_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                   plan.C_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                   plan.ptdma_world, &request[2]);
    MPI_Ialltoallw(plan.D_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                   plan.D_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                   plan.ptdma_world, &request[3]);
    MPI_Waitall(4, request.data(), MPI_STATUSES_IGNORE);

    tdma_many(plan.A_rt, plan.B_rt, plan.C_rt, plan.D_rt, plan.n_sys_rt, plan.n_row_rt);

    MPI_Ialltoallw(plan.D_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                   plan.D_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                   plan.ptdma_world, &request[0]);
    MPI_Waitall(1, request.data(), MPI_STATUSES_IGNORE);

    // --- Final Local Solve ---
    for (i = 0; i < n_sys; ++i) {
        D[0 * n_sys + i] = plan.D_rd[0 * (n_sys) + i];
        D[(n_row - 1) * n_sys + i] = plan.D_rd[1 * (n_sys) + i];
    }
    for (j = 1; j < n_row - 1; ++j) {
        for (i = 0; i < n_sys; ++i) {
            D[j * n_sys + i] -= A[j * n_sys + i] * D[0 * n_sys + i] + C[j * n_sys + i] * D[(n_row - 1) * n_sys + i];
        }
    }
}

void PaScaL_TDMA::PaScaL_TDMA_many_solve_cycle(ptdma_plan_many& plan,
                                std::vector<double>& A, 
                                std::vector<double>& B, 
                                std::vector<double>& C, 
                                std::vector<double>& D,
                                int n_sys, int n_row) {
    
        // Temporary variables for computation and parameters for MPI functions.
    int i, j;
    std::vector<MPI_Request> request(4);
    double r;
    int idx;

    for (j = 0; j < n_sys; ++j) {
        idx = j * n_row + 0;
        A[idx] /= B[idx];
        D[idx] /= B[idx];
        C[idx] /= B[idx];

        idx = j * n_row + 1;
        A[idx] /= B[idx];
        D[idx] /= B[idx];
        C[idx] /= B[idx];
    }

    for (j = 0; j < n_sys; ++j) {
        for (i = 2; i < n_row; ++i) {
            idx = j * n_row + i;
            r = 1.0 / (B[idx] - A[idx]*C[idx - 1]);
            D[idx] = r * (D[idx] - A[idx]*D[idx - 1]);
            C[idx] = r * C[idx];
            A[idx] = -r * A[idx] * A[idx - 1];
        }
    }

    for (j = 0; j < n_sys; ++j) {
        for (i = n_row - 3; i >= 1; --i) {
            idx = j * n_row + i;

            D[idx] -= C[idx] * D[idx + 1];
            A[idx] -= C[idx] * A[idx + 1];
            C[idx] = -C[idx] * C[idx + 1];
        }
    }

    for (j = 0; j < n_sys; ++j) {
        idx = j*n_row + 0;

        r = 1.0 / (1.0 - A[idx + 1] * C[idx]);
        D[idx] = r * (D[idx] - C[idx] * D[idx + 1]);
        A[idx] *= r;
        C[idx] = -r * C[idx] * C[idx + 1];

        plan.A_rd[j * 2 + 0] = A[idx];
        plan.A_rd[j * 2 + 1] = A[idx + n_row-1];
        plan.B_rd[j * 2 + 0] = 1.0;
        plan.B_rd[j * 2 + 1] = 1.0;
        plan.C_rd[j * 2 + 0] = C[idx];
        plan.C_rd[j * 2 + 1] = C[idx + n_row-1];
        plan.D_rd[j * 2 + 0] = D[idx];
        plan.D_rd[j * 2 + 1] = D[idx + n_row-1];
    }
    
    MPI_Ialltoallw(plan.A_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                    plan.A_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                    plan.ptdma_world, &request[0]);
    MPI_Ialltoallw(plan.B_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                    plan.B_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                    plan.ptdma_world, &request[1]);
    MPI_Ialltoallw(plan.C_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                    plan.C_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                    plan.ptdma_world, &request[2]);     
    MPI_Ialltoallw(plan.D_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                    plan.D_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                    plan.ptdma_world, &request[3]);
   
    MPI_Waitall(4, request.data(), MPI_STATUSES_IGNORE);

    tdma_cycl_many(plan.A_rt, plan.B_rt, plan.C_rt, plan.D_rt, plan.n_sys_rt, plan.n_row_rt);

    // 일단 이렇게 하면 작동은 잘 되는데 왜 잘 작동하는지는 모르겠다
    // D_rt -> D_rd의 경우 보내는 입장이 바뀌었기 때문에 ddtype을 (ddtype_Bs, ddtype_F) 이렇게 사용해야 하는줄 알았는데 반대로 사용해야 잘 작동함.
    MPI_Ialltoallw(plan.D_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                   plan.D_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                   plan.ptdma_world, &request[0]);        

    MPI_Waitall(1, request.data(), MPI_STATUSES_IGNORE);

    for (j = 0; j < n_sys; ++j) {
        idx = j*n_row + 0;

        D[idx] = plan.D_rd[j*2 + 0];
        D[idx + n_row-1] = plan.D_rd[j*2 + 1];
    }
    for (j = 1; j < n_row - 1; ++j) {
        const double* __restrict d0 = D; // 0번째 행
        const double* __restrict dn = D + (size_t)(n_row - 1) * n_sys; // 마지막 행

        double* __restrict dj = D + (size_t)j * n_sys;
        const double* __restrict aj = A + (size_t)j * n_sys;
        const double* __restrict cj = C + (size_t)j * n_sys;

        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            dj[i] -= aj[i] * d0[i] + cj[i] * dn[i];
        }
    }

    // for (j = 0; j < n_sys; ++j) {
    //     for (i = 1; i < n_row-1; ++i) {
    //         idx = j*n_row + 0;
    //         D[idx + i] -= A[idx + i] * D[idx] + C[idx + i] * D[idx + n_row-1];
    //     }
    // }
}