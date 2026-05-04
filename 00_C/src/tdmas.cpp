#include <vector>
#include <iostream>
#include <cmath>
#include "tdmas.hpp"
#include <mpi.h>

void tdma_single(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& d, int n1) {

    int i;
    double r;

    d[0] = d[0]/b[0];
    c[0] = c[0]/b[0];

    for (i=1; i<n1; ++i) {
        r = 1.0/(b[i]-a[i]*c[i-1]);
        d[i] = r*(d[i]-a[i]*d[i-1]);
        c[i] = r*c[i];
    }

    for (i=n1-2; i>=0; --i) {
        d[i] = d[i]-c[i]*d[i+1];
    }
}

void tdma_cycl_single(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& d, int n1) {

    int i;
    double rr;
    // n1 = n_row

    std::vector<double> e(n1, 0.0);
    e[1] = -a[1];
    e[n1-1] = -c[n1-1];

    d[1] = d[1]/b[1];
    e[1] = e[1]/b[1];
    c[1] = c[1]/b[1];

    for (i=2; i<=n1-1; ++i) {
        rr = 1.0/(b[i]-a[i]*c[i-1]);
        d[i] = rr*(d[i]-a[i]*d[i-1]);
        e[i] = rr*(e[i]-a[i]*e[i-1]);
        c[i] = rr*c[i];
    }

    for (i=n1-2; i>=1; --i) {
        d[i] = d[i]-c[i]*d[i+1];
        e[i] = e[i]-c[i]*e[i+1];
    }

    d[0] = (d[0] - a[0]*d[n1-1] - c[0]*d[1])/(b[0] + a[0]*e[n1-1] + c[0]*e[1]);

    for (i=1; i<=n1-1; ++i) {
        d[i] = d[i] + d[0]*e[i];
    }
}

// void tdma_many(
//     std::vector<double>& A_, std::vector<double>& B_,
//     std::vector<double>& C_, std::vector<double>& D_,
//     int n1, int n2) {
//     // n1: n_sys
//     // n2: n_row

//     double* __restrict A = A_.data();
//     double* __restrict B = B_.data();
//     double* __restrict C = C_.data();
//     double* __restrict D = D_.data();

//     // 임시 변수 r을 위한 벡터 (Fortran의 r(1:n1))
//     std::vector<double> r(n1);

//     // --- Preprocess (i=0) ---
//     for (int i = 0; i < n1; ++i) {
//         double inv_b = 1.0 / B[0 * n1 + i];
//         D[0 * n1 + i] *= inv_b;
//         C[0 * n1 + i] *= inv_b;
//     }

//     // --- Forward Elimination ---
//     for (int j = 1; j < n2; ++j) {
//         for (int i = 0; i < n1; ++i) {
//             int current_idx = j * n1 + i;
//             int prev_idx_i  = (j - 1) * n1 + i;
            
//             r[i] = 1.0 / (B[current_idx] - A[current_idx] * C[prev_idx_i]);
//             D[current_idx] = r[i] * (D[current_idx] - A[current_idx] * D[prev_idx_i]);
//             C[current_idx] = r[i] * C[current_idx];
//         }
//     }

//     // --- Backward Substitution ---
//     for (int j = n2 - 2; j >= 0; --j) {
//         for (int i = 0; i < n1; ++i) {
//             int current_idx = j * n1 + i;
//             int next_idx_i  = (j + 1) * n1 + i;
//             D[current_idx] -= C[current_idx] * D[next_idx_i];
//         }
//     }
// }

// void tdma_many(
//     double* __restrict A,
//     double* __restrict B,
//     double* __restrict C,
//     double* __restrict D,
//     int n1, int n2)
// {

//     // --- Preprocess (row 0): i-loop 벡터화 가능 ---
//     {
//         double* Crow = C + 0 * n1;
//         double* Brow = B + 0 * n1;
//         double* Drow = D + 0 * n1;

//         #pragma omp simd
//         for (int i = 0; i < n1; ++i) {
//             const double inv_b = 1.0 / Brow[i];
//             Drow[i] *= inv_b;
//             Crow[i] *= inv_b;
//         }
//     }

//     // --- Forward Elimination: j = 1 .. n2-1 ---
//     for (int j = 1; j < n2; ++j) {
//         const double* Acur = A + j * n1;
//         double*       Bcur = B + j * n1;
//         double*       Ccur = C + j * n1;
//         double*       Dcur = D + j * n1;

//         const double* Cprev = C + (j - 1) * n1;
//         const double* Dprev = D + (j - 1) * n1;

//         #pragma omp simd
//         for (int i = 0; i < n1; ++i) {
//             const double denom = Bcur[i] - Acur[i] * Cprev[i];
//             const double rinv  = 1.0 / denom;
//             Dcur[i] = rinv * (Dcur[i] - Acur[i] * Dprev[i]);
//             Ccur[i] = rinv *  Ccur[i];
//         }
//     }

//     // --- Backward Substitution: j = n2-2 .. 0 ---
//     for (int j = n2 - 2; j >= 0; --j) {
//         double*       Ccur = C + j * n1;
//         double*       Dcur = D + j * n1;
//         const double* Dnxt = D + (j + 1) * n1;

//         #pragma omp simd
//         for (int i = 0; i < n1; ++i) {
//             Dcur[i] -= Ccur[i] * Dnxt[i];
//         }
//     }
// }

void tdma_many(
    double* __restrict A,
    double* __restrict B,
    double* __restrict C,
    double* __restrict D,
    int n1, int n2) {
    // n1: n_sys
    // n2: n_row

    // resriction ----------------------------------------------------
    double* B0 = B + (size_t)(0)*n1;
    double* C0 = C + (size_t)(0)*n1;
    double* D0 = D + (size_t)(0)*n1;

    // 임시 변수 r을 위한 벡터 (Fortran의 r(1:n1))
    std::vector<double> r(n1);
    
    // --- Preprocess (i=0) ---
    #pragma omp simd
    for (int i = 0; i < n1; ++i) {
        double inv_b = 1.0 / B0[i];
        D0[i] *= inv_b;
        C0[i] *= inv_b;
    }

    // --- Forward Elimination ---
    for (int j = 1; j < n2; ++j) {

        double* Aj = A + (size_t)j*n1;
        double* Bj = B + (size_t)j*n1;
        double* Cj = C + (size_t)j*n1;
        double* Dj = D + (size_t)j*n1;

        const double* Cjm = C + (size_t)(j-1)*n1;
        const double* Djm = D + (size_t)(j-1)*n1;

        #pragma omp simd
        for (int i = 0; i < n1; ++i) {
            
            r[i] = 1.0 / (Bj[i] - Aj[i]*Cjm[i]);
            Dj[i] = r[i] * (Dj[i] - Aj[i] * Djm[i]);
            Cj[i] = r[i] * Cj[i];           
        }
    }

    // --- Backward Substitution ---
    for (int j = n2 - 2; j >= 0; --j) {

        double* Cj = C + (size_t)j*n1;
        double* Dj = D + (size_t)j*n1;

        const double* Djp = D + (size_t)(j+1)*n1;

        #pragma omp simd
        for (int i = 0; i < n1; ++i) {

            Dj[i] -= Cj[i] * Djp[i];
        }
    }
}

void tdma_many_debug(
    double* __restrict A,
    double* __restrict B,
    double* __restrict C,
    double* __restrict D,
    int n1, int n2, std::vector<double>& time_list) {
    // n1: n_sys
    // n2: n_row

    double forward_1, forward_2;
    double backward_1, backward_2;
    // double pack_1, pack_2;
    // double scatter_1, scatter_2;
    // double solve_1, solve_2;
    // double gather_1, gather_2;
    // double final_solve_1, final_solve_2;

    // resriction ----------------------------------------------------
    double* B0 = B + (size_t)(0)*n1;
    double* C0 = C + (size_t)(0)*n1;
    double* D0 = D + (size_t)(0)*n1;

    // 임시 변수 r을 위한 벡터 (Fortran의 r(1:n1))
    std::vector<double> r(n1);
    
    MPI_Barrier(MPI_COMM_WORLD);
    forward_1 = MPI_Wtime();
    // --- Preprocess (i=0) ---
    #pragma omp simd
    for (int i = 0; i < n1; ++i) {
        double inv_b = 1.0 / B0[i];
        D0[i] *= inv_b;
        C0[i] *= inv_b;
    }

    // --- Forward Elimination ---
    for (int j = 1; j < n2; ++j) {

        double* Aj = A + (size_t)j*n1;
        double* Bj = B + (size_t)j*n1;
        double* Cj = C + (size_t)j*n1;
        double* Dj = D + (size_t)j*n1;

        const double* Cjm = C + (size_t)(j-1)*n1;
        const double* Djm = D + (size_t)(j-1)*n1;

        #pragma omp simd
        for (int i = 0; i < n1; ++i) {
            
            r[i] = 1.0 / (Bj[i] - Aj[i]*Cjm[i]);
            Dj[i] = r[i] * (Dj[i] - Aj[i] * Djm[i]);
            Cj[i] = r[i] * Cj[i];           
        }
    }
    forward_2 = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    backward_1 = MPI_Wtime();
    // --- Backward Substitution ---
    for (int j = n2 - 2; j >= 0; --j) {

        double* Cj = C + (size_t)j*n1;
        double* Dj = D + (size_t)j*n1;

        const double* Djp = D + (size_t)(j+1)*n1;

        #pragma omp simd
        for (int i = 0; i < n1; ++i) {

            Dj[i] -= Cj[i] * Djp[i];
        }
    }
    backward_2 = MPI_Wtime();

    time_list.resize(7);
    time_list[0] = forward_2 - forward_1;
    time_list[1] = backward_2 - backward_1;
    time_list[2] = 0;
    time_list[3] = 0;
    time_list[4] = 0;
    time_list[5] = 0;
    time_list[6] = 0;
}

void tdma_cycl_many(
    std::vector<double> &A_,
    std::vector<double> &B_,
    std::vector<double> &C_,
    std::vector<double> &D_,
    int n1, int n2) {
    // n1: n_sys
    // n2: n_row

    double* __restrict A = A_.data();
    double* __restrict B = B_.data();
    double* __restrict C = C_.data();
    double* __restrict D = D_.data();

    std::vector<double> r(n1);
    std::vector<double> e(n1*n2, 0.0);
    int idx, idx1, idxN, idx_jp, idx_jm;
    int i, j;
    
    
    for (j=0; j<n2; ++j) {
        idx1 = j*n1 + 1;
        e[idx1] = -A[idx1];

        idxN = j*n1 + (n1-1);
        e[idxN] = -A[idxN];
    }
    
    // --- Preprocess (i=1) ---
    for (j=0; j<n2; ++j) {
        idx1 = j*n1 + 1;

        double inv_b = 1.0 / B[idx1];
        D[idx1] *= inv_b;
        e[idx1] *= inv_b;
        C[idx1] *= inv_b;
    }

    // --- Forward Elimination ---
    for (j=2; j<n2; ++j) {
        for (i=0; i<n1; ++i) {
            idx    = j*n1 + i;
            idx_jm = (j-1)*n1 + i;

            r[i] = 1.0 / (B[idx] - A[idx] * C[idx_jm]);
            D[idx] = r[i] * (D[idx] - A[idx] * D[idx_jm]);
            e[idx] = r[i] * (e[idx] - A[idx] * D[idx_jm]);
            C[idx] = r[i] * C[idx];
        }
    }

    // --- Backward Substitution ---
    for (j=2; j<n2; ++j) {
        for (i=0; i<n1; ++i) {
            idx    = j*n1 + i;
            idx_jp = (j+1)*n1 + i;

            D[idx] = D[idx] - C[idx] * D[idx_jp];
            e[idx] = e[idx] - C[idx] * e[idx_jp];
        }
    }

    for (i=0; i<n1; ++i) {
        idx = 0*n1 + i;
        idx1 = 1*n1 + i;
        idxN = (n2-1)*n1 + i;

        D[idx1] = (D[idx] - A[idx]*D[idxN] - C[idx]*D[idx1]) \
                / (D[idx] + A[idx]*e[idxN] + C[idx]*e[idx1]);

    }

    for (j=1; j<n2; ++j) {
        for (i=0; i<n1; ++i) {
            idx = j*n1 + i;
            idx1 = 1*n1 + i;

            D[idx] += D[idx1]*e[idx];
        }
    }
}

// void tdma_cycl_many(
//     std::vector<double> &a,
//     std::vector<double> &b,
//     std::vector<double> &c,
//     std::vector<double> &d,
//     int n1, int n2) {
//     // n1: n_sys
//     // n2: n_row
        
//     // std::vector<double> r(n2);
//     std::vector<double> e(n2*n1);
//     int idx;
//     int i, j;

//     for (j=0; j<n1; ++j) {
//         for (i=0; i<n2; ++i) {
//             idx = j*n2 +i;
//             e[idx] = 0;
//         }
//         idx = j*n2 + 1;
//         e[idx] = -a[idx];
        
//         idx = j*n2 + (n2-1);
//         e[idx] = -c[idx];
//     }

//     for (j=0; j<n1; ++j) {
//         idx = j*n2 + 1;
//         d[idx] /= b[idx];
//         e[idx] /= b[idx];
//         c[idx] /= b[idx];
//     }

//     double r;
//     for (j=0; j<n1; ++j) {
//         for (i=2; i<n2; ++i) {
//             idx = j*n2 + i;
//             r = 1.0 / (b[idx] - a[idx]*c[idx-1]);
//             d[idx] = r * (d[idx] - a[idx]*d[idx-1]);
//             e[idx] = r * (e[idx] - a[idx]*e[idx-1]);
//             c[idx] = r * c[idx];
//         }
//     }

//     for (j=0; j<n1; ++j) {
//         for (i=n2-2; i>=1; --i) {
//             idx = j*n2 + i;
//             d[idx] = d[idx] - c[idx] * d[idx + 1];
//             e[idx] = e[idx] - c[idx] * e[idx + 1];
//         }
//     }

//     for (j=0; j<n1; ++j) {
//         idx = j*n2 + 0;
//         d[idx] = (d[idx] - a[idx]*d[idx + (n2-1)] - c[idx]*d[idx + 1]) \
//                / (b[idx] + a[idx]*e[idx + (n2-1)] + c[idx]*e[idx + 1]);
//     }

//     double dd;
//     for (j=0; j<n1; ++j) {
//         dd = d[j*n2 + 0];
//         for (i=1; i<n2; ++i) {
//             idx = j*n2 + i;
//             d[idx] = d[idx] + dd*e[idx];
//         }
//     }
// }

