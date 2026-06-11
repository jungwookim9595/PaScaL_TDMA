// solve_theta.cpp
#include "solve_theta.hpp"
// #include "../examples_lab/save.hpp"
#include "iostream"
#include "index.hpp"
#include "timer.hpp"
#include "Debug.hpp"

#include <string>
#include <cmath>
#include <chrono> 

// salloc -w cpu02 --nodes=1 --ntasks-per-node=64

// solve_theta.cpp
solve_theta::solve_theta(const GlobalParams& params,
                         const MPITopology& topo,
                         MPISubdomain& sub)
    : params(params), topo(topo), sub(sub) {}

void solve_theta::solve_theta_plan_single(std::vector<double>& theta) 
{   
    // int myrank, ierr;
    // int time_step;      // Current time step
    // double t_curr;      // Current simulation time

    // Loop and index variables (그냥 셀 개수)
    int nz1 = sub.nz_sub+1; // number of cell with ghost cell in z axis 
    int ny1 = sub.ny_sub+1; // number of cell with ghost cell in y axis 
    int nx1 = sub.nx_sub+1; // number of cell with ghost cell in x axis

    // 내가 사용할거
    int i, j, k;
    int idx;
    int ijk;
    int ijk_z, ikj_y, jki_x;
    int idx_ip, idx_im;
    int idx_jp, idx_jm;
    int idx_kp, idx_km;
    double dxdx, dydy, dzdz;

    double coef_x_a, coef_x_b, coef_x_c;
    double coef_y_a, coef_y_b, coef_y_c;
    double coef_z_a, coef_z_b, coef_z_c;

    // 1) 계획(plan) 객체 선언
    PaScaL_TDMA::ptdma_plan_many px_many, py_many, pz_many;

    auto cx = topo.commX();
    // int rankx = cx.myrank;
    PaScaL_TDMA tdma_x;
    tdma_x.PaScaL_TDMA_plan_many_create(px_many, (ny1-2)*(nz1-2), cx.myrank, cx.nprocs, cx.comm);
    std::vector<double> Axx((nx1-2)*(ny1-2)*(nz1-2)), Bxx((nx1-2)*(ny1-2)*(nz1-2)), Cxx((nx1-2)*(ny1-2)*(nz1-2)), Dxx((nx1-2)*(ny1-2)*(nz1-2));

    auto cy = topo.commY();
    // int ranky = cy.myrank;
    PaScaL_TDMA tdma_y;
    tdma_y.PaScaL_TDMA_plan_many_create(py_many, (nx1-2)*(nz1-2), cy.myrank, cy.nprocs, cy.comm);
    std::vector<double> Ayy((ny1-2)*(nx1-2)*(nz1-2)), Byy((ny1-2)*(nx1-2)*(nz1-2)), Cyy((ny1-2)*(nx1-2)*(nz1-2)), Dyy((ny1-2)*(nx1-2)*(nz1-2));

    auto cz = topo.commZ();
    // int rankz = cz.myrank;
    PaScaL_TDMA tdma_z;
    tdma_z.PaScaL_TDMA_plan_many_create(pz_many, (nx1-2)*(ny1-2), cz.myrank, cz.nprocs, cz.comm);
    std::vector<double> Azz((nz1-2)*(nx1-2)*(ny1-2)), Bzz((nz1-2)*(nx1-2)*(ny1-2)), Czz((nz1-2)*(nx1-2)*(ny1-2)), Dzz((nz1-2)*(nx1-2)*(ny1-2));
    
    std::vector<double> rhs(nx1 * ny1 * nz1, 0.0);
    // std::vector<double> theta_z(nx1 * ny1 * nz1, 0.0);
    // std::vector<double> theta_y(nx1 * ny1 * nz1, 0.0);
    
    double dt = params.dt;
    int max_iter = params.Nt;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank==0) {
        std::cout << "Start | iteration = " << max_iter << std::endl;
        std::cout << "nx = " << nx1-2 << " | ny = " << ny1-2 << "| nz = " << nz1-2 << std::endl;
    }
    // MPI_Barrier(MPI_COMM_WORLD); // ---------------------------------------------------------------------------------
    for (int t_step=0; t_step<max_iter; ++t_step) {

        double rhs_1 = 0.0;
        double rhs_2 = 0.0;
        double solve_z_1 = 0.0;
        double solve_z_2 = 0.0;
        double solve_y_1 = 0.0;
        double solve_y_2 = 0.0;
        double solve_x_1 = 0.0;
        double solve_x_2 = 0.0;
        double comm_1 = 0.0;
        double comm_2 = 0.0;
        
        // double solve_1 = MPI_Wtime();

        rhs_1 = MPI_Wtime();
        for (k=1; k<nz1-1; ++k) {

            dzdz = sub.dmz_sub[k]*sub.dmz_sub[k];
            coef_z_a = (dt / 2.0 / dzdz) * ( 1.0 + (5.0/3.0) * sub.theta_z_left_index[k] + (1.0/3.0) * sub.theta_z_right_index[k] );
            coef_z_b = (dt / 2.0 / dzdz) * (-2.0 -     (2.0) * sub.theta_z_left_index[k] -     (2.0) * sub.theta_z_right_index[k] );
            coef_z_c = (dt / 2.0 / dzdz) * ( 1.0 + (1.0/3.0) * sub.theta_z_left_index[k] + (5.0/3.0) * sub.theta_z_right_index[k] );
            
            for (j=1; j<ny1-1; ++j) {

                dydy = sub.dmy_sub[j]*sub.dmy_sub[j];
                coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) * sub.theta_y_left_index[j] + (1.0/3.0) * sub.theta_y_right_index[j] );
                coef_y_b = (dt / 2.0 / dydy) * (-2.0 -     (2.0) * sub.theta_y_left_index[j] -     (2.0) * sub.theta_y_right_index[j] );
                coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (1.0/3.0) * sub.theta_y_left_index[j] + (5.0/3.0) * sub.theta_y_right_index[j] );

                for (i=1; i<nx1-1; ++i) {
                    
                    dxdx = sub.dmx_sub[i]*sub.dmx_sub[i];
                    coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
                    coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
                    coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );

                    ijk    = idx_ijk(i  , j, k, nx1, ny1);
                    idx_ip = idx_ijk(i+1, j, k, nx1, ny1);
                    idx_im = idx_ijk(i-1, j, k, nx1, ny1);
                    idx_jp = idx_ijk(i, j+1, k, nx1, ny1); 
                    idx_jm = idx_ijk(i, j-1, k, nx1, ny1);
                    idx_kp = idx_ijk(i, j, k+1, nx1, ny1); 
                    idx_km = idx_ijk(i, j, k-1, nx1, ny1);

                    rhs[ijk] = (coef_x_c*theta[idx_ip] + coef_x_a*theta[idx_im])
                               + (coef_y_c*theta[idx_jp] + coef_y_a*theta[idx_jm])
                               + (coef_z_c*theta[idx_kp] + coef_z_a*theta[idx_km])
                               + (1.0 + coef_x_b + coef_y_b + coef_z_b)*theta[ijk]
                               + dt * ( 3.0 * Pi*Pi * cos(Pi*sub.x_sub[i]) * cos(Pi*sub.y_sub[j]) * cos(Pi*sub.z_sub[k]));
                }
            }
        }
        rhs_2 = MPI_Wtime();

        // Calculating A matrix ----------------------------------------------------------------

        // bdy(z)
        // bdy_z_1 = MPI_Wtime();
        for (j=1; j<ny1-1; ++j) {

            dydy = sub.dmy_sub[j]*sub.dmy_sub[j];
            coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) * sub.theta_y_left_index[j] + (1.0/3.0) * sub.theta_y_right_index[j] );
            coef_y_b = (dt / 2.0 / dydy) * (-2.0 -     (2.0) * sub.theta_y_left_index[j] -     (2.0) * sub.theta_y_right_index[j] );
            coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (1.0/3.0) * sub.theta_y_left_index[j] + (5.0/3.0) * sub.theta_y_right_index[j] ); 

            for (i=1; i<nx1-1; ++i) {

                dxdx = sub.dmx_sub[i]*sub.dmx_sub[i];
                coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
                coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
                coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );

                // k=0
                dzdz = sub.dmz_sub[0]*sub.dmz_sub[0];
                coef_z_a = (dt / 2.0 / dzdz) * ( 1.0 + (5.0/3.0) );

                idx    = idx_ij(i  , j-1, nx1);
                idx_ip = idx_ij(i+1, j-1, nx1);
                idx_im = idx_ij(i-1, j-1, nx1);
                rhs[idx_ijk(i, j, 1, nx1, ny1)] += (coef_z_a) * (-coef_y_a) *
                                                     (-coef_x_a*sub.theta_z_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_left_sub[idx] - coef_x_c*sub.theta_z_left_sub[idx_ip]) *
                                                     sub.theta_z_left_index[1];

                idx    = idx_ij(i  , j, nx1);
                idx_ip = idx_ij(i+1, j, nx1);
                idx_im = idx_ij(i-1, j, nx1);
                rhs[idx_ijk(i, j, 1, nx1, ny1)] += (coef_z_a) * (1.0-coef_y_b) * 
                                                     (-coef_x_a*sub.theta_z_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_left_sub[idx] - coef_x_c*sub.theta_z_left_sub[idx_ip]) *
                                                     sub.theta_z_left_index[1];

                idx    = idx_ij(i  , j+1, nx1);
                idx_ip = idx_ij(i+1, j+1, nx1);
                idx_im = idx_ij(i-1, j+1, nx1);
                rhs[idx_ijk(i, j, 1, nx1, ny1)] += (coef_z_a) * (-coef_y_c)  *
                                                     (-coef_x_a*sub.theta_z_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_left_sub[idx] - coef_x_c*sub.theta_z_left_sub[idx_ip]) * 
                                                     sub.theta_z_left_index[1];

                // k=nz1-1
                dzdz = sub.dmz_sub[nz1-1]*sub.dmz_sub[nz1-1];
                coef_z_c = (dt / 2.0 / dzdz) * ( 1.0 + (5.0/3.0) );

                idx    = idx_ij(i  , j-1, nx1);
                idx_ip = idx_ij(i+1, j-1, nx1);
                idx_im = idx_ij(i-1, j-1, nx1);
                rhs[idx_ijk(i, j, nz1-2, nx1, ny1)] += (coef_z_c) * (-coef_y_a) *
                                                         (-coef_x_a*sub.theta_z_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_right_sub[idx] - coef_x_c*sub.theta_z_right_sub[idx_ip]) *
                                                         sub.theta_z_right_index[nz1-2];

                idx    = idx_ij(i  , j, nx1);
                idx_ip = idx_ij(i+1, j, nx1);
                idx_im = idx_ij(i-1, j, nx1);
                rhs[idx_ijk(i, j, nz1-2, nx1, ny1)] += (coef_z_c) * (1.0-coef_y_b) *
                                                         (-coef_x_a*sub.theta_z_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_right_sub[idx] - coef_x_c*sub.theta_z_right_sub[idx_ip]) *
                                                         sub.theta_z_right_index[nz1-2];
                idx    = idx_ij(i  , j+1, nx1);
                idx_ip = idx_ij(i+1, j+1, nx1);
                idx_im = idx_ij(i-1, j+1, nx1);
                rhs[idx_ijk(i, j, nz1-2, nx1, ny1)] += (coef_z_c) * (-coef_y_c) *
                                                         (-coef_x_a*sub.theta_z_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_right_sub[idx] - coef_x_c*sub.theta_z_right_sub[idx_ip]) *
                                                         sub.theta_z_right_index[nz1-2];         
            }
        }
        // bdy_z_2 = MPI_Wtime();

        // z solve 
        MPI_Barrier(MPI_COMM_WORLD);
        solve_z_1 = MPI_Wtime();
        for (k=1; k<nz1-1; ++k) {

            dzdz = sub.dmz_sub[k]*sub.dmz_sub[k];
            coef_z_a = (dt / 2.0 / dzdz) * ( 1.0 + (5.0/3.0) * sub.theta_z_left_index[k] + (1.0/3.0) * sub.theta_z_right_index[k] );
            coef_z_b = (dt / 2.0 / dzdz) * (-2.0 -     (2.0) * sub.theta_z_left_index[k] -     (2.0) * sub.theta_z_right_index[k] );
            coef_z_c = (dt / 2.0 / dzdz) * ( 1.0 + (1.0/3.0) * sub.theta_z_left_index[k] + (5.0/3.0) * sub.theta_z_right_index[k] );

            for (j=1; j<ny1-1; ++j) {
                for (i=1; i<nx1-1; ++i) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    ijk_z = idx_ijk(i-1, j-1, k-1, nx1-2, ny1-2);
                    
                    Azz[ijk_z] = -coef_z_a;
                    Bzz[ijk_z] = (1.0-coef_z_b);
                    Czz[ijk_z] = -coef_z_c;
                    Dzz[ijk_z] = rhs[ijk];
                }
            }
        }
        tdma_z.PaScaL_TDMA_many_solve(pz_many, Azz.data(), Bzz.data(), Czz.data(), Dzz.data(), (nx1-2)*(ny1-2), (nz1-2));
        for (k=1; k<nz1-1; ++k) {
            for (j=1; j<ny1-1; ++j) {
                for (i=1; i<nx1-1; ++i) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    ijk_z = idx_ijk(i-1, j-1, k-1, nx1-2, ny1-2);

                    rhs[ijk] = Dzz[ijk_z];
                }
            }
        }
        solve_z_2 = MPI_Wtime();

        // bdy(y)
        // bdy_y_1 = MPI_Wtime();
        for (k=1; k<nz1-1; ++k) {
            for (i=1; i<nx1-1; ++i) {

                dxdx = sub.dmx_sub[i]*sub.dmx_sub[i];
                coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
                coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
                coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );

                // j=0
                dydy = sub.dmy_sub[0]*sub.dmy_sub[0];
                coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) );

                idx    = idx_ik(i  , k, nx1);
                idx_ip = idx_ik(i+1, k, nx1);
                idx_im = idx_ik(i-1, k, nx1);
                rhs[idx_ijk(i, 1, k, nx1, ny1)] += coef_y_a * sub.theta_y_left_index[1] * 
                                                       (-coef_x_a*sub.theta_y_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_y_left_sub[idx] - coef_x_c*sub.theta_y_left_sub[idx_ip]);

                // j=ny1-1
                dydy = sub.dmy_sub[ny1-1]*sub.dmy_sub[ny1-1];
                coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) );
                
                idx    = idx_ik(i  , k, nx1);
                idx_ip = idx_ik(i+1, k, nx1);
                idx_im = idx_ik(i-1, k, nx1);
                rhs[idx_ijk(i, ny1-2, k, nx1, ny1)] += coef_y_c * sub.theta_y_right_index[ny1-2] * 
                                                           (-coef_x_a*sub.theta_y_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_y_right_sub[idx] - coef_x_c*sub.theta_y_right_sub[idx_ip]);
            }
        }
        // bdy_y_2 = MPI_Wtime();

        // y solve
        MPI_Barrier(MPI_COMM_WORLD);
        solve_y_1 = MPI_Wtime();
        for (j=1; j<ny1-1; ++j) {

            dydy = sub.dmy_sub[j]*sub.dmy_sub[j];
            coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) * sub.theta_y_left_index[j] + (1.0/3.0) * sub.theta_y_right_index[j] );
            coef_y_b = (dt / 2.0 / dydy) * (-2.0 -     (2.0) * sub.theta_y_left_index[j] -     (2.0) * sub.theta_y_right_index[j] );
            coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (1.0/3.0) * sub.theta_y_left_index[j] + (5.0/3.0) * sub.theta_y_right_index[j] );

            for (k=1; k<nz1-1; ++k) {
                for (i=1; i<nx1-1; ++i) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    ikj_y  = idx_ijk(i-1, k-1, j-1, nx1-2, nz1-2);

                    Ayy[ikj_y] = -coef_y_a;
                    Byy[ikj_y] = (1.0-coef_y_b);
                    Cyy[ikj_y] = -coef_y_c;
                    Dyy[ikj_y] = rhs[ijk];
                }
            }
        }
        tdma_y.PaScaL_TDMA_many_solve(py_many, Ayy.data(), Byy.data(), Cyy.data(), Dyy.data(), (nx1-2)*(nz1-2), ny1-2);
        for (j=1; j<ny1-1; ++j) {
            for (k=1; k<nz1-1; ++k) {
                for (i=1; i<nx1-1; ++i) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    ikj_y  = idx_ijk(i-1, k-1, j-1, nx1-2, nz1-2);

                    rhs[ijk] = Dyy[ikj_y];
                }
            }
        }
        solve_y_2 = MPI_Wtime();

        // bdy(x)
        // bdy_x_1 = MPI_Wtime();
        for (k=1; k<nz1-1; ++k) {
            for (j=1; j<ny1-1; ++j) {
                
                // i=0
                dxdx = sub.dmx_sub[0]*sub.dmx_sub[0];
                coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) );

                idx = idx_jk(j, k, ny1);
                rhs[idx_ijk(1, j, k, nx1, ny1)] += coef_x_a * sub.theta_x_left_index[1] * 
                                                       sub.theta_x_left_sub[idx];

                // i=nx1-1
                dxdx = sub.dmx_sub[nx1-1]*sub.dmx_sub[nx1-1];
                coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) );

                idx = idx_jk(j, k, ny1);
                rhs[idx_ijk(nx1-2, j, k, nx1, ny1)] += coef_x_c * sub.theta_x_right_index[nx1-2] * 
                                                           sub.theta_x_right_sub[idx];
            }
        }
        // bdy_x_2 = MPI_Wtime();

        // x solve
        MPI_Barrier(MPI_COMM_WORLD);
        solve_x_1 = MPI_Wtime();
        for (i=1; i<nx1-1; ++i) {

            dxdx = (sub.dmx_sub[i]*sub.dmx_sub[i]);
            coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
            coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
            coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );

            for (k=1; k<nz1-1; ++k) {
                for (j=1; j<ny1-1; ++j) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    jki_x = idx_ijk(j-1, k-1, i-1, ny1-2, nz1-2);

                    Axx[jki_x] = -coef_x_a;
                    Bxx[jki_x] = (1.0-coef_x_b);
                    Cxx[jki_x] = -coef_x_c;
                    Dxx[jki_x] = rhs[ijk];
                }
            }
        }
        tdma_x.PaScaL_TDMA_many_solve(px_many, Axx.data(), Bxx.data(), Cxx.data(), Dxx.data(), (ny1-2)*(nz1-2), nx1-2);
        for (i=1; i<nx1-1; ++i) {
            for (k=1; k<nz1-1; ++k) {
                for (j=1; j<ny1-1; ++j) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    jki_x = idx_ijk(j-1, k-1, i-1, ny1-2, nz1-2);

                    theta[ijk] = Dxx[jki_x];
                }
            }
        }
        solve_x_2 = MPI_Wtime();

        // Update ghostcells from the solutions.
        MPI_Barrier(MPI_COMM_WORLD);
        comm_1 = MPI_Wtime();
        sub.ghostcellUpdate(theta, cx, cy, cz, params);
        comm_2 = MPI_Wtime();

        std::vector<std::string> event_names = {
            "rhs",
            "solve_z",
            "solve_y",
            "solve_x",
            "comm"
        };
        std::vector<double> local_times = {
            rhs_2 - rhs_1,
            solve_z_2 - solve_z_1,
            solve_y_2 - solve_y_1,
            solve_x_2 - solve_x_1,
            comm_2 - comm_1
        };
        int npxyz = params.np_dim[0] * params.np_dim[1] * params.np_dim[2];
        save_timing_data("results/t_" + std::to_string(npxyz)  + "_" + std::to_string(t_step) + ".txt", MPI_COMM_WORLD, event_names, local_times);
 
        
        // solve_2 = MPI_Wtime();

        // std::vector<std::string> event_names = {
        //     "solve",
        // };
        // std::vector<double> local_times = {
        //     solve_2 - solve_1
        // };
        // int npxyz = params.np_dim[0] * params.np_dim[1] * params.np_dim[2];
        // save_timing_data("results/t_" + std::to_string(npxyz)  + "_" + std::to_string(t_step) + ".txt", MPI_COMM_WORLD, event_names, local_times);

    }   // Time step end------------------------
    // solve_2 = MPI_Wtime();

    tdma_x.PaScaL_TDMA_plan_many_destroy(px_many, px_many.nprocs);
    tdma_y.PaScaL_TDMA_plan_many_destroy(py_many, py_many.nprocs);
    tdma_z.PaScaL_TDMA_plan_many_destroy(pz_many, pz_many.nprocs);
}