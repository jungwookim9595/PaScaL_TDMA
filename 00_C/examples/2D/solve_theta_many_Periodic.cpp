// solve_theta.cpp
#include "solve_theta.hpp"
#include "../examples_lab/save.hpp"
#include "iostream"
#include <cmath>
#include <chrono> 

void solve_theta::solve_theta_plan_single(std::vector<double>& theta) 
{   
    int myrank, ierr;
    int time_step;      // Current time step
    double t_curr;      // Current simulation time

    // 내가 사용할거
    int i, j;
    int idx;
    int idx_ip, idx_im;
    int idx_jp, idx_jm;
    double dxdx, dydy;
    double coef_x_a, coef_x_b, coef_x_c;
    double coef_y_a, coef_y_b, coef_y_c;
    
    // Loop and index variables
    
    int ny1 = sub.ny_sub+1; // number of cell with ghost cell in y axis (그냥 셀 개수)
    int nx1 = sub.nx_sub+1; // number of cell with ghost cell in x axis

    // 1) 계획(plan) 객체 선언
    PaScaL_TDMA::ptdma_plan_many px_many, py_many;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank==0) {
        std::cout << "Start to solve" << std::endl;
    }

    auto cx = topo.commX();
    int rankx = cx.myrank;
    PaScaL_TDMA tdma_x;
    tdma_x.PaScaL_TDMA_plan_many_create(px_many, sub.ny_sub-1, cx.myrank, cx.nprocs, cx.comm);
    std::vector<double> Axx((nx1-2)*(ny1-2)), Bxx((nx1-2)*(ny1-2)), Cxx((nx1-2)*(ny1-2)), Dxx((nx1-2)*(ny1-2));

    auto cy = topo.commY();
    int ranky = cy.myrank;
    PaScaL_TDMA tdma_y;
    tdma_y.PaScaL_TDMA_plan_many_create(py_many, sub.nx_sub-1, cy.myrank, cy.nprocs, cy.comm);
    std::vector<double> Ayy((ny1-2)*(nx1-2)), Byy((ny1-2)*(nx1-2)), Cyy((ny1-2)*(nx1-2)), Dyy((ny1-2)*(nx1-2));
    
    std::vector<double> rhs_x(nx1 * ny1, 0.0);
    std::vector<double> rhs_y(nx1 * ny1, 0.0);
    std::vector<double> theta_half(nx1 * ny1, 0.0);
    
    double dt = 0.01;
    int max_iter = 100;
    auto start = std::chrono::steady_clock::now();
    for (int t_step=0; t_step<max_iter; ++t_step) {

        // ---------------------- 차분 공식 -----------------------------
        // bdy 에서 거리 정보가 달라진다.
        // ( u(i+1) - 3u(i) + 2u(i-1) ) / (3/4)dxdx  left_bdy
        // ( u(i+1) - 2u(i) + u(i-1) ) / (3/4)dxdx   interior
        // ( 2u(i+1) - 3u(i) + u(i-1) ) / (3/4)dxdx  right_bdy
        // -----------------------------

        // Calculating r.h.s -----------------------------------------------------------------------------------------------------

        // (1+Ax/2)(1+Ay/2)u_n
        // y ---------
        for (j=1; j<ny1-1; ++j) {
            for (i=0; i<nx1; ++i) {
                idx = j * nx1 + i;
                idx_jm = (j-1) * nx1 + i;
                idx_jp = (j+1) * nx1 + i;
                dydy = (sub.dmy_sub[j]*sub.dmy_sub[j]);

                coef_y_a = (dt / 2.0 / dydy) * ( 1.0 );
                coef_y_b = (dt / 2.0 / dydy) * (-2.0 );
                coef_y_c = (dt / 2.0 / dydy) * ( 1.0 );

                rhs_y[idx] = (coef_y_c*theta[idx_jp] + (1.0+coef_y_b)*theta[idx] + coef_y_a*theta[idx_jm]);
            }
        }

        sub.ghostcellUpdate(rhs_y, cx, cy, params);

        // save_rhs_to_csv(rhs_y, nx1, ny1, "results", "rhs_y_" + std::to_string(cy.myrank) + std::to_string(cx.myrank) + std::to_string(t_step) +".csv", 15);

        // x ---------
        for (j=1; j<ny1-1; ++j) {
            for (i=1; i<nx1-1; ++i) {
                idx = j * nx1 + i;
                idx_im = j * nx1 + (i-1);
                idx_ip = j * nx1 + (i+1);
                dxdx = (sub.dmx_sub[i]*sub.dmx_sub[i]);
                
                coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
                coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
                coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );

                rhs_x[idx] = (coef_x_c*rhs_y[idx_ip] + (1.0+coef_x_b)*rhs_y[idx] + coef_x_a*rhs_y[idx_im]);
                
                // source func
                rhs_x[idx] += (dt) * (2.0 * Pi*Pi * cos(Pi*sub.x_sub[i]) * cos(Pi*sub.y_sub[j]));
            }
        }

        // save_rhs_to_csv(rhs_x, nx1, ny1, "results", "rhs_x_" + std::to_string(cy.myrank) + std::to_string(cx.myrank) + std::to_string(t_step) +".csv", 15);

        // cal -----------------  A system 만들기 ------------------------------------------------

        // bdy
        for (j=1; j<ny1-1; ++j) {
            coef_y_a = (dt / 2.0 / dydy) * ( 1.0 );
            coef_y_b = (dt / 2.0 / dydy) * (-2.0 );
            coef_y_c = (dt / 2.0 / dydy) * ( 1.0 );
            coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + 5.0/3.0 );
            coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + 5.0/3.0 );

            i = 0;
            idx = j * nx1 + i;
            idx_jm = (j-1) * nx1 + i;
            idx_jp = (j+1) * nx1 + i;
            rhs_x[j * nx1 + (1)] += coef_x_a * (-coef_y_a*theta[idx_jm] + (1.0-coef_y_b)*theta[idx] - coef_y_c*theta[idx_jp]) * sub.theta_x_left_index[1];

            i = (nx1-1);
            idx = j * nx1 + i;
            idx_jm = (j-1) * nx1 + i;
            idx_jp = (j+1) * nx1 + i;
            rhs_x[j * nx1 + (nx1-2)] += coef_x_c * (-coef_y_a*theta[idx_jm] + (1.0-coef_y_b)*theta[idx] - coef_y_c*theta[idx_jp]) * sub.theta_x_right_index[nx1-2];
        }

        // x -------------------------------------------
        for (j=1; j<ny1-1; ++j) {
            for (i=1; i<nx1-1; ++i) {
                idx = j * nx1 + i;
                dxdx = (sub.dmx_sub[i]*sub.dmx_sub[i]);
                
                coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
                coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
                coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );

                Axx[(j-1)*(nx1-2) + (i-1)] = -coef_x_a;
                Bxx[(j-1)*(nx1-2) + (i-1)] = 1.0-coef_x_b;
                Cxx[(j-1)*(nx1-2) + (i-1)] = -coef_x_c;
                Dxx[(j-1)*(nx1-2) + (i-1)] = rhs_x[idx];
            }
        }

        tdma_x.PaScaL_TDMA_many_solve(px_many, Axx, Bxx, Cxx, Dxx, ny1-2, nx1-2);
        for (j=1; j<ny1-1; ++j) {
            for (i=1; i<nx1-1; ++i) {
                idx = j * nx1 + i;
                theta_half[idx] = Dxx[(j-1)*(nx1-2) + (i-1)]; 
            }
        }

        // y -------------------------------------------
        for (i=1; i<nx1-1; ++i) {
            for (j=1; j<ny1-1; ++j) {
                idx = j * nx1 + i;
                dydy = (sub.dmy_sub[j]*sub.dmy_sub[j]);
                
                coef_y_a = (dt / 2.0 / dydy) * ( 1.0 );
                coef_y_b = (dt / 2.0 / dydy) * (-2.0 );
                coef_y_c = (dt / 2.0 / dydy) * ( 1.0 );

                Ayy[(i-1)*(ny1-2) + (j-1)] = -coef_y_a;
                Byy[(i-1)*(ny1-2) + (j-1)] = 1.0-coef_y_b;
                Cyy[(i-1)*(ny1-2) + (j-1)] = -coef_y_c;
                Dyy[(i-1)*(ny1-2) + (j-1)] = theta_half[idx];
            }        
        }
        tdma_y.PaScaL_TDMA_many_solve_cycle(py_many, Ayy, Byy, Cyy, Dyy, nx1-2, ny1-2);
        for (i=1; i<nx1-1; ++i) {
            for (j=1; j<ny1-1; ++j) {
                idx = j * nx1 + i;
                theta[idx] = Dyy[(i-1)*(ny1-2) + (j-1)];
            }
        }

        // Update ghostcells from the solutions.
        sub.ghostcellUpdate(theta, cx, cy, params);

        // save_rhs_to_csv(theta, nx1, ny1, "results", "theta_" + std::to_string(cy.myrank) + std::to_string(cx.myrank) + std::to_string(t_step) +".csv", 15);

    }   // Time step end------------------------
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (myrank==0) {
        std::cout << "elapsed time: " << elapsed_ms << " ms\n";
    }

    tdma_x.PaScaL_TDMA_plan_many_destroy(px_many, px_many.nprocs);
    tdma_y.PaScaL_TDMA_plan_many_destroy(py_many, py_many.nprocs);

    // save results
    save_rhs_to_csv(theta, nx1, ny1, "results", "theta_" + std::to_string(cy.myrank) + std::to_string(cx.myrank) +".csv", 15);
}