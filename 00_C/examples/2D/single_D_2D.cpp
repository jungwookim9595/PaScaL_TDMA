#include "save.hpp"
#include "iostream"
#include <vector>
#include <cmath>
#include <chrono> 

void tdma_single(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& d, int n1) {
    int i;
    double r;

    d[0] = d[0]/b[0];
    c[0] = c[0]/b[0];

    for (i=1; i<=n1-1; ++i) {
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

    std::vector<double> e(n1, 0.0);
    e[1] = -a[1];
    e[n1-1] = -c[n1-1];

    d[1] = d[1]/b[1];
    e[1] = e[1]/b[1];
    c[1] = c[1]/b[1];

    for (i=2; i<=n1-1; ++i) {
        rr = 1.0 / (b[i] - a[i]*c[i-1]);
        d[i] = rr*(d[i] - a[i]*d[i-1]);
        e[i] = rr*(e[i] - a[i]*e[i-1]);
        c[i] = rr*c[i];
    }

    for (i=n1-2; i>=1; --i) {
        d[i] = d[i] - c[i]*d[i+1];
        e[i] = e[i] - c[i]*e[i+1];
    }

    d[0] = (d[0] - a[0]*d[n1-1] - c[0]*d[1])/(b[0] + a[0]*e[n1-1] + c[0]*e[1]);

    for (i=1; i<=n1-1; ++i) {
        d[i] = d[i] + d[0]*e[i];
    }
}

// compile = g++ -O2 -std=c++17 -static-libgcc -static-libstdc++ -o single_D.exe single_D.cpp
// Run = ./single_D.exe 
int main() {

    int i, j;

    int Nx = 128; 
    int Ny = 128;
    int nx1 = Nx+2;
    int ny1 = Ny+2;

    double x0 = -1;
    double xN = 1;
    double y0 = -1;
    double yN = 1;

    double dy = (yN-y0)/Ny;
    double dx = (xN-x0)/Nx;

    std::vector<double> X(nx1);
    std::vector<double> Y(ny1);
    for (i=0; i<nx1; ++i) {
        if (i==0) {
            X[i] = x0;
            Y[i] = y0;
        }
        else if (i==nx1-1) {
            X[i] = xN;
            Y[i] = yN;
        }
        else {
            X[i] = x0 + dx/2 + (i-1)*dx;
            Y[i] = y0 + dy/2 + (i-1)*dy;
        }
    }

    std::vector<double> theta_x_left_index(nx1, 0.0);
    std::vector<double> theta_x_right_index(nx1, 0.0);
    std::vector<double> theta_y_left_index(ny1, 0.0);
    std::vector<double> theta_y_right_index(ny1, 0.0);
    theta_x_left_index[1] = 1;
    theta_x_right_index[nx1-2] = 1;
    theta_y_left_index[1] = 1;
    theta_y_right_index[ny1-2] = 1;

    std::vector<double> Axx((nx1-2)*(ny1-2)), Bxx((nx1-2)*(ny1-2)), Cxx((nx1-2)*(ny1-2)), Dxx((nx1-2)*(ny1-2));
    std::vector<double> Ax(nx1-2), Bx(nx1-2), Cx(nx1-2), Dx(nx1-2);

    std::vector<double> Ayy((ny1-2)*(nx1-2)), Byy((ny1-2)*(nx1-2)), Cyy((ny1-2)*(nx1-2)), Dyy((ny1-2)*(nx1-2));
    std::vector<double> Ay(ny1-2), By(ny1-2), Cy(ny1-2), Dy(ny1-2);

    std::vector<double> rhs_x(nx1 * ny1, 0.0);
    std::vector<double> rhs_y(nx1 * ny1, 0.0);
    std::vector<double> theta_half(nx1 * ny1, 0.0);
    std::vector<double> theta(nx1 * ny1, 0.0);
    std::vector<double> theta_old(nx1 * ny1, 0.0);
    
    int idx;    // row major index
    int idxx;   // column major index
    int idx_ip, idx_im;
    int idx_jp, idx_jm;
    double dxdx, dydy;
    double coef_x_a, coef_x_b, coef_x_c;
    double coef_y_a, coef_y_b, coef_y_c;
    double Pi = 3.14159265358979323846;

    // theta Init
    for (j=0; j<ny1; ++j) {
        for (i=0; i<nx1; ++i) {
            idx = j * nx1 + i;

            theta[idx] = sin(Pi*X[i]) * sin(Pi*Y[j]) * exp(-2.0 * Pi*Pi * 0.0 ) + cos(Pi*X[i]) * cos(Pi*Y[j]);
        }
    }

    int max_iter = 100;
    double dt = 0.01;
    int time;
    auto start = std::chrono::steady_clock::now();
    for (time=0; time<max_iter; ++time) {

        // Calculating r.h.s -------------------------------------------------------------------
        for (j=1; j<ny1-1; ++j) {
            for (i=1; i<nx1-1; ++i) {
                idx = j * nx1 + i;
                idx_jm = (j-1) * nx1 + i;
                idx_jp = (j+1) * nx1 + i;
                dydy = (dy*dy);
                
                coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) * theta_y_left_index[j] + (1.0/3.0) * theta_y_right_index[j] );
                coef_y_b = (dt / 2.0 / dydy) * (-2.0 -     (2.0) * theta_y_left_index[j] -     (2.0) * theta_y_right_index[j] );
                coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (1.0/3.0) * theta_y_left_index[j] + (5.0/3.0) * theta_y_right_index[j] );

                rhs_y[idx] = (coef_y_c*theta[idx_jp] + (1.0+coef_y_b)*theta[idx] + coef_y_a*theta[idx_jm]);
            }
        }

        // save_rhs_to_csv(rhs_y, nx1, ny1, "results", "rhs_y_single" + std::to_string(time) + ".csv", 15);

        // !!!(row -> column)!!!
        for (j=1; j<ny1-1; ++j) {
            for (i=1; i<nx1-1; ++i) {
                idxx = (i) * ny1 + (j); 
                idx = j * nx1 + i;
                idx_im = j * nx1 + (i-1);
                idx_ip = j * nx1 + (i+1);
                dxdx = (dx*dx);
                
                coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * theta_x_left_index[i] + (1.0/3.0) * theta_x_right_index[i] );
                coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * theta_x_left_index[i] -     (2.0) * theta_x_right_index[i] );
                coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * theta_x_left_index[i] + (5.0/3.0) * theta_x_right_index[i] );

                rhs_x[idxx] = (coef_x_c*rhs_y[idx_ip] + (1.0+coef_x_b)*rhs_y[idx] + coef_x_a*rhs_y[idx_im]);
                
                // source func
                rhs_x[idxx] += (dt) * ( 2.0 * Pi*Pi * cos(Pi*X[i]) * cos(Pi*Y[j]) );

            }
        }

        // save_rhs_to_csv(rhs_x, nx1, ny1, "results", "rhs_x_single"+ std::to_string(time) +".csv", 15);

        // Calculating A matrix ----------------------------------------------------------------

        // bdy
        for (i=1; i<nx1-1; ++i) {

            coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * theta_x_left_index[i] + (1.0/3.0) * theta_x_right_index[i] );
            coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * theta_x_left_index[i] -     (2.0) * theta_x_right_index[i] );
            coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * theta_x_left_index[i] + (5.0/3.0) * theta_x_right_index[i] );
            coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) );
            coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) );  

            j = 0;
            idxx = i * ny1 + j;
            idx_im = (i-1) * ny1 + j;
            idx_ip = (i+1) * ny1 + j;
            idx = i * ny1 + 1;
            rhs_x[idx] += coef_y_a * (-coef_x_a*theta[idx_im] + (1.0-coef_x_b)*theta[idxx] - coef_x_c*theta[idx_ip]);

            j = (ny1-1);
            idxx = i * ny1 + j;
            idx_im = (i-1) * ny1 + j;
            idx_ip = (i+1) * ny1 + j;
            idx = i * ny1 + (ny1-2);
            rhs_x[idx] += coef_y_c * (-coef_x_a*theta[idx_im] + (1.0-coef_x_b)*theta[idxx] - coef_x_c*theta[idx_ip]);
        }

        // y solve !!!(column -> row)!!!
        for (i=1; i<nx1-1; ++i) {
            for (j=1; j<ny1-1; ++j) {
                idxx = (i) * ny1 + (j);
                dydy = dy*dy;

                coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) * theta_y_left_index[j] + (1.0/3.0) * theta_y_right_index[j] );
                coef_y_b = (dt / 2.0 / dydy) * (-2.0 -     (2.0) * theta_y_left_index[j] -     (2.0) * theta_y_right_index[j] );
                coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (1.0/3.0) * theta_y_left_index[j] + (5.0/3.0) * theta_y_right_index[j] );

                Ay[j-1] = -coef_y_a;
                By[j-1] = (1.0-coef_y_b);
                Cy[j-1] = -coef_y_c;
                Dy[j-1] = rhs_x[idxx];
            }
            tdma_single(Ay, By, Cy, Dy, ny1-2);
            // Return the solution to the r.h.s. line-by-line.
            for (j=1; j<ny1-1; ++j) {
                idx = j * nx1 + i;
                theta_half[idx] = Dy[j-1];
            } 
        }

        // save_rhs_to_csv(theta_half, nx1, ny1, "results", "theta_half_single"+ std::to_string(time) +".csv", 15);

        // bdy 
        for (j=1; j<ny1-1; ++j) {
            
            coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + 5.0/3.0 );
            coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + 5.0/3.0 );

            i = 1;
            idx = (j) * nx1 + (i);
            idx_im = (j) * nx1 + (i-1);
            theta_half[idx] += coef_x_a * theta[idx_im];

            i = nx1-2;
            idx = (j) * nx1 + (i);
            idx_ip = (j) * nx1 + (i+1);
            theta_half[idx] += coef_x_c * theta[idx_ip];
        }

        // x solve 
        for (j=1; j<ny1-1; ++j) {
            for (i=1; i<nx1-1; ++i) {
                idx = j * nx1 + i;
                dxdx = (dx*dx);

                coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * theta_x_left_index[i] + (1.0/3.0) * theta_x_right_index[i] );
                coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * theta_x_left_index[i] -     (2.0) * theta_x_right_index[i] );
                coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * theta_x_left_index[i] + (5.0/3.0) * theta_x_right_index[i] );

                Ax[i-1] = -coef_x_a;
                Bx[i-1] = (1.0-coef_x_b);
                Cx[i-1] = -coef_x_c;
                Dx[i-1] = theta_half[idx];
            }
            tdma_single(Ax, Bx, Cx, Dx, nx1-2);
            // Return the solution to the r.h.s. line-by-line.
            for (i=1; i<nx1-1; ++i) {
                idx = j * nx1 + i;
                theta[idx] = Dx[i-1];
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "elapsed time: " << elapsed_ms << " ms\n";

    // save results
    std::cout << "tN = " << dt * max_iter << std::endl;
    // save_rhs_to_csv(theta, nx1, ny1, "results", "theta_single.csv", 17);

    return 0;
}