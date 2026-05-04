// mpi_subdomain.cpp
#include "mpi_subdomain.hpp"
#include "../src/para_range.hpp"
#include <cmath>
#include "index.hpp"
#include "iostream"
#include <algorithm>

const double Pi = 3.14159265358979323846;

void MPISubdomain::make(const GlobalParams& params,
                        int npx, int rankx,
                        int npy, int ranky,
                        int npz, int rankz) {

    // (--0--)|--ista--|--2--|--3--|--iend--|(--nx_sub--) 이게 각 block이 가지는 도메인이라고 가정
    para_range(1, params.nx-1, npx, rankx, ista, iend);
    nx_sub = iend - ista + 2;
    para_range(1, params.ny-1, npy, ranky, jsta, jend);
    ny_sub = jend - jsta + 2;
    para_range(1, params.nz-1, npz, rankz, ksta, kend);
    nz_sub = kend - ksta + 2;

    // ghost cell 포함한 cell을 기준으로 하는 저장공간
    x_sub.resize(nx_sub+1);
    dmx_sub.resize(nx_sub+1);
    y_sub.resize(ny_sub+1);
    dmy_sub.resize(ny_sub+1);
    z_sub.resize(nz_sub+1);
    dmz_sub.resize(nz_sub+1);

    // 이놈은 x_left는 x 축의 왼쪽, 즉 y축 크기의 bdy정보를 저장
    theta_x_left_sub.assign( (ny_sub+1)*(nz_sub+1), 0.0);
    theta_x_right_sub.assign((ny_sub+1)*(nz_sub+1), 0.0);
    theta_y_left_sub.assign( (nz_sub+1)*(nx_sub+1), 0.0);
    theta_y_right_sub.assign((nz_sub+1)*(nx_sub+1), 0.0);
    theta_z_left_sub.assign( (nx_sub+1)*(ny_sub+1), 0.0);
    theta_z_right_sub.assign((nx_sub+1)*(ny_sub+1), 0.0);

    // 이놈은 각 축에서 어느 위치가 bdy인지 나타내는 index
    theta_x_left_index.assign(nx_sub+1, 0);
    theta_x_right_index.assign(nx_sub+1, 0);
    theta_y_left_index.assign(ny_sub+1, 0);
    theta_y_right_index.assign(ny_sub+1, 0);
    theta_z_left_index.assign(nz_sub+1, 0);
    theta_z_right_index.assign(nz_sub+1, 0);
}

void MPISubdomain::clean() {
    auto freeType = [](MPI_Datatype& dt) {
        if (dt != MPI_DATATYPE_NULL) MPI_Type_free(&dt);
    };
    freeType(ddtype_sendto_x_right);
    freeType(ddtype_recvfrom_x_left);
    freeType(ddtype_sendto_x_left);
    freeType(ddtype_recvfrom_x_right);
    freeType(ddtype_sendto_y_right);
    freeType(ddtype_recvfrom_y_left);
    freeType(ddtype_sendto_y_left);
    freeType(ddtype_recvfrom_y_right);
    freeType(ddtype_sendto_z_right);
    freeType(ddtype_recvfrom_z_left);
    freeType(ddtype_sendto_z_left);
    freeType(ddtype_recvfrom_z_right);
}

void MPISubdomain::makeGhostcellDDType() {
    int sizes[3] = {nz_sub+1, ny_sub+1, nx_sub+1};
    int subs[3];
    int starts[3];

    // --------- X direction (i) ---------
    subs[0] = nz_sub+1;
    subs[1] = ny_sub+1;
    subs[2] = 1;

    // send right face  (i = nx_sub-1)
    starts[0] = 0;
    starts[1] = 0;
    starts[2] = nx_sub-1;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_x_right);
    MPI_Type_commit(&ddtype_sendto_x_right);

    // recv left ghost (i = 0)
    starts[0] = 0;
    starts[1] = 0;
    starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_x_left);
    MPI_Type_commit(&ddtype_recvfrom_x_left);

    // send left face   (i = 1)
    starts[0] = 0;
    starts[1] = 0;
    starts[2] = 1;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_x_left);
    MPI_Type_commit(&ddtype_sendto_x_left);

    // recv right ghost(i = nx_sub)
    starts[0] = 0;
    starts[1] = 0;
    starts[2] = nx_sub;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_x_right);
    MPI_Type_commit(&ddtype_recvfrom_x_right);

    // --------- Y direction (j) ---------
    subs[0] = nz_sub+1;
    subs[1] = 1;
    subs[2] = nx_sub+1;

    // send top face (j = ny_sub-1)
    starts[0] = 0;
    starts[1] = ny_sub-1;
    starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_y_right);
    MPI_Type_commit(&ddtype_sendto_y_right);

    // recv bottom ghost (j = 0)
    starts[0] = 0;
    starts[1] = 0;
    starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_y_left);
    MPI_Type_commit(&ddtype_recvfrom_y_left);

    // send bottom face (j = 1)
    starts[0] = 0;
    starts[1] = 1;
    starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_y_left);
    MPI_Type_commit(&ddtype_sendto_y_left);

    // recv top ghost (j = ny_sub)
    starts[0] = 0;
    starts[1] = ny_sub;
    starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_y_right);
    MPI_Type_commit(&ddtype_recvfrom_y_right);

    // --------- Z direction (k) ---------
    subs[0] = 1;
    subs[1] = ny_sub+1;
    subs[2] = nx_sub+1;

    // send top face (k = nz_sub-1)
    starts[0] = nz_sub-1;
    starts[1] = 0;
    starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_z_right);
    MPI_Type_commit(&ddtype_sendto_z_right);

    // recv bottom ghost (k = 0)
    starts[0] = 0;
    starts[1] = 0;
    starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_z_left);
    MPI_Type_commit(&ddtype_recvfrom_z_left);

    // send bottom face (k = 1)
    starts[0] = 1;
    starts[1] = 0;
    starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_z_left);
    MPI_Type_commit(&ddtype_sendto_z_left);

    // recv top ghost (k = nz_sub)
    starts[0] = nz_sub;
    starts[1] = 0;
    starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_z_right);
    MPI_Type_commit(&ddtype_recvfrom_z_right);
}

void MPISubdomain::ghostcellUpdate(std::vector<double>& theta,
                                   const CartComm1D& cx,
                                   const CartComm1D& cy,
                                   const CartComm1D& cz,
                                   const GlobalParams& /*params*/) {
    MPI_Request reqs[12];
    int r=0;

    // X
    MPI_Isend(theta.data(), 1, ddtype_sendto_x_right,   cx.east_rank, 111, cx.comm, &reqs[r++]);
    MPI_Irecv(theta.data(), 1, ddtype_recvfrom_x_left,  cx.west_rank, 111, cx.comm, &reqs[r++]);
    MPI_Isend(theta.data(), 1, ddtype_sendto_x_left,    cx.west_rank, 222, cx.comm, &reqs[r++]);
    MPI_Irecv(theta.data(), 1, ddtype_recvfrom_x_right, cx.east_rank, 222, cx.comm, &reqs[r++]);

    // Y
    MPI_Isend(theta.data(), 1, ddtype_sendto_y_right,   cy.east_rank, 333, cy.comm, &reqs[r++]);
    MPI_Irecv(theta.data(), 1, ddtype_recvfrom_y_left,  cy.west_rank, 333, cy.comm, &reqs[r++]);
    MPI_Isend(theta.data(), 1, ddtype_sendto_y_left,    cy.west_rank, 444, cy.comm, &reqs[r++]);
    MPI_Irecv(theta.data(), 1, ddtype_recvfrom_y_right, cy.east_rank, 444, cy.comm, &reqs[r++]);

    // Z
    MPI_Isend(theta.data(), 1, ddtype_sendto_z_right,   cz.east_rank, 555, cz.comm, &reqs[r++]);
    MPI_Irecv(theta.data(), 1, ddtype_recvfrom_z_left,  cz.west_rank, 555, cz.comm, &reqs[r++]);
    MPI_Isend(theta.data(), 1, ddtype_sendto_z_left,    cz.west_rank, 666, cz.comm, &reqs[r++]);
    MPI_Irecv(theta.data(), 1, ddtype_recvfrom_z_right, cz.east_rank, 666, cz.comm, &reqs[r++]);

    MPI_Waitall(r, reqs, MPI_STATUSES_IGNORE);
}

void MPISubdomain::indices(const GlobalParams& /*params*/, 
                        int rankx, int npx,
                        int ranky, int npy,
                        int rankz, int npz) {
    std::fill(theta_x_left_index.begin(), theta_x_left_index.end(), 0);
    std::fill(theta_x_right_index.begin(), theta_x_right_index.end(), 0);
    if (rankx==0)       theta_x_left_index[1]       = 1;
    if (rankx==npx-1)   theta_x_right_index[nx_sub-1] = 1;

    std::fill(theta_y_left_index.begin(), theta_y_left_index.end(), 0);
    std::fill(theta_y_right_index.begin(), theta_y_right_index.end(), 0);
    if (ranky==0)       theta_y_left_index[1]       = 1;
    if (ranky==npy-1)   theta_y_right_index[ny_sub-1] = 1;

    std::fill(theta_z_left_index.begin(), theta_z_left_index.end(), 0);
    std::fill(theta_z_right_index.begin(), theta_z_right_index.end(), 0);
    if (rankz==0)       theta_z_left_index[1]       = 1;
    if (rankz==npz-1)   theta_z_right_index[nz_sub-1] = 1;
}

void MPISubdomain::mesh(const GlobalParams& params,
                        int rankx, int ranky, int rankz,
                        int npx, int npy, int npz) {

    double dx = params.lx / (params.nx - 1);
    for(int i=0; i<=nx_sub;++i) {
        
        if (rankx==0 && i==0) {
            x_sub[i] = params.x0;
            dmx_sub[i] = dx;
        }
        else if (rankx==npx-1 && i==nx_sub) {
            x_sub[i] = params.xN;
            dmx_sub[i] = dx;
        }
        else {
            x_sub[i] = params.x0 + dx/2 + (ista - 2 + i)*dx;
            dmx_sub[i] = dx;
        }
    }

    double dy = params.ly / (params.ny-1);
    for(int j=0; j<=ny_sub;++j) {

        if (ranky==0 && j==0) {
            y_sub[j] = params.y0;
            dmy_sub[j] = dy;
        }
        else if (ranky==npy-1 && j==ny_sub) {
            y_sub[j] = params.yN;
            dmy_sub[j] = dy;
        }
        else {
            y_sub[j] = params.y0 + dy/2 + (jsta - 2 + j)*dy;
            dmy_sub[j] = dy;
        }
    }

    double dz = params.lz / (params.nz-1);
    for(int k=0; k<=nz_sub; ++k) {

        if (rankz==0 && k==0) {
            z_sub[k] = params.z0;
            dmz_sub[k] = dz;
        }
        else if (rankz==npz-1 && k==nz_sub) {
            z_sub[k] = params.zN;
            dmz_sub[k] = dz;
        }
        else {
            z_sub[k] = params.z0 + dz/2 + (ksta - 2 + k)*dz;
            dmz_sub[k] = dz;
        }
    }
}

void MPISubdomain::initialization(std::vector<double>& theta) {
    
    int idx;
    int nx1 = nx_sub + 1;
    int ny1 = ny_sub + 1;
    int nz1 = nz_sub + 1;
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    for (int k=0; k<nz1; ++k){ 
        for(int j=0; j<ny1; ++j) {
            for(int i=0; i<nx1; ++i) {
                idx = idx_ijk(i, j, k, nx1, ny1);
                // theta[idx] = myrank; // debug
                theta[idx] = sin(Pi*x_sub[i]) * sin(Pi*y_sub[j]) * sin(Pi*z_sub[k]) * exp(-3.0 * Pi*Pi * 0.0) + cos(Pi*x_sub[i]) * cos(Pi*y_sub[j]) * cos(Pi*z_sub[k]);
            }
        }
    }
}

void MPISubdomain::boundary(std::vector<double>& theta) {

    int nx1 = nx_sub + 1;
    int ny1 = ny_sub + 1;
    int nz1 = nz_sub + 1;
    int ij, jk, ik;         // fixed
    // int ij, jk, ki;    // flexible
    int ijk;

    // X
    for (int k=0; k<nz1; ++k) {
        for (int j=0; j<ny1; ++j) {
            jk = idx_jk(j, k, ny1);

            ijk = idx_ijk(0, j, k, nx1, ny1);
            theta_x_left_sub[jk] = theta[ijk];

            ijk = idx_ijk(nx1-1, j, k, nx1, ny1);
            theta_x_right_sub[jk] = theta[ijk];
        }
    }

    // Y (fixed)
    for (int k=0; k<nz1; ++k) {
        for (int i=0; i<nx1; ++i) {
            ik = idx_ik(i, k, nx1);

            ijk = idx_ijk(i, 0, k, nx1, ny1);
            theta_y_left_sub[ik] = theta[ijk];

            ijk = idx_ijk(i, ny1-1, k, nx1, ny1);
            theta_y_right_sub[ik] = theta[ijk];
        }
    }

    // Z
    for (int j=0; j<ny1; ++j) {
        for (int i=0; i<nx1; ++i) {
            ij = idx_ij(i, j, nx1);

            ijk = idx_ijk(i, j, 0, nx1, ny1);
            theta_z_left_sub[ij] = theta[ijk];

            ijk = idx_ijk(i, j, nz1-1, nx1, ny1);
            theta_z_right_sub[ij] = theta[ijk];
        }
    }

}

// MPISubdomain sub;