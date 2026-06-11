#include <mpi.h>
#include <iostream>
#include <vector>
#include "mpi_subdomain.hpp"
#include "global.hpp"
#include "../src/para_range.hpp"

void gather_theta(std::vector<double>& global_theta, std::vector<double>& theta, const GlobalParams& params, MPISubdomain& sub, int myrank) {

    if (myrank==0) {

        // 000 을 일단 global에 넣는다.
        for (int k=1; k<sub.nz_sub; ++k) {
            for (int j=1; j<sub.ny_sub; ++j) {
                for (int i=0; i<sub.nx_sub; ++i) {
                    
                    int global_k = (sub.ksta-1) + (k); // sub.ksta = 1 or 5
                    int global_j = (sub.jsta-1) + (j);
                    int global_i = (sub.ista-1) + (i);
                    int global_idx = global_k * (params.nx+1)*(params.ny+1) + global_j * (params.nx+1) + global_i;
                    int local_idx = k * (sub.nx_sub+1)*(sub.ny_sub+1) + j * (sub.nx_sub+1) + i;

                    global_theta[global_idx] = theta[local_idx];
                }
            }
        }

        for (int r=1; r<(params.np_dim[0] * params.np_dim[1] * params.np_dim[2]); ++r) {

            int Px = params.np_dim[0];
            int Py = params.np_dim[1];
            int Pz = params.np_dim[2];

            int rkz = r % Pz;
            int rky = (r / Pz) % Py;
            int rkx = r / (Pz * Py);

            int ista, iend;
            int jsta, jend;
            int ksta, kend;

            para_range(1, params.nx-1, Px, rkx, ista, iend);
            int nx_sub = iend - ista + 2;
            para_range(1, params.ny-1, Py, rky, jsta, jend);
            int ny_sub = jend - jsta + 2;
            para_range(1, params.nz-1, Pz, rkz, ksta, kend);
            int nz_sub = kend - ksta + 2;

            std::vector<int> globalsizes ={ (params.nz+1), (params.ny+1), (params.nx+1) };
            std::vector<int> subsizes = { nz_sub-1, ny_sub-1, nx_sub-1 };
            std::vector<int> starts = { ksta, jsta, ista };

            // std::cout << "[myrank] = " << r
            //           << "| rank_xyz = " << rkx << rky << rkz
            //           << "| globalsizes = " << (params.nx+1) << ", "<< (params.ny+1) << ", " << (params.nz+1)
            //           << "| subsizes = " << nx_sub+1 << ", " << ny_sub+1 << ", " << nz_sub+1
            //           << "| starts = " << ista-1 << ", " << jsta-1 << ", " << ksta-1 << std::endl;

            MPI_Datatype recvtype;
            MPI_Type_create_subarray(3, globalsizes.data(), subsizes.data(), starts.data(),
                                    MPI_ORDER_C, MPI_DOUBLE, &recvtype);
            MPI_Type_commit(&recvtype);

            MPI_Recv(global_theta.data(), 1, recvtype, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Type_free(&recvtype);
        }
    }
    else {

        std::vector<int> globalsizes ={ (sub.nz_sub+1), (sub.ny_sub+1), (sub.nx_sub+1) };
        std::vector<int> subsizes = { sub.nz_sub-1, sub.ny_sub-1, sub.nx_sub-1 };
        std::vector<int> starts = { 1, 1, 1 };
        
        // auto cx = topo.commX();
        // int rankx = cx.myrank;
        
        // auto cy = topo.commY();
        // int ranky = cy.myrank;

        // auto cz = topo.commZ();
        // int rankz = cz.myrank;

        // std::cout << "myrank = " << myrank
        //         << "| rank_xyz = " << rankx << ranky << rankz
        //         << "| globalsizes = " << (params.nx+1) << ", "<< (params.ny+1) << ", " << (params.nz+1)
        //         << "| subsizes = " << sub.nx_sub+1 << ", " << sub.ny_sub+1 << ", " << sub.nz_sub+1
        //         << "| starts = " << sub.ista-1 << ", " << sub.jsta-1 << ", " << sub.ksta-1 << std::endl;

        // std::cout << "myrank = " << myrank
        //         << "| rank_xyz = " << rankx << ranky << rankz
        //         << "| globalsizes = " << (sub.nx_sub+1) << ", "<< (sub.ny_sub+1) << ", " << (sub.nz_sub+1)
        //         << "| subsizes = " << (sub.nx_sub-1) << ", " << (sub.ny_sub-1) << ", " << (sub.nz_sub-1)
        //         << "| starts = " << 1 << ", " << 1 << ", " << 1 << std::endl;

        MPI_Datatype sendtype;
        MPI_Type_create_subarray(3, globalsizes.data(), subsizes.data(), starts.data(),
                                MPI_ORDER_C, MPI_DOUBLE, &sendtype);
        MPI_Type_commit(&sendtype);

        MPI_Send(theta.data(), 1, sendtype, 0, 0, MPI_COMM_WORLD);
    }
}