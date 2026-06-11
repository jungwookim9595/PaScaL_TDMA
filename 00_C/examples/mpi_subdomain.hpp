// mpi_subdomain.hpp
#ifndef MPI_SUBDOMAIN_HPP
#define MPI_SUBDOMAIN_HPP

#include <mpi.h>
#include <vector>
#include <array>
#include "global.hpp"
#include "mpi_topology.hpp"

extern const double Pi;

class MPISubdomain {
public:
    // 기본 생성자: MPI_Datatype을 MPI_DATATYPE_NULL로 초기화
    MPISubdomain()
      : ddtype_sendto_x_right(MPI_DATATYPE_NULL), ddtype_recvfrom_x_left(MPI_DATATYPE_NULL),
        ddtype_sendto_x_left(MPI_DATATYPE_NULL), ddtype_recvfrom_x_right(MPI_DATATYPE_NULL),
        ddtype_sendto_y_right(MPI_DATATYPE_NULL), ddtype_recvfrom_y_left(MPI_DATATYPE_NULL),
        ddtype_sendto_y_left(MPI_DATATYPE_NULL), ddtype_recvfrom_y_right(MPI_DATATYPE_NULL),
        ddtype_sendto_z_right(MPI_DATATYPE_NULL), ddtype_recvfrom_z_left(MPI_DATATYPE_NULL),
        ddtype_sendto_z_left(MPI_DATATYPE_NULL), ddtype_recvfrom_z_right(MPI_DATATYPE_NULL) {}

    // Subdomain setup: pass simulation params and ranks
    void make(const GlobalParams& params,
              int npx, int rankx,
              int npy, int ranky,
              int npz, int rankz);
    void clean();

    // Ghost-cell MPI derived types
    void makeGhostcellDDType();

    // Exchange ghost cells for a flat theta array
    void ghostcellUpdate(std::vector<double>& theta,
                         const CartComm1D& cx,
                         const CartComm1D& cy,
                         const CartComm1D& cz,
                         const GlobalParams& params);

    // Boundary indices in y-direction
    void indices(const GlobalParams& params,
                 int rankx, int npx,
                 int ranky, int npy,
                 int rankz, int npz);

    // Mesh coordinates and spacings
    void mesh(const GlobalParams& params,
              int rankx, int ranky, int rankz,
              int npx, int npy, int npz);

    // Initialize field in subdomain
    void initialization(std::vector<double>& theta);

    // Boundary extraction
    void boundary(std::vector<double>& theta);

    // Subdomain sizes and offsets
    int nx_sub, ny_sub, nz_sub;
    int ista, iend;
    int jsta, jend;
    int ksta, kend;

    // Coordinates and spacings
    std::vector<double> x_sub, y_sub, z_sub;
    std::vector<double> dmx_sub, dmy_sub, dmz_sub;

    // Ghost-cell boundary buffers (flattened)
    std::vector<double> theta_x_left_sub, theta_x_right_sub;
    std::vector<double> theta_y_left_sub, theta_y_right_sub;
    std::vector<double> theta_z_left_sub, theta_z_right_sub;

    std::vector<int> theta_x_left_index, theta_x_right_index;
    std::vector<int> theta_y_left_index, theta_y_right_index;
    std::vector<int> theta_z_left_index, theta_z_right_index;

    // MPI derived datatypes
    MPI_Datatype ddtype_sendto_x_right, ddtype_recvfrom_x_left, ddtype_sendto_x_left, ddtype_recvfrom_x_right;
    MPI_Datatype ddtype_sendto_y_right, ddtype_recvfrom_y_left, ddtype_sendto_y_left, ddtype_recvfrom_y_right;
    MPI_Datatype ddtype_sendto_z_right, ddtype_recvfrom_z_left, ddtype_sendto_z_left, ddtype_recvfrom_z_right;
};

// extern MPISubdomain sub;

#endif // MPI_SUBDOMAIN_HPP