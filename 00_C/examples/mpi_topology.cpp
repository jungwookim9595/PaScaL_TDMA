// mpi_topology.cpp
#include "mpi_topology.hpp"

// 생성자
MPITopology::MPITopology(const std::array<int, 3>& dims,
                         const std::array<bool, 3>& periods)
    : dims_(dims), periods_(periods), world_cart_(MPI_COMM_NULL) {}

// 2D Cartesian + 1D sub communicator 생성
void MPITopology::make() {
    int per[3] = {int(periods_[0]), int(periods_[1]), int(periods_[2])};
    if (MPI_Cart_create(MPI_COMM_WORLD, 3, dims_.data(), per, 0, &world_cart_) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Cart_create failed");
    defineSubcomm(0, comm_x_);
    defineSubcomm(1, comm_y_);
    defineSubcomm(2, comm_z_);
}

// 해제
void MPITopology::clean() {
    if (world_cart_!=MPI_COMM_NULL) MPI_Comm_free(&world_cart_);
}

// 접근자
MPI_Comm          MPITopology::worldCart() const { return world_cart_; }
const CartComm1D& MPITopology::commX()     const { return comm_x_; }
const CartComm1D& MPITopology::commY()     const { return comm_y_; }
const CartComm1D& MPITopology::commZ()     const { return comm_z_; }

// 1D 서브커뮤 헬퍼
void MPITopology::defineSubcomm(int dim, CartComm1D& sub) {
    int remain[3]={0, 0, 0}; remain[dim]=1;
    MPI_Cart_sub(world_cart_, remain, &sub.comm);
    MPI_Comm_rank(sub.comm, &sub.myrank);
    MPI_Comm_size(sub.comm, &sub.nprocs);
    MPI_Cart_shift(sub.comm, 0, 1, &sub.west_rank, &sub.east_rank);
}

// MPITopology topo; 