// mpi_topology.hpp
#ifndef MPI_TOPOLOGY_HPP
#define MPI_TOPOLOGY_HPP

#include <mpi.h>
#include <array>
#include <memory> // ?
#include <stdexcept> // ? 

struct CartComm1D {
    int      myrank, nprocs, west_rank, east_rank;
    MPI_Comm comm;
};

class MPITopology {
public:
    // ① 기본 생성자: 아무 일도 안 함
    MPITopology() = default;

    MPITopology(const std::array<int, 3>& dims,
                const std::array<bool, 3>& periods);

    // ③ 런타임 초기화를 위한 init()
    void init(const std::array<int, 3>& dims,
              const std::array<bool, 3>& periods) {
        dims_    = dims;
        periods_ = periods;
        world_cart_ = MPI_COMM_NULL;
        // comm_x_, comm_y_, comm_z_ 등은 make() 호출 시 정의됩니다
    }

    void make();
    void clean();
    MPI_Comm            worldCart() const;
    const CartComm1D&   commX()     const;
    const CartComm1D&   commY()     const;
    const CartComm1D&   commZ()     const;
private:
    void defineSubcomm(int dim, CartComm1D& sub);
    std::array<int, 3>   dims_;
    std::array<bool, 3>  periods_;
    MPI_Comm            world_cart_;
    CartComm1D          comm_x_, comm_y_, comm_z_;
};

// extern MPITopology topo;

#endif // MPI_TOPOLOGY_HPP
