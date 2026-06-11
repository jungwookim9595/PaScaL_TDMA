// global_params.hpp
#ifndef GLOBAL_PARAMS_HPP
#define GLOBAL_PARAMS_HPP

#include <array>
#include <string>
#include <mpi.h>

class GlobalParams {
public:
    GlobalParams() = default;

    // input file로부터 파라미터 읽기
    void load(const std::string& filename);

    // 물리 및 수치 파라미터
    double Tmax, dt;
    int Nt;
    int nx, ny, nz;
    int nxm, nym, nzm;
    int nxp, nyp, nzp;

    double lx, ly, lz;
    double x0, xN, y0, yN, z0, zN;
    double dx, dy, dz;

    // MPI 프로세스 분할
    std::array<int, 3> np_dim;
};

#endif // GLOBAL_PARAMS_HPP
