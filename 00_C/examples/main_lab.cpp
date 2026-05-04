#include <mpi.h>
#include "global.hpp"
#include "mpi_topology.hpp"
#include "mpi_subdomain.hpp"
#include "solve_theta.hpp"
#include "save.hpp"
#include "gather_theta.hpp"

#include <iostream>
#include <array>
#include <string>
#include <cmath>

int main(int argc, char** argv) {

  int nprocs, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  GlobalParams params;
  MPITopology topo;
  MPISubdomain sub;

  params.load(argv[1]); 

  // x, y, z
  topo.init(
    { params.np_dim[0], params.np_dim[1], params.np_dim[2] },
    { false, false, false }
  );

  topo.make();

  auto cx = topo.commX();
  auto cy = topo.commY();
  auto cz = topo.commZ();

  // 3) npx, rankx 등 변수 정의
  int npx = params.np_dim[0], rankx = cx.myrank;
  int npy = params.np_dim[1], ranky = cy.myrank;
  int npz = params.np_dim[2], rankz = cz.myrank;

  // 4) 서브도메인 객체 만들기
  sub.make(params, npx, rankx, npy, ranky, npz, rankz);

  // 5) ghostcell 통신용 MPI_Datatype 생성
  sub.makeGhostcellDDType();

  sub.mesh(params, rankx, ranky, rankz, npx, npy, npz);

  // 7) x, y-방향 경계 인덱스 설정
  sub.indices(params, rankx, npx, ranky, npy, rankz, npz);

  // 8) theta 필드 (subdomain 크기에 맞춘 flat 배열) 준비
  std::vector<double> theta((sub.nz_sub + 1) * (sub.ny_sub + 1) * (sub.nx_sub + 1), 0.0);

  // 9) 내부 값 초기화
  sub.initialization(theta);
  MPI_Barrier(MPI_COMM_WORLD);

  // 10) ghostcell 교환
  sub.ghostcellUpdate(theta, cx, cy, cz, params);

  // 11) bdy값 저장하기
  sub.boundary(theta);

  // 12) 솔버 호출해서 실행하기
  solve_theta solver(params, topo, sub);
  solver.solve_theta_plan_single(theta);

  // 13) call error
  double local_error=0;
  double exact_value;
  int ijk;

  for (int k=1; k<(sub.nz_sub + 1)-1; ++k) {
    for (int j=1; j<(sub.ny_sub + 1)-1; ++j) {
      for (int i=1; i<(sub.nx_sub + 1)-1; ++i) {
        ijk = k*(sub.nx_sub + 1)*(sub.ny_sub + 1) + j*(sub.nx_sub + 1) + i;

        exact_value = sin(Pi*sub.x_sub[i])*sin(Pi*sub.y_sub[j])*sin(Pi*sub.z_sub[k]) * exp(-3*Pi*Pi * params.Tmax) +
                      cos(Pi*sub.x_sub[i])*cos(Pi*sub.y_sub[j])*cos(Pi*sub.z_sub[k]);

        local_error += pow(theta[ijk] - exact_value, 2);
      }
    }
  }

  double global_error;
  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (myrank == 0) {
      std::cout << "Global L2 error = " << std::sqrt(global_error / params.nx / params.ny / params.nz) << "\n";
  }

  // // 13) solution 하나로 합치기
  // std::vector<double> global_theta;
  // if (myrank==0) {
  //   global_theta.assign((params.nx+1)*(params.ny+1)*(params.nz+1), -1.0);
  // }
  
  // gather_theta(global_theta, theta, params, sub, myrank);

  // if (myrank==0) {
    
  //   // std::cout << "\n[global result, z-slices]\n";
  //   // for (int k = 0; k < (params.nz + 1); ++k) {
  //   //     std::cout << "k = " << k << "\n";
  //   //     for (int j = 0; j < (params.ny + 1); ++j) {
  //   //         for (int i = 0; i < (params.nx + 1); ++i) {
  //   //             int idx = k * (params.ny + 1) * ((params.nx + 1)) + j * (params.nx + 1) + i;
  //   //             std::cout << global_theta[idx] << " ";
  //   //         }
  //   //         std::cout << "\n";
  //   //     }
  //   //     std::cout << "\n";
  //   // }

  //   save_3d_to_csv(global_theta, (params.nx+1), (params.ny+1), (params.nz+1), "results", "theta", 15);
  // }


  sub.clean();
  topo.clean();
  MPI_Finalize();
}
