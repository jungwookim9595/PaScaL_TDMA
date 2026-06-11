#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip> // std::setprecision, std::fixed
#include <mpi.h>

void save_timing_data(const std::string& filename, MPI_Comm comm,
                      const std::vector<std::string>& event_names,
                      const std::vector<double>& local_times)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    // 랭크 0에서 모든 랭크의 데이터를 받을 버퍼를 준비합니다.
    std::vector<double> all_times;
    if (myrank == 0) {
        all_times.resize(nprocs * local_times.size());
    }

    // 각 랭크의 local_times 데이터를 랭크 0의 all_times 버퍼로 모읍니다.
    MPI_Gather(local_times.data(), local_times.size(), MPI_DOUBLE,
               all_times.data(), local_times.size(), MPI_DOUBLE,
               0, comm);

    // 랭크 0만 파일에 데이터를 기록합니다.
    if (myrank == 0) {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        // 소수점 9자리까지 고정 형식으로 출력 설정
        outfile << std::fixed << std::setprecision(9);

        // 각 랭크의 데이터를 순서대로 작성합니다.
        for (int rank = 0; rank < nprocs; ++rank) {
            for (size_t event_idx = 0; event_idx < event_names.size(); ++event_idx) {
                // 수집된 데이터(all_times)에서 현재 랭크와 이벤트에 해당하는 값의 위치를 계산
                int global_idx = rank * event_names.size() + event_idx;
                
                // "[이벤트이름]: 값" 형식으로 파일에 씁니다.
                outfile << "[" << event_names[event_idx] << "]: " << all_times[global_idx] << "\n";
            }
        }

        outfile.close();
        std::cout << "Timing data successfully saved to " << filename << std::endl;
    }
}

#endif // DEBUG_HPP