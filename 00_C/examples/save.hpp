// save.hpp
#ifndef SAVE_HPP
#define SAVE_HPP

#include <filesystem>   // C++17
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>      // std::setprecision, std::fixed

inline void save_rhs_to_csv(const std::vector<double>& rhs,
                     int nx, int ny,
                     const std::string& folder,
                     const std::string& filename,
                     int precision = 6) // 기본 소수점 자리수 6
{
    // 1) 폴더가 없다면 생성
    std::filesystem::path dir(folder);
    if (!std::filesystem::exists(dir)) {
        if (!std::filesystem::create_directories(dir)) {
            std::cerr << "Error: 디렉토리 생성 실패: " << folder << std::endl;
            return;
        }
    }

    // 2) 파일 경로 조합
    std::filesystem::path filepath = dir / filename;

    // 3) 파일 열기
    std::ofstream ofs(filepath.string());
    if (!ofs.is_open()) {
        std::cerr << "Error: 파일을 열 수 없습니다: " << filepath << std::endl;
        return;
    }

    // 4) 소수점 자리수 설정
    ofs << std::fixed << std::setprecision(precision);

    // 5) CSV로 쓰기
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            ofs << rhs[j*nx + i];
            if (i < nx - 1) ofs << ',';
        }
        ofs << '\n';
    }

    ofs.close();
    // std::cout << "Saved rhs to " << filepath << std::endl;
}

// 3D 저장 (z 슬라이스별로 여러 CSV 파일 생성)
inline void save_3d_to_csv(const std::vector<double>& arr3d,
                                  int nx, int ny, int nz,
                                  const std::string& folder,
                                  const std::string& base_filename,
                                  int precision = 6)
{   
    std::filesystem::path dir(folder);
    if (!std::filesystem::exists(dir)) {
        if (!std::filesystem::create_directories(dir)) {
            std::cerr << "Error: 디렉토리 생성 실패: " << folder << std::endl;
            return;
        }
    }

    for (int k = 0; k < nz; ++k) {
        std::string filename = base_filename + "_k" + std::to_string(k) + ".csv";
        std::filesystem::path filepath = dir / filename;
        std::ofstream ofs(filepath.string());
        if (!ofs.is_open()) {
            std::cerr << "Error: 파일을 열 수 없습니다: " << filepath << std::endl;
            continue;
        }

        ofs << std::fixed << std::setprecision(precision);

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = k * ny * nx + j * nx + i;
                ofs << arr3d[idx];
                if (i < nx - 1) ofs << ',';
            }
            ofs << '\n';
        }

        ofs.close();
    }

    std::cout << "save file" << std::endl;
}

#endif // SAVE_HPP
