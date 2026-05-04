#include "para_range.hpp"
#include <cmath>

void para_range(int start, int end, int nproc, int rank, int &ista, int &iend) {
    // 전체 도메인에서 ghost cell 포함한 index를 기준으로 
    // ghost cell을 뺀 인덱스의 범위 (start <= idx <= end)
    // ex: |--*--|--*--|--*--|--*--| = 실제 도메인
    // (--0--)|--start--|--2--|--3--|--end--|(--5--)  = ghost cell을 추가한 도메인
    int len = end - start + 1;
    int base = len / nproc;
    int rem  = len % nproc;
    ista = start + rank * base + std::min(rank, rem);
    iend = ista + base - 1 + (rank < rem ? 1 : 0);
}