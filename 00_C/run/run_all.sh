#!/bin/bash
#SBATCH -J test_kdh_all      # Job 이름 (구분을 위해 scaling 추가)
#SBATCH -p batch                 # 파티션 이름
#SBATCH -w cpu01                 # 실행할 노드 이름
#SBATCH --nodes=1                # 항상 1개의 노드만 사용
#SBATCH --ntasks-per-node=64     # 루프에서 사용할 최대 프로세스 개수를 할당
#SBATCH -o results/%x_%j.out     # 표준 출력 파일
#SBATCH -e results/%x_%j.err     # 표준 에러 파일
#SBATCH --comment xxx

# --- 환경 설정 ---
echo "Loading modules..."
module purge 
module load nvhpc/23.7
echo "Modules loaded."
echo ""

# --- 실행할 프로세스 개수 목록 ---
# 이 배열의 값을 수정하여 원하는 테스트 케이스를 실행할 수 있습니다.
PROCESS_COUNTS=(1 2 4 8 16 32 64)

# --- 각 프로세스 개수에 대해 루프 실행 ---
for NP in "${PROCESS_COUNTS[@]}"
do
    echo "========================================="
    echo "RUNNING WITH $NP PROCESSES"
    echo "========================================="
    
    INPUT_FILE="./PARA_INPUT_${NP}.txt"
    
    # 실행 전, 해당 프로세스 개수에 맞는 입력 파일이 있는지 확인
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: Input file '$INPUT_FILE' not found. Skipping this run."
        echo ""
        continue # 파일이 없으면 이 단계는 건너뛰고 다음 루프로 이동
    fi

    # mpirun 명령어 실행
    # -np 플래그와 입력 파일 이름을 현재 루프의 $NP 값으로 설정
    mpirun -np "$NP" ./a.out "$INPUT_FILE"
    
    echo "Finished run with $NP processes."
    echo "" # 출력 파일의 가독성을 위해 빈 줄 추가

done

echo "All scaling tests are complete."