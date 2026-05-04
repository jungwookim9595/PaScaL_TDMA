include ../Makefile.inc

# 최종 결과물: 정적 라이브러리 파일
LIB = libpascal_tdma.a

# 소스 파일과 오브젝트 파일이 위치할 디렉터리
SRCDIR = .
OBJDIR = ./obj

# C++ 소스 파일 목록을 자동으로 찾도록 설정
# .cpp 확장자를 가진 모든 파일을 찾습니다.
SOURCES = $(wildcard $(SRCDIR)/*.cpp)

# 오브젝트 파일 목록을 소스 파일 목록으로부터 생성
OBJS = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES))

# # CUDA 사용 여부에 따라 CUDA 관련 오브젝트 추가
# ifdef USE_CUDA
#     # .cu 확장자를 가진 CUDA 소스 파일을 찾습니다.
#     CU_SOURCES = $(wildcard $(SRCDIR)/*.cu)
#     # CUDA 오브젝트 파일 목록 생성
#     CU_OBJS = $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(CU_SOURCES))
#     # 전체 오브젝트 목록에 CUDA 오브젝트 추가
#     OBJS += $(CU_OBJS)
# endif

# 기본 빌드 타겟: 라이브러리와 헤더 파일을 모두 처리
all: $(LIB) inc

# 라이브러리 파일 생성 규칙
# OBJS 목록에 있는 모든 오브젝트 파일이 준비되면 라이브러리를 생성합니다.
$(LIB): $(OBJS)
	@echo "Creating static library: $@"
	ar qc $@ $^
	@echo "Copying library to ../lib/"
	cp $@ ../lib

# 헤더 파일 복사 규칙
# 프로젝트의 공용 헤더 파일들을 ../include 디렉터리로 복사합니다.
inc:
	@echo "Copying header files to ../include/"
	cp *.hpp ../include/

# C++ 소스 파일(.cpp)을 오브젝트 파일(.o)로 컴파일하는 패턴 규칙
# -MMD -MP 플래그는 헤더 파일 의존성을 자동으로 생성해줍니다.
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

# # CUDA 소스 파일(.cu)을 오브젝트 파일(.o)로 컴파일하는 패턴 규칙
# # NVCC는 NVIDIA CUDA 컴파일러입니다.
# $(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
# 	@echo "Compiling CUDA $< -> $@"
# 	$(NVCC) $(NVFLAGS) -c $< -o $@

# obj 디렉터리가 없으면 생성하는 규칙
$(OBJDIR):
	mkdir -p $(OBJDIR)

# 빌드 과정에서 생성된 파일들을 삭제
clean:
	@echo "Cleaning project..."
	rm -rf ../lib ../include $(LIB) $(OBJDIR)

# 자동 생성된 의존성 파일들을 포함
# .d 파일이 존재하면, make가 이를 읽어 헤더 파일이 변경될 때 자동으로 재컴파일합니다.
-include $(OBJS:.o=.d)