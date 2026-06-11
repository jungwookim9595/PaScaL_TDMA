include ../Makefile.inc

# --- 기본 설정 ---
# 최종 생성될 실행 파일 이름
EXE = a.out

# 소스, 인클루드, 라이브러리, 실행 파일 디렉터리 경로
SRCDIR 		:= .
IDIR 		:= ../include
LDIR 		:= ../lib
RUNDIR 		:= ../run

# --- 소스 및 오브젝트 파일 자동 감지 ---
# SRCDIR에 있는 모든 .cpp 파일을 찾아 SOURCES 변수에 저장합니다.
SOURCES := $(wildcard $(SRCDIR)/*.cpp)
# SOURCES 목록을 기반으로 .o 파일 목록을 자동으로 생성합니다.
OBJS 	:= $(SOURCES:.cpp=.o)

# --- 링커 설정 ---
# 링크할 라이브러리 경로 (-L) 와 라이브러리 이름 (-l) 을 지정합니다.
LDFLAGS := -L$(LDIR)
LIBS 	:= -lpascal_tdma

# --- 컴파일러 추가 설정 ---
# 헤더 파일 경로 (-I) 를 CXXFLAGS에 추가합니다.
CXXFLAGS += -I$(IDIR)


# --- 타겟 규칙 ---

# 기본 타겟: 'all'을 입력하면 실행 파일이 생성됩니다.
all: $(EXE)

# 실행 파일 생성 규칙
# OBJS에 명시된 모든 .o 파일이 준비되면 링크를 시작합니다.
$(EXE): $(OBJS)
	@echo "Linking to create executable: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)
	@echo "Moving executable to $(RUNDIR)/"
	mv $@ $(RUNDIR)/

# C++ 소스 파일(.cpp)을 오브젝트 파일(.o)로 컴파일하는 패턴 규칙
# -MMD 플래그는 헤더 파일 의존성을 자동으로 추적하는 .d 파일을 생성합니다.
%.o: %.cpp
	@echo "Compiling $< -> $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

# 빌드 산출물 정리
clean:
	@echo "Cleaning project..."
	rm -f $(OBJS) $(OBJS:.o=.d) $(RUNDIR)/$(EXE)

# 자동 생성된 의존성(.d) 파일들을 포함시킵니다.
# 헤더 파일이 변경되면, 해당 헤더를 포함하는 소스 파일만 자동으로 재컴파일됩니다.
-include $(OBJS:.o=.d)