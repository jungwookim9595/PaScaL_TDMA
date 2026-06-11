# nvhpc
# CXX       := mpicxx
# CXXFLAGS  := -O1 -std=c++17 -g -fPIC
# CXXFLAGS  += -tp=icelake
# CXXFLAGS  += -Minfo=all 
# run: make lib 2> ./minfo.log

# g++
# CXXFLAGS  := -O3 -std=c++17 -g -fPIC
# CXXFLAGS  += -fno-fast-math -ffp-contract=off
# CXXFLAGS  +=-fopenmp # openmp

# intel
# CXX       := mpiicpx
# CXXFLAGS  := -O3 -std=c++17 -g -fPIC
# CXXFLAGS  += -tp=icelake


# ---------- vector report -----------
CXX       := mpicxx
CXXFLAGS  := -O1 -std=c++17 -g -fPIC
CXXFLAGS  += -tp=icelake
# CXXFLAGS  += -Minfo=vec -Mcache_align -mp