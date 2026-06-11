#ifndef INDEX_UTIL_HPP
#define INDEX_UTIL_HPP

// ------------------ 3D ------------------

// (i, j, k) → ijk layout
inline int idx_ijk(int i, int j, int k, int nx, int ny) {
    return k * ny * nx + j * nx + i;
}

// (j, k, i) → jki layout
inline int idx_jki(int j, int k, int i, int ny, int nz) {
    return i * ny * nz + k * ny + j;
}

// (k, i, j) → kij layout
inline int idx_kij(int k, int i, int j, int nz, int nx) {
    return j * nz * nx + i * nz + k;
}

// ------------------ 2D ------------------

// (i, j) → ij layout
inline int idx_ij(int i, int j, int nx) {
    return j * nx + i;
}

// (i, k) → ki layout
inline int idx_ik(int i, int k, int nx) {
    return k * nx + i;
}

// (j, i) → jk layout
inline int idx_ji(int j, int i, int ny) {
    return i * ny + j;
}

// (j, k) → jk layout
inline int idx_jk(int j, int k, int ny) {
    return k * ny + j;
}

// (k, i) → ki layout
inline int idx_ki(int k, int i, int nz) {
    return i * nz + k;
}



#endif // INDEX_UTIL_HPP
