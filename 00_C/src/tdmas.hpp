// tdmas.hpp
#ifndef tdmas_HPP
#define tdmas_HPP

#include <vector>


void tdma_single(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& d, int n1);

void tdma_cycl_single(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& d, int n1);

void tdma_many(
    double* __restrict A,
    double* __restrict B,
    double* __restrict C,
    double* __restrict D,
    int n1,int n2);

void tdma_many_debug(
    double* __restrict A,
    double* __restrict B,
    double* __restrict C,
    double* __restrict D,
    int n1, int n2, std::vector<double>& time_list);

void tdma_cycl_many(
    std::vector<double> &a,
    std::vector<double> &b,
    std::vector<double> &c,
    std::vector<double> &d,
    int n1,int n2);

#endif // tdmas.hpp