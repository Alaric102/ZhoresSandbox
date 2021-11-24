#pragma once
#include <math.h>	
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <vector>
#include <complex>

using namespace std;

namespace advection
{
    class advection_2d
    {
    protected:
        int solver_type;
        int size;
        int rank;
        double * face_normals;
        double * tr_areas;
        double * xbarys;
        double * ybarys;
        int * neighbor_ids;
        double limiter(double const& r_factor, double const& weight);
        void solver_small_cpu(double *&u, double const &dt, double const *vel, double t);

    public:
        bool * is_boundary;
        void calc_dynamic_arrays();
        advection_2d(int solver_type, int size, int rank) : solver_type(solver_type), size(size), rank(rank)
        {
            calc_dynamic_arrays();
        }
        ~advection_2d();
        virtual void solver_small(double *&u, double const dt, double const *vel, double t);
    };
}
