#include <algorithm>
#include <sstream>
#include <memory>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <iomanip>
#include <complex>
#include <cassert>
#include <utility>
#include "advection/transport.h"
#include <mpi.h>
#include <omp.h>
#include <ctime>
#include <cstdlib>

using namespace std;
using namespace advection;


int main(int argc, char ** argv)
{
	MPI_Init(&argc, &argv);
	int size_of_group;
    	int rank;
	MPI_Comm comm = MPI_COMM_WORLD;
    	MPI_Comm_size(comm, &size_of_group);
    	MPI_Comm_rank(comm, &rank);
	if (rank==0) cout << "mpi 2d version" << endl;
	double h, dr, dt;
	int TIME, TOTAL_FRAMES;
	double epsilon_1 = 2.25;
	double epsilon_2 = 10.0;
	double mu_1 = 1.0;
	double mu_2 = 1.0;

	int N_THREADS_PER_PROC;
	
	if(argc !=2) {cout << "need filename" << endl; return 2;}

	string filename{argv[1]};
	ifstream arguments;
	arguments.open(filename);
	vector<string> data;
	string line;
	int i = 0;
	while(getline(arguments, line))
	{
		data.push_back(line);
		i++;
	}
	if (data.size() != 4) {cout << "wrong data" << endl; return 2;}
	dt = stod(data[0]);
	TIME = stoi(data[1]);
	TOTAL_FRAMES = stoi(data[2]);
	N_THREADS_PER_PROC = stoi(data[3]);

	bool if_paint = (TOTAL_FRAMES == 0) ? 0 : 1;
	int periods = (TOTAL_FRAMES == 0) ? TIME : TIME/TOTAL_FRAMES;

	
	omp_set_num_threads(N_THREADS_PER_PROC);

	//DATA FOR PYTHON VISUALIZER BEGIN
	ofstream colours;
	std::stringstream ss;
	ss << "colours_" << rank << ".txt";
	if (if_paint) colours.open(ss.str(), std::ios::out/*std::ios_base::app*/);
	
	stringstream ss_input;
	ss_input << "sizes_" << rank << ".txt";
	std::ifstream reader;
	reader.open(ss_input.str(), std::ios::in);
	getline(reader, line);
	int number_of_faces = std::stoi(line);
	getline(reader, line);
	int local_size = std::stoi(line);
	reader.close();

	advection_2d_mpi * equation;
	equation = new advection_2d_mpi(0, rank, size_of_group, local_size);	

	double *E_y;	
	double *E_x;
	double *H_z;
	double * coefs_E_y;
	double * coefs_E_x;
	double * coefs_H_z;
	double *epsilon;
	double *mu;
	
	E_y = new double [local_size];
	E_x = new double [local_size];
	H_z = new double [local_size];
	coefs_E_y = new double [2*local_size];
	coefs_E_x = new double [2*local_size];
	coefs_H_z = new double [2*local_size];
	epsilon = new double [local_size];
	mu = new double [local_size];
	
	ss_input.str("");
	ss_input.clear();
	ss_input << "centroids_" << rank << ".txt";
	reader.open(ss_input.str(), std::ios::in);
	for (int p = 0; p < local_size; p++)
	{
		getline(reader, line);
		double x = std::stod(line);
		getline(reader, line);
		double y = std::stod(line);
		E_x[p] = 0.0;
		E_y[p] = 0.0;
		H_z[p] = 0.0;
		epsilon[p] = pow(x - 1.0,2)+pow(y - 0.5,2) < 0.01 ? epsilon_2 : epsilon_1;
		mu[p] = pow(x - 1.0,2)+pow(y - 0.5,2) < 0.01 ? mu_2 : mu_1;
	}	
	reader.close();
	double start;
	MPI_Barrier(comm);
	if(rank==0) cout << "starting parallel stuff..." << endl;
	if(rank == 0) start = MPI_Wtime();
	for (int t = 0; t < TIME; t++)
	{
		#pragma omp parallel for
		for (int p = 0; p < local_size; p++)
		{
			coefs_H_z[2*p] = 1.0/mu[p]*(E_y[p]);
			coefs_H_z[2*p+1] = -1.0/mu[p]*E_x[p];
				
			coefs_E_x[2*p] = 0.0;
			coefs_E_x[2*p+1] = -1.0/epsilon[p]*(H_z[p]);

			coefs_E_y[2*p] = 1.0/epsilon[p]*(H_z[p]);
			coefs_E_y[2*p+1] = 0.0;
		}
		MPI_Barrier(comm);
		equation->solver_small(H_z, dt, coefs_H_z, true, true, t*dt);
		equation->solver_small(E_x, dt, coefs_E_x, true, false, t*dt);
		equation->solver_small(E_y, dt, coefs_E_y, false, false, t*dt);
		if (t%periods == 0 && if_paint) 
		{
			double * p_n = &H_z[0];
			double maximum = 0.0;
			double minimum = 0.0;
			double average = 0.0;
			for (int k = 0; k < local_size; k++)
			{
				if (p_n[k] > maximum) maximum = p_n[k];
				if (p_n[k] < minimum) minimum = p_n[k];
				average+=p_n[k];
			}
			average = average/local_size;
			double * p_new = new double [local_size];
			for (int k = 0; k < local_size; k++)
			{
				p_new[k] = (p_n[k] - -1.0)/(1.0 - -1.0);
			}
			for (int k = 0; k < local_size; k++)
			{
				for (int i = 0; i < 1; i++)
				{
					int index = k+local_size*i;
					if ((p_new[index]) <= 0.25) colours << 0.0 << " " << ((p_new[index])/0.25)  << " " << 1.0 << " " << 1.0 << " "; 
					else if ((p_new[index]) <= 0.5) colours << 0.0 << " " << 1.0 << " " << ((0.5-(p_new[index]))/0.25) << " " << 1.0 << " ";
					else if ((p_new[index]) <= 0.75) colours << (((p_new[index])-0.5)/0.25) << " " << 1.0 << " " << 0.0 << " " << 1.0 << " ";
					else colours << 1.0 << " " << ((1.0-p_new[index])/0.25) << " " << 0.0 << " " << 1.0 << " ";	
				}
				colours << endl;
			}
			delete [] p_new;
		}
	}
	MPI_Barrier(comm);	
	double end;
	if (rank == 0)
	{
		end = MPI_Wtime();
		cout << "duration " << end-start << endl;
	}
	if (if_paint) colours.close();
	MPI_Finalize();
	return 0;
}

