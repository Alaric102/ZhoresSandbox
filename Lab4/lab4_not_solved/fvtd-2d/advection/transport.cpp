#include "transport.h"

namespace advection
{
	double sigma(double const &r, double const &dr, int const &PML, int const & size)
	{
		if (r>(dr*(size-PML))) return pow((r-(size-PML)*dr)/(PML*dr),2)*3.0*log(10.0)*13.0/(PML*dr);
		else if (r < dr*(PML)) return pow((PML*dr-r)/(PML*dr),2)*3.0*log(10.0)*13.0/(PML*dr);
		else return 0;
	}

	advection_2d::~advection_2d()
	{
		delete [] face_normals;
		delete [] neighbor_ids;
		delete [] tr_areas;
		delete [] xbarys;
		delete [] ybarys;
		delete [] is_boundary;
	}

	advection_2d_mpi::~advection_2d_mpi()
	{
                for (int i = 0; i < size_of_group; i++)
		{
			if (interfaces_size[i]!=0)
			{
				delete [] receive_data [i];
				delete [] send_data [i];
			}
		}

		delete [] send_data;
		delete [] receive_data;
		for (int i = 0; i < size_of_group; i++)
		{
			delete [] ids_to_send[i];
			delete [] received_ids[i];
		}
		delete [] received_ids;
		delete [] ids_to_send;
		delete [] interfaces_size;
		delete [] global_ids_reference;
	}

	void advection_2d::calc_dynamic_arrays()
	{
		std::stringstream ss;
		ss << "centroids_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		std::ifstream reader;
		reader.open(ss.str(), std::fstream::in);
		is_boundary = new bool [size];
		neighbor_ids = new int [3*size];
		face_normals = new double [3*2*size];
		tr_areas = new double [size];
		xbarys =  new double [size];
		ybarys = new double [size];
		std::string line;
		for (int counter = 0; counter < size; counter++)
		{
			getline(reader, line);
			xbarys[counter] = std::stod(line);
			getline(reader, line);
			ybarys[counter] = std::stod(line);
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "is_boundary_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		reader.open(ss.str(), std::fstream::in);
		for (int k = 0; k < size; k++)
		{
			getline(reader, line);
			is_boundary[k] = std::stoi(line);
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "neighbor_ids_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		reader.open(ss.str(), std::fstream::in);
		for (int k = 0; k < size; k++)
		{
			for (int i = 0; i < 3; i++)
			{
				getline(reader, line);
				int index = k*3+i;
				neighbor_ids[index] = stoi(line);
			}
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "face_normals_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		reader.open(ss.str());
		
		for (int k = 0; k < size; k++)
		{
			for (int i = 0; i < 3; i++)
			{
				getline(reader, line);
				face_normals[2*(k*3+i)] = std::stod(line);
				getline(reader, line);
				face_normals[2*(k*3+i)+1] = std::stod(line);
			}
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "tr_areas_" << rank << ".txt";
		//std::cout << ss.str() << std::endl;
		reader.open(ss.str());
		
		for (int k = 0; k < size; k++)
		{
			getline(reader, line);
			tr_areas[k] = std::stod(line);
		}
	}

	void advection_2d::solver_small(double *&u, double const dt, double *&velocities, bool if_y, bool if_h, double t)
	{
		double * interpolated_velocities;
		interpolated_velocities = new double [size * 3 * 2];
		solver_small_cpu(u, dt, velocities, interpolated_velocities, if_y, if_h, t);
		delete [] interpolated_velocities;
	}

	void advection_2d::solver_small_cpu(double *&u, double const &dt, double *&velocities, double *&interpolated_velocities, bool if_y, bool if_h, double t)
	{
		#pragma omp parallel for
		for (int j = 0; j < size; j++)
		{	
			for (int k = 0; k < 3; k++)
			{
				if (neighbor_ids[j*3+k] != -1)
				{
					if (if_h && if_y)
					{
						if (is_boundary[j]!=is_boundary[neighbor_ids[j*3+k]] && ybarys[j] < 0.9 && ybarys[j] > 0.1 && ybarys[neighbor_ids[j*3+k]] < 0.9 && ybarys[neighbor_ids[j*3+k]] > 0.1)
						{
							interpolated_velocities[2*(j*3+k)] = (is_boundary[j] ? 1.0 : -1.0)*cos(5.0*(xbarys[neighbor_ids[j*3+k]] - 3*t))+0.5*velocities[2*j]+velocities[2*neighbor_ids[j*3+k]]*0.5;
							interpolated_velocities[2*(j*3+k)+1] = velocities[2*j+1]*0.5+velocities[2*neighbor_ids[j*3+k]+1]*0.5;
						}
						else
						{
							interpolated_velocities[2*(j*3+k)] = velocities[2*neighbor_ids[j*3+k]]*0.5 + velocities[2*j]*0.5;
							interpolated_velocities[2*(j*3+k)+1] = velocities[2*neighbor_ids[j*3+k]+1]*0.5 + velocities[2*j+1]*0.5;
						}
					}
					else
					{
						interpolated_velocities[2*(j*3+k)] = velocities[2*neighbor_ids[j*3+k]]*0.5 + velocities[2*j]*0.5;
						interpolated_velocities[2*(j*3+k)+1] = velocities[2*neighbor_ids[j*3+k]+1]*0.5 + velocities[2*j+1]*0.5;
					}
				}
				else
				{
					interpolated_velocities[2*(j*3+k)] = velocities[2*j]+0.0;
					interpolated_velocities[2*(j*3+k)+1] = velocities[2*j+1]+0.0;
				}
			}
		
		}
		#pragma omp parallel for
		for (int j = 0; j < size; j++)
		{
			double temp = 0.0;
			for (int k = 0; k < 3; k++)
			{
				temp += interpolated_velocities[2*(3*j+k)] * face_normals[2*(j*3+k)] + interpolated_velocities[2*(3*j+k)+1] * face_normals[2*(j*3+k)+1]; 
			}
			if (!if_h) u[j] = u[j] - dt * (temp / tr_areas[j] + (if_y ? 0.5*pow(2.25, 0.5)*sigma(ybarys[j], 0.01, 10, 100)*u[j] : 0.5*pow(2.25, 0.5)*sigma(xbarys[j], 0.01, 20, 200)*u[j]) )/(1.0+0.5*dt*( if_y ? pow(2.25, 0.5)*sigma(ybarys[j], 0.01, 10, 100) : pow(2.25, 0.5)*sigma(xbarys[j], 0.01, 20, 200) ) );
			else u[j] = u[j] - dt * (temp / tr_areas[j] + 0.5*pow(1.0/2.25, 0.5)*sigma(ybarys[j], 0.01, 10, 100)*u[j] + 0.5*pow(1.0/2.25, 0.5)*sigma(xbarys[j], 0.01, 20, 200)*u[j])/(1.0+0.5*dt*( pow(1.0/2.25, 0.5)*sigma(ybarys[j], 0.01, 10, 100) + pow(1.0/2.25, 0.5)*sigma(xbarys[j], 0.01, 20, 200)));
		}
	}

	void advection_2d_mpi::solver_small(double *&u, double const &dt, double *&velocities, bool if_y, bool if_h, double t)
	{
		//pack u and velocities to `send_data` here
		double **send_data = new double* [size_of_group];
		double **received_data = new double* [size_of_group];
		
		for (int i = 0; i < size_of_group; ++i){
			send_data[i] 	 = new double [3*interfaces_size[i]];
			received_data[i] = new double [3*interfaces_size[i]];
			if (i != rank){
				// std::cout << rank << " with " << i << " number of shared triangles: " << interfaces_size[i] << std::endl;
				for (int j = 0; j < interfaces_size[i]; ++j){
					int id = ids_to_send[i][j];

					send_data[i][3*j]   = u[id];
					send_data[i][3*j+1] = velocities[id];
					send_data[i][3*j+2] = velocities[id + 1];

					// std::cout << j << " triangle ID: " << id << std::endl;
					// std::cout << id << " triangle\n"
					// 		  << u[id] << ", "
					// 		  << velocities[id] << ", "
					// 		  << velocities[id + 1] << ", " << std::endl;
				}
			} 
		}
		
		std::cout << rank << " Finised packing" << std::endl;
		MPI_Barrier(MPI_COMM_WORLD);
		
		MPI_Request request[size_of_group];
		for (int i = 0; i < size_of_group; i++)
		{
			if (i!=rank){
				std::cout << rank << " ---> " << i << ". " << interfaces_size[i] << std::endl;
				MPI_Isend(send_data[i], interfaces_size[i], MPI_INT, i, rank, MPI_COMM_WORLD, &request[i]);
			}
		}
		for (int i = 0; i < size_of_group; i++) {
			if (i != rank) {
				MPI_Recv(received_data[i], interfaces_size[i], MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
				for (int j = 0; j < size; j++)
				{
					for (int k = 0; k < interfaces_size[i]; k++)
					{
						if(global_ids_reference[j] == buffer[k]) received_ids[i][k] = j;
					}	
				}
				for (int k = 0; k < interfaces_size[i]; k++)
				{
					for (int j = 0; j < size; j++)
					{
					
						if(global_ids_reference[j] == ids_to_send[i][k]) ids_to_send[i][k] = j;
					}	
				}
				delete [] buffer;
			}
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << rank << " finised" << std::endl;

		//send_data[0] - 1d array to send to process 0
		//send_data[1] - 1d array to send to process 1
		//send_data[2] - 1d array to send to process 2
		//send_data[3] - 1d array to send to process 3
		//class has variable called `size_of_group` - total amount of processes
		//class has array interfaces_size, for example interfaces_size[2] is number of shared triangles of current process with process 2
		
		//####################################################################################################
		
		//send `send_data` to everyone who needs it and receive it to `receive_data[0]`, `receive_data[1]` and so on
		//this can be achieved with MPI_Isend, MPI_Irecv, MPI_Waitall, MPI_Request and MPI_STATUS_IGNORE
		
		//####################################################################################################
		
		//unpack u and velocities from `receive_data` here
		//received_data[0] - 1d array from process 0
		//received_data[1] - 1d array from process 1
		//received_data[2] - 1d array from process 2
		//received_data[3] - 1d array from process 3
		//class has received_ids two-dimensional array to map received data to proper local ids of triangles
		//received_ids[1][3]: 4th pack of u and velocities received from process 1 should go to triangle with id that's stored at received_ids[1][3]

		//####################################################################################################

		advection_2d::solver_small(u, dt, velocities, if_y, if_h, t);
		MPI_Barrier(MPI_COMM_WORLD);
		
	}

	void advection_2d_mpi::calc_mpi_arrays()
	{
		ids_to_send = new int * [size_of_group];
		for (int i = 0; i < size_of_group; i++)
		{
			ids_to_send[i] = new int [size];
		}
		interfaces_size = new int [size_of_group];

		global_ids_reference = new int [size];

		std::stringstream ss;
		ss << "global_ids_reference_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		std::ifstream reader;
		reader.open(ss.str(), std::fstream::in);
		std::string line;
		for (int j = 0; j < size; j++)
		{
			getline(reader, line);
			global_ids_reference[j] = std::stoi(line);
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "ids_send_size_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		reader.open(ss.str(), std::fstream::in);
		for (int j = 0; j < size_of_group; j++)
		{
			getline(reader, line);
			interfaces_size[j] = std::stoi(line);
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "ids_to_send_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		reader.open(ss.str(), std::fstream::in);
		for (int j = 0; j < size_of_group; j++)
			for (int i = 0; i < interfaces_size[j]; i++)
			{
				reader >> ids_to_send[j][i];
			}
		reader.close();

		received_ids = new int * [size_of_group];
		for (int i =0; i < size_of_group; i++)
		{
			received_ids[i] = new int [interfaces_size[i]];
		}
		cout << " mpi stuff is starting in constructor" << endl;
		MPI_Request request[size_of_group];
		for (int i = 0; i < size_of_group; i++)
		{
			if (i!=rank) MPI_Isend(ids_to_send[i], interfaces_size[i], MPI_INT, i, rank, MPI_COMM_WORLD, &request[i]);
		}
		for (int i = 0; i < size_of_group; i++)
		{
			if (i != rank)
			{
				int * buffer = new int [interfaces_size[i]];
				MPI_Recv(buffer, interfaces_size[i], MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
				for (int j = 0; j < size; j++)
				{
					for (int k = 0; k < interfaces_size[i]; k++)
					{
						if(global_ids_reference[j] == buffer[k]) received_ids[i][k] = j;
					}	
				}
				for (int k = 0; k < interfaces_size[i]; k++)
				{
					for (int j = 0; j < size; j++)
					{
					
						if(global_ids_reference[j] == ids_to_send[i][k]) ids_to_send[i][k] = j;
					}	
				}
				delete [] buffer;
			}
		}
		cout << "mpi stuf finished in constructor" << endl;
		send_data = new double * [size_of_group];
		receive_data = new double * [size_of_group];
		for (int i = 0; i < size_of_group; i++)
		{
			if (interfaces_size[i]!=0)
			{
				send_data[i] = new double [interfaces_size[i]*3];
				receive_data[i] = new double [interfaces_size[i]*3];    
			}
		}
	}
}
