#include "transport.h"
namespace advection
{
    void initialize_arrays(int size, bool*& is_boundary, double *& face_normals, int *& neighbor_ids, double *& tr_areas, double *& xbarys, double *& ybarys)
    {
        is_boundary = new bool [size];
        neighbor_ids = new int [3*size];
        face_normals = new double [3*2*size];
        tr_areas = new double [size];
        xbarys =  new double [size];
        ybarys = new double [size];
    }

    void dealloc_arrays(bool*& is_boundary, double *& face_normals, int *& neighbor_ids, double *& tr_areas, double *& xbarys, double *& ybarys)
    {
        delete [] face_normals;
        delete [] neighbor_ids;
        delete [] tr_areas;
        delete [] xbarys;
        delete [] ybarys;
        delete [] is_boundary;
    }

    double sigma(double const &r, double const &dr, int const &PML, int const & size)
    {
        if (r>(dr*(size-PML))) return pow((r-(size-PML)*dr)/(PML*dr),2)*3.0*log(10.0)*13.0/(PML*dr);
        else return 0;
    }

    advection_2d::~advection_2d()
    {
        dealloc_arrays(is_boundary, face_normals, neighbor_ids, tr_areas, xbarys, ybarys);
    }
	
    void advection_2d::calc_dynamic_arrays()
    {
        std::stringstream ss;
        ss << "centroids_" << rank << ".txt";
        std::cout << ss.str()<< std::endl;
        std::ifstream reader;
        reader.open(ss.str(), std::fstream::in);
        initialize_arrays(size, is_boundary, face_normals, neighbor_ids, tr_areas, xbarys, ybarys);
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
        std::cout << ss.str()<< std::endl;
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
        std::cout << ss.str()<< std::endl;
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
        std::cout << ss.str()<< std::endl;
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
        std::cout << ss.str() << std::endl;
        reader.open(ss.str());

        for (int k = 0; k < size; k++)
        {
            getline(reader, line);
            tr_areas[k] = std::stod(line);
        }
    }

    void advection_2d::solver_small(double *&u, double const dt, double const *vel, double t)
    {
        //insert something here if you need to
        //************************************
        //
        //

        solver_small_cpu(u, dt, vel, t);

        //insert something here if you need to
        //************************************
        //
        //
    }

    void advection_2d::solver_small_cpu(double *&u, double const &dt, double const *vel, double t)
    {
        double * face_values = new double [size*3];
        for (int i = 0; i < size; i++)
        {
        	for (int k = 0; k < 3; k++)
        	{
        		face_values[3*i+k] = 0.5*u[i] + 0.5*u[neighbor_ids[3*i+k]];
        	}
        }
        
        for (int i = 0; i < size; i++)
        {
        	double temp = 0;
        	for (int k = 0; k < 3; k++)
        	{
        		temp += face_values[3*i+k]*(vel[2*i]*face_normals[2*(i*3+k)] + vel[2*i+1]*face_normals[2*(i*3+k)+1]);
        	}
        	u[i] = u[i] - dt * temp/tr_areas[i];
        }
        
        delete [] face_values;
    }
}























