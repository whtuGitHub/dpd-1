#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <vector>
#include <map>
#include <set>
#include <cstdlib>
#include <cstring>
#include <string>
#include <queue>
#include <utility>
#include <algorithm>
#include <sys/types.h>
#include <unistd.h>

#include "lib/rand.cpp"
typedef float64 kfloat;
MPI_Datatype MPI_KFLOAT = MPI_DOUBLE;
#include "lib/mpi-timed.cpp"

const int VERSION = 1;
const char D = 3;
const kfloat PREDICTOR_LAMBDA = 0.5;
const kfloat PI = 3.14159265358979323846264338327950288419716939937510582097494459;
const kfloat sqrt2 = 1.4142135623730950488016887242096980785696718753769480;
const kfloat c1d3 = 1. / 3.;
const kfloat sqrt1d3 = 0.5773502691896257645091487805019574556476017512701268;
kfloat *precomputed_interaction_cutoff;

#ifdef FORCE_TEST
kfloat total_velocity=0;
#endif

//volume of sphere
kfloat sphere_volume(kfloat r) {
  return 4. * PI * r * r * r /  3.;
}

//round to nearest uint64
unsigned long long kround(kfloat x) {
  return (unsigned long long)(x + 0.5);
}

//places scalars onto a buffer
template <class T>
void on_buffer(char *&buff, const T &data) {
  *(T *)buff = data;
  buff += sizeof(T);
}

//reads scalars from a buffer
template <class T>
void off_buffer(char *&buff, T &data) {
  data=*(T *)buff;
  buff += sizeof(T);
}

// entity count information
typedef struct {
  int type, number;
  kfloat density;
} exact_number;

// particle type information
typedef struct {
  kfloat mass;
  kfloat inverse_moment[D];
  bool write;
  int first;
  int last;
  std::string name;
  char type;
  kfloat radius;
  kfloat activity;
} particle_type;

// interaction type information
typedef struct {
  kfloat lambda;
  kfloat gamma_s;
  kfloat sigma_s;
  kfloat gamma_c;
  kfloat sigma_c;
  kfloat a;
  kfloat b;
  kfloat c;
  kfloat (*force_t)(const kfloat, const kfloat,
                    const kfloat, const kfloat);
} interaction_type;

// simulation parameters
namespace parameters {
  kfloat cutoff;
  kfloat cutoff_squared;
  kfloat delta_t;
  kfloat kT;
  int box_x[D];
  kfloat density;
  kfloat volume;
  unsigned int particle_count;
  int ptype_count;
  particle_type *ptype_map;
  interaction_type **itype_map;
  int x_ranks;
  int y_ranks;
  int z_ranks;
  unsigned long long total_steps;
  kfloat neighbor_cutoff;
  kfloat neighbor_cutoff_squared;
  int neighbor_box_cutoff;
  unsigned long long neighbor_search_time;
  unsigned long long write_time;
  unsigned long long backup_time;
  bool continuation;
  kfloat *activity;
#ifdef DETERMINE_VISCOSITY
  kfloat pforce;
#endif
}

#include "lib/vmath.cpp"

// function is defined at runtime from options below
kfloat (*weight_function)(const kfloat);

kfloat weight_function_0_125(const kfloat r) {
  return std::sqrt(std::sqrt(std::sqrt(1. - r / parameters::cutoff)));
}

kfloat weight_function_0_25(const kfloat r) {
  return std::sqrt(std::sqrt(1. - r / parameters::cutoff));
}

kfloat weight_function_0_50(const kfloat r) {
  return std::sqrt(1. - r / parameters::cutoff);
}

kfloat weight_function_1_00(const kfloat r) {
  return 1. - r / parameters::cutoff;
}

kfloat weight_function_2_00(const kfloat r) {
  return (1. - r / parameters::cutoff) * (1. - r / parameters::cutoff);
}

// two force types
kfloat force_t_a(const kfloat r, const kfloat a, const kfloat b,
                                                 const kfloat c) {
  if (r > 1.) return 0;
  return a * (1. - r);
}

kfloat force_t_b(const kfloat r, const kfloat a, const kfloat b,
                                                 const kfloat c) {
  if (r > 1.) return 0;
  return a * (std::exp(-b * r) - c)/(1. - c);
}

// std::ofstream extension
class kofstream: public std::ofstream {
public:
  void open(const char *, std::ios_base::openmode);
  void close();
};

#include "lib/particle.cpp"
#include "lib/run_variable.cpp"

// the process keeps track of its open files
void kofstream::open(const char *filename,
                     std::ios_base::openmode mode=std::ios_base::out) {
  this->std::ofstream::open(filename, mode);
  run_var.open_file_list.push_back(&(*this));
}

// close this file (and remove from active files list)
void kofstream::close() {
  this->std::ofstream::close();
  run_var.open_file_list.remove(&(*this));
}

// abort program and print exit code
void crash(int i) {
  std::cerr << "CRASH: Exit Code " << i << "!\n";
  std::cerr.flush(); std::cout.flush();
  for (std::list<kofstream*>::iterator p  =
       run_var.open_file_list.begin(); p !=
       run_var.open_file_list.end(); p++) {
    (*p)->close();
  }
  MPI::abort(i);
  std::exit(i);
}

#include "lib/neighbor.cpp"
#include "lib/neighbor-util.cpp"
#include "lib/fileio.cpp"

void neighbor_search_alone() {
  for (unsigned int i=0;i<run_var.visible_particles;++i) {
    run_var.particle_index[run_var.particle_array[i].index]=i;
  }
  run_var.empty_the_box();
  run_var.fill_the_box();
  particle temp;
  unsigned int fill_index=0;
  for (int i=0; i<parameters::box_x[0];++i) {
    for (int j=0;j<parameters::box_x[1];++j) {
      for (int k=0;k<parameters::box_x[2];++k) {
        std::vector<unsigned int> *box=&run_var.particle_box[i][j][k];
        for (int n=0,max_n=box->size();n<max_n;++n) {
          temp=run_var.particle_array[fill_index];
          run_var.particle_array[fill_index]=run_var.particle_array[run_var.particle_index[(*box)[n]]];
          run_var.particle_array[run_var.particle_index[(*box)[n]]]=temp;
          run_var.particle_index[temp.index]=run_var.particle_index[(*box)[n]];
          run_var.particle_index[run_var.particle_array[fill_index].index]=fill_index;
          (*box)[n]=fill_index;
          ++fill_index;
        }
      }
    }
  }
  int ints=0;
  run_var.interaction_list.resize(run_var.my_particles);
  for (unsigned int i=0;i<run_var.my_particles;++i) {
    run_var.interaction_list[i].clear();
    run_var.particle_index[run_var.particle_array[i].index]=i;
    particle *p=&run_var.particle_array[i];
    for (int box_i=-parameters::neighbor_box_cutoff;box_i<=parameters::neighbor_box_cutoff;++box_i) {
      for (int box_j=-parameters::neighbor_box_cutoff;box_j<=parameters::neighbor_box_cutoff;++box_j) {
        for (int box_k=-parameters::neighbor_box_cutoff;box_k<=parameters::neighbor_box_cutoff;++box_k) {
          int b_i=(int)p->r[0]+box_i; if (b_i<0) b_i+=parameters::box_x[0]; else if (b_i>=parameters::box_x[0]) b_i-=parameters::box_x[0];
          int b_j=(int)p->r[1]+box_j; if (b_j<0) b_j+=parameters::box_x[1]; else if (b_j>=parameters::box_x[1]) b_j-=parameters::box_x[1];
          int b_k=(int)p->r[2]+box_k; if (b_k<0) b_k+=parameters::box_x[2]; else if (b_k>=parameters::box_x[2]) b_k-=parameters::box_x[2];
          std::vector<unsigned int> *box=&run_var.particle_box[b_i][b_j][b_k];
          for (int j=0,max_j=box->size();j<max_j;++j) {
            particle *q=&run_var.particle_array[(*box)[j]];
            if (i<(*box)[j] && vmath::distance_squared(p->r,q->r)<parameters::neighbor_cutoff_squared) {
              ints++;
              run_var.interaction_list[i].push_back((*box)[j]);
            }
          }
        }
      }
    }
  }
}

void neighbor_search() {
  run_var.interaction_list.clear();
  if (run_var.neighbor_count==0) return neighbor_search_alone();
  
  for (unsigned int i=0;i<run_var.neighbor_count;i++) {
    run_var.neighbors[i].search_init();
  }
  
  particle_exchange();
  
  //test particles for sending to neighbors
  for (unsigned int i=run_var.search_point;i<run_var.my_particles;++i) for (unsigned int j=0;j<run_var.neighbor_count;++j) {
    run_var.neighbors[j].search_check(&run_var.particle_array[i],i);
  }
  
  int out_buffer_size=0;
  int in_buffer_size=0;
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    out_buffer_size+=run_var.neighbors[i].search_out_list.size()*particle::search_size+sizeof(int);
    in_buffer_size+=run_var.neighbors[i].search_in_size;
  }
  int buffer_size=out_buffer_size+in_buffer_size;
  
  //send & receive particle information
  char *particle_buffer=new char[buffer_size];
  char *particle_out_buffer=particle_buffer;
  char *particle_in_buffer=particle_buffer+out_buffer_size;
  MPI_Request *req=new MPI_Request[run_var.neighbor_count];
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    run_var.neighbors[i].search_receive_particles(particle_in_buffer,&req[i]);
  }
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    run_var.neighbors[i].search_send_particles(particle_out_buffer);
  }
  
  int index;
  run_var.visible_particles=run_var.my_particles;
  int slot=run_var.my_particles;
  //put incoming particles into vector
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    MPI::wait_any(run_var.neighbor_count,req,&index);
    run_var.neighbors[index].search_process(slot);
  }
  
  run_var.empty_the_box();
  run_var.fill_the_box();
  particle temp;
  unsigned int fill_index=0;
  unsigned int fill_index_n=run_var.my_particles;
  for (int i=0; i<parameters::box_x[0];++i) {
    for (int j=0;j<parameters::box_x[1];++j) {
      for (int k=0;k<parameters::box_x[2];++k) {
        std::vector<unsigned int> *box=&run_var.particle_box[i][j][k];
        for (int n=0,max_n=box->size();n<max_n;++n) {
          if (run_var.particle_array[run_var.particle_index[(*box)[n]]].rank==MPI::rank) {
            temp=run_var.particle_array[fill_index];
            run_var.particle_array[fill_index]=run_var.particle_array[run_var.particle_index[(*box)[n]]];
            run_var.particle_array[run_var.particle_index[(*box)[n]]]=temp;
            run_var.particle_index[temp.index]=run_var.particle_index[(*box)[n]];
            run_var.particle_index[run_var.particle_array[fill_index].index]=fill_index;
            (*box)[n]=fill_index;
            ++fill_index;
          }
          else {
            temp=run_var.particle_array[fill_index_n];
            run_var.particle_array[fill_index_n]=run_var.particle_array[run_var.particle_index[(*box)[n]]];
            run_var.particle_array[run_var.particle_index[(*box)[n]]]=temp;
            run_var.particle_index[temp.index]=run_var.particle_index[(*box)[n]];
            run_var.particle_index[run_var.particle_array[fill_index_n].index]=fill_index_n;
            (*box)[n]=fill_index_n;
            ++fill_index_n;
          }
        }
      }
    }
  }
  int ints=0;
  run_var.interaction_list.resize(run_var.my_particles);
  
  for (unsigned int i=0;i<run_var.my_particles;++i) {
    run_var.interaction_list[i].clear();
    particle *p=&run_var.particle_array[i];
    for (int box_i=-parameters::neighbor_box_cutoff;box_i<=parameters::neighbor_box_cutoff;++box_i) {
      for (int box_j=-parameters::neighbor_box_cutoff;box_j<=parameters::neighbor_box_cutoff;++box_j) {
        for (int box_k=-parameters::neighbor_box_cutoff;box_k<=parameters::neighbor_box_cutoff;++box_k) {
          int b_i=(int)p->r[0]+box_i; if (b_i<0) b_i+=parameters::box_x[0]; else if (b_i>=parameters::box_x[0]) b_i-=parameters::box_x[0];
          int b_j=(int)p->r[1]+box_j; if (b_j<0) b_j+=parameters::box_x[1]; else if (b_j>=parameters::box_x[1]) b_j-=parameters::box_x[1];
          int b_k=(int)p->r[2]+box_k; if (b_k<0) b_k+=parameters::box_x[2]; else if (b_k>=parameters::box_x[2]) b_k-=parameters::box_x[2];
          std::vector<unsigned int> *box=&run_var.particle_box[b_i][b_j][b_k];
          
          for (unsigned int j=0,max_j=box->size();j<max_j;++j) {
            unsigned int qi=(*box)[j];
            particle *q=&run_var.particle_array[qi];
            if (i<qi && vmath::distance_squared(p->r,q->r)<parameters::neighbor_cutoff_squared) {
              if (q->rank==MPI::rank) {
                ints++;
                run_var.interaction_list[i].push_back(qi);
              }
              else {
                neighbor *n=&run_var.neighbors[run_var.rank_to_neighbor[q->rank]];
                if (
                  ((p->rank<q->rank) && (p->index+q->index)%2==0) || 
                  ((p->rank>q->rank) && (p->index+q->index)%2==1)
                ) {
                  ints++;
                  run_var.interaction_list[i].push_back(qi);
                  if (n->force_send_list.find(qi)==n->force_send_list.end()) {
                    n->force_send_list.insert(qi);
                    n->force_send_size+=particle::force_send_size;
                    n->position_receive_size+=particle::position_send_size;
                  }
                }
                else {
                  if (n->force_receive_list.find(i)==n->force_receive_list.end()) {
                    n->force_receive_list.insert(i);
                    n->force_receive_size+=particle::force_send_size;
                    n->position_send_size+=particle::position_send_size;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  
  MPI::empty_all();
  delete [] req;
  delete [] particle_buffer;
}

void communicate_forces() {
  if (run_var.neighbor_count==0) return;
  MPI::empty(req_out);
  if (run_var.buffer) delete [] run_var.buffer;
  
  int total_send_size=0;
  int total_receive_size=0;
  int total_receive_count=0;
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    total_send_size+=run_var.neighbors[i].force_send_size;
    total_receive_size+=run_var.neighbors[i].force_receive_size;
    total_receive_count+=run_var.neighbors[i].force_receive_list.size();
  }
  
  char *incoming_buffer=new char[total_receive_size];
  run_var.buffer=new char[total_send_size];
  char *in_buffer=incoming_buffer;
  char *out_buffer=run_var.buffer;
  MPI_Request *req=new MPI_Request[run_var.neighbor_count];
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    run_var.neighbors[i].force_receive(in_buffer,&req[i]);
  }
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    run_var.neighbors[i].force_send(out_buffer);
  }
  
  int index;
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    MPI::wait_any(run_var.neighbor_count,req,&index);
    run_var.neighbors[index].force_process();
  }
  
  delete [] req;
  delete [] incoming_buffer;
}

void communicate_positions() {
  if (run_var.neighbor_count==0) return;
  MPI::empty(req_out);
  if (run_var.buffer) delete [] run_var.buffer;
  
  int total_send_size=0;
  int total_receive_size=0;
  int total_receive_count=0;
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    total_send_size+=run_var.neighbors[i].position_send_size;
    total_receive_size+=run_var.neighbors[i].position_receive_size;
    total_receive_count+=run_var.neighbors[i].force_send_list.size(); //force send is position receive
  }
  
  char *incoming_buffer=new char[total_receive_size];
  run_var.buffer=new char[total_send_size];
  char *in_buffer=incoming_buffer;
  char *out_buffer=run_var.buffer;
  MPI_Request *req=new MPI_Request[run_var.neighbor_count];
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    run_var.neighbors[i].position_receive(in_buffer,&req[i]);
  }
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    run_var.neighbors[i].position_send(out_buffer);
  }
  
  int index;
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    MPI::wait_any(run_var.neighbor_count,req,&index);
    run_var.neighbors[index].position_process();
  }
  
  delete [] req;
  delete [] incoming_buffer;
}

void run_algorithm(rngSource generator) {
  unsigned long long neighbor_search_t=parameters::neighbor_search_time;
  for (unsigned long long t=0;t<parameters::total_steps;) {
#ifdef NCUTOFF_TEST
    static bool once=true;
    static unsigned int num_s=0;
    static kfloat **x=new kfloat*[run_var.my_particles];
    static kfloat d=0;
#endif
    if (neighbor_search_t==parameters::neighbor_search_time) {
      if (MPI::rank==0) std::cerr << (kfloat)t*parameters::delta_t << "\n";
      
      MPI::tsearch.start();
      neighbor_search();
      MPI::tsearch.stop();
      
      neighbor_search_t=0;
#ifdef NCUTOFF_TEST
      ++num_s;
      if (num_s>4) {
      if (once) {
        once=false;
        for (unsigned int i=0;i<run_var.my_particles;++i) {
          x[i]=new kfloat[D];
          for (int j=0;j<D;++j) x[i][j]=run_var.particle_array[run_var.particle_index[i]].r[j];
        }
      }
      for (unsigned int i=0;i<run_var.my_particles;++i) {
        for (int j=0;j<D;++j) x[i][j]=run_var.particle_array[run_var.particle_index[i]].r[j];
      }
      }
#endif
#ifdef DETERMINE_VISCOSITY
      kfloat total_v[D];
      vmath::zero(total_v);
      for (unsigned int i=0;i<run_var.my_particles;++i) {
        vmath::add(total_v,run_var.particle_array[i].v);
      }
      kfloat rtotal_v[D];
      MPI::allred(&total_v,&rtotal_v,D,MPI_KFLOAT,MPI_SUM);
      vmath::div(rtotal_v,parameters::particle_count);
      for (unsigned int i=0;i<run_var.my_particles;++i) {
        vmath::sub(run_var.particle_array[i].v,rtotal_v);
        vmath::sub(run_var.particle_array[i].v_,rtotal_v);
      }
      static unsigned long long t_start=(unsigned long long)(100./parameters::delta_t);
      if (t+run_var.elapsed_steps>t_start) for (unsigned int i=0;i<run_var.my_particles;++i) {
        int bin=(int)(run_var.particle_array[i].r[0]/run_var.v_division);
        ++run_var.v_count[bin];
        run_var.v_sum[bin]+=run_var.particle_array[i].v[1];
      }
#endif
    }
    ++neighbor_search_t;
    
    MPI::tposition.start();
    communicate_positions();
    MPI::tposition.stop();
    
    MPI::tinteract.start();
    for (unsigned int i=0;i<run_var.my_particles;++i) {
      for (unsigned int j=0;j<run_var.interaction_list[i].size();) {
        if (run_var.particle_array[i].interact(run_var.particle_array[run_var.interaction_list[i][j]],generator,precomputed_interaction_cutoff[neighbor_search_t-1])) {
          for (unsigned int k=j;k<run_var.interaction_list[i].size()-1;++k) {
            run_var.interaction_list[i][k]=run_var.interaction_list[i][k+1];
          }
          run_var.interaction_list[i].resize(run_var.interaction_list[i].size()-1);
        } else ++j;
      }
    }
    MPI::tinteract.stop();
    
    MPI::tforce.start();
    communicate_forces();
    MPI::tforce.stop();
    MPI::tstep.start();
    for (unsigned int i=0;i<run_var.my_particles;++i) {
      run_var.particle_array[i].step();
    }
    MPI::tstep.stop();
#ifdef FORCE_TEST
    total_velocity=0;
    for (unsigned int i=0;i<run_var.my_particles;++i) {
      if (run_var.particle_array[i].type==0) {
        total_velocity+=run_var.particle_array[i].v[0];
      }
    }
    total_velocity/=(run_var.my_particles-1);
#endif
#ifdef NCUTOFF_TEST
    if (num_s>4) {
    for (unsigned int i=0;i<run_var.my_particles;++i) {
      kfloat dd=vmath::distance_squared(x[i],run_var.particle_array[run_var.particle_index[i]].r);
      if (dd>d) {
        d=dd;
        std::cerr << "MAX: " << sqrt(d) << "\n";
      }
    }
    }
#endif
#ifdef TEMPERATURE_TEST
    if ((kfloat)t*parameters::delta_t>0) {
      kfloat avg=0;
      kfloat avg2=0;
      unsigned int count=0;
      for (unsigned int i=0;i<run_var.my_particles;++i) {
        if (run_var.particle_array[i].type==1) {
          ++count;
          avg+=run_var.particle_array[i].temperature();
          avg2+=run_var.particle_array[i].temperature_a();
        }
      }
      std::cout << avg/count << " " << avg2/count << "\n";
    }
#endif
    //write
    ++t;
    if (parameters::write_time && (t+run_var.elapsed_steps)%parameters::write_time==0) {
      write_coordinates(run_var.output_file);
      run_var.output_location=run_var.output_file.tellp();
    }
    
    if (parameters::backup_time && t%parameters::backup_time==0) {
      backup(t);
    }
  }
}

void init() {
  parameters::continuation=false;
  parameters::x_ranks=1;
  parameters::y_ranks=1;
  parameters::z_ranks=1;
}

int main(int argc,char* argv[]) {
  init();
  std::string input_string="input";
  std::string output_root="output";
  long seed_override=0;
  bool show_cycles=false;
  rngSource generator;
  
  //parse command line
  for (int i=1;i<argc;++i) {
    if (!std::strcmp(argv[i],"-continue")) {
      parameters::continuation=true;
      continue;
    }
    if (!std::strcmp(argv[i],"-show-cycles")) {
      show_cycles=true;
      continue;
    }
    if (i==argc-1) continue;
    if (!std::strcmp(argv[i],"-seed")) {
      seed_override=std::atoi(argv[++i]);
      continue;
    }
    if (!std::strcmp(argv[i],"-input")) {
      input_string=argv[++i];
      continue;
    }
    if (!std::strcmp(argv[i],"-output")) {
      output_root=argv[++i];
      continue;
    }
    if (!std::strcmp(argv[i],"-x")) {
      parameters::x_ranks=std::atoi(argv[++i]);
      continue;
    }
    if (!std::strcmp(argv[i],"-y")) {
      parameters::y_ranks=std::atoi(argv[++i]);
      continue;
    }
    if (!std::strcmp(argv[i],"-z")) {
      parameters::z_ranks=std::atoi(argv[++i]);
      continue;
    }
    
    std::cerr << "Unknown command line parameter \"" << argv[i] << "\"! (Aborting)\n";
    crash(1);
  }
  
  MPI::init(&argc,&argv);
  
  //seed rng for initialization
  if (seed_override) {
    generator.rseed(seed_override);
  }
  else {
    long this_seed=getpid()*time(NULL);
    MPI::bcast(&this_seed,sizeof(this_seed),0);
    generator.rseed(this_seed);
  }
  std::ifstream input_file(input_string.c_str());
  if (!input_file.is_open()) {
    std::cerr << "Unable to open input file \"" << input_string << "\"! (Aborting)\n";
    crash(2);
  }
  
  std::stringstream ss;
  ss.str(std::string());
  ss << output_root << MPI::rank << ".vtf";
  std::string output_string=ss.str();
  
  if (!parameters::continuation) {
    run_var.output_file.open(output_string.c_str(),std::ios_base::out|std::ios_base::trunc);
    if (!run_var.output_file.is_open()) {
      std::cerr << "Unable to open output file \"" << output_string << "\"! (Aborting)\n";
      crash(2);
    }
  }
  
  read_parameters(input_file,generator);
  
#ifdef DETERMINE_VISCOSITY
  ss.str(std::string());
  ss << output_string << "_v" << MPI::rank;
  std::string viscosity_string=ss.str();
  run_var.viscosity_file.open(viscosity_string.c_str(),std::ios_base::out|std::ios_base::trunc);
  if (!run_var.viscosity_file.is_open()) {
    std::cerr << "Unable to open viscosity file \"" << viscosity_string << "\"! (Aborting)\n";
    crash(2);
  }
  
  ss.str(std::string());
  ss << output_root << "_v" << MPI::rank << ".backup";
  run_var.backup_viscosity_name=ss.str();
#endif
  ss.str(std::string());
  ss << output_root << MPI::rank << ".backup";
  run_var.backup_file_name=ss.str();
  
  if (parameters::continuation) {
    std::ifstream restore_file(run_var.backup_file_name.c_str());
    int old_version;
    restore_file.read((char*)&old_version,sizeof(int));
    restore_file.read((char*)&run_var.elapsed_steps,sizeof(unsigned long long));
    restore_file.read((char*)&run_var.output_location,sizeof(unsigned long long));
    restore_file.read((char*)&run_var.my_particles,sizeof(int));
    restore_file.read((char*)&run_var.visible_particles,sizeof(int));
    run_var.particle_array.resize(run_var.visible_particles);
    for (unsigned int i=0;i<run_var.visible_particles;++i) {
      run_var.particle_array[i].restore(restore_file);
      run_var.particle_index[run_var.particle_array[i].index]=i;
    }
    restore_file.close();
    
    run_var.output_file.open(output_string.c_str(),std::ios_base::out|std::ios_base::app);
    if (!run_var.output_file.is_open()) {
      std::cerr << "Unable to open output file \"" << output_string << "\"! (Aborting)\n";
      crash(2);
    }
    if (truncate(output_string.c_str(),run_var.output_location)) {
      std::cerr << "Failed to truncate file \"" << output_string << "\"! (Aborting)\n";
      crash(2);
    }
    run_var.output_file.seekp(run_var.output_location);
  }
  
  //seed rng for run
  if (seed_override) {
    generator.rseed(seed_override*(MPI::rank+1));
  }
  else {
    generator.rseed((MPI::rank+1)*time(NULL)*getpid());
  }
  
  run_var.init();
  run_algorithm(generator);
  backup(parameters::total_steps);
#ifdef DETERMINE_VISCOSITY
  run_var.viscosity_print();
#endif
  if (show_cycles) MPI::clock_assembly();
  MPI::end();
  
  return 0;
}
