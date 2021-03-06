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
#include "lib/neighbor-search.cpp"
#include "lib/neighbor-comm.cpp"
#include "lib/fileio.cpp"
#include "lib/algorithm.cpp"

// initial parameters
void init() {
  parameters::continuation = false;
  parameters::x_ranks = 1;
  parameters::y_ranks = 1;
  parameters::z_ranks = 1;
}

// main
int main(int argc,char* argv[]) {
  init();
  // defaults
  std::string input_string = "input";
  std::string output_root = "output";
  long seed_override = 0;
  bool show_cycles = false;
  rngSource generator;
  
  //parse command line
  for (int i=1;i<argc;++i) {
    if (!std::strcmp(argv[i], "-continue")) {
      parameters::continuation = true;
      continue;
    }
    if (!std::strcmp(argv[i], "-show-cycles")) {
      show_cycles = true;
      continue;
    }
    if (i == argc-1) continue;
    if (!std::strcmp(argv[i], "-seed")) {
      seed_override = std::atoi(argv[++i]);
      continue;
    }
    if (!std::strcmp(argv[i], "-input")) {
      input_string = argv[++i];
      continue;
    }
    if (!std::strcmp(argv[i], "-output")) {
      output_root = argv[++i];
      continue;
    }
    if (!std::strcmp(argv[i], "-x")) {
      parameters::x_ranks = std::atoi(argv[++i]);
      continue;
    }
    if (!std::strcmp(argv[i], "-y")) {
      parameters::y_ranks = std::atoi(argv[++i]);
      continue;
    }
    if (!std::strcmp(argv[i], "-z")) {
      parameters::z_ranks = std::atoi(argv[++i]);
      continue;
    }
    
    std::cerr << "Unknown command line parameter \"" << argv[i] <<
                                                     "\"! (Aborting)\n";
    crash(1);
  }
  
  MPI::init(&argc, &argv);
  
  //seed rng for initialization
  if (seed_override) generator.rseed(seed_override);
  else {
    long this_seed = getpid() * time(NULL);
    MPI::bcast(&this_seed, sizeof(this_seed), 0);
    generator.rseed(this_seed);
  }
  
  
  std::ifstream input_file(input_string.c_str());
  if (!input_file.is_open()) {
    std::cerr << "Unable to open input file \"" << input_string <<
                                                     "\"! (Aborting)\n";
    crash(2);
  }
  
  // set output file name
  std::stringstream ss;
  ss.str(std::string());
  ss << output_root << MPI::rank << ".vtf";
  std::string output_string = ss.str();
  
  // if not restoring from a backup
  if (!parameters::continuation) {
    run_var.output_file.open(output_string.c_str(),
                             std::ios_base::out | std::ios_base::trunc);
    if (!run_var.output_file.is_open()) {
      std::cerr << "Unable to open output file \"" << output_string << "\"! (Aborting)\n";
      crash(2);
    }
  }
  
  read_parameters(input_file, generator);
  
#ifdef DETERMINE_VISCOSITY
  ss.str(std::string());
  ss << output_string << "_v" << MPI::rank;
  std::string viscosity_string = ss.str();
  run_var.viscosity_file.open(viscosity_string.c_str(),
                             std::ios_base::out | std::ios_base::trunc);
  if (!run_var.viscosity_file.is_open()) {
    std::cerr << "Unable to open viscosity file \"" <<
                                 viscosity_string << "\"! (Aborting)\n";
    crash(2);
  }
  
  ss.str(std::string());
  ss << output_root << "_v" << MPI::rank << ".backup";
  run_var.backup_viscosity_name = ss.str();
#endif
  ss.str(std::string());
  ss << output_root << MPI::rank << ".backup";
  run_var.backup_file_name = ss.str();
  
  // restore backup if continuing
  if (parameters::continuation) {
    std::ifstream restore_file(run_var.backup_file_name.c_str());
    int old_version;
    restore_file.read((char*)&old_version, sizeof(int));
    restore_file.read((char*)&run_var.elapsed_steps,
                              sizeof(unsigned long long));
    restore_file.read((char*)&run_var.output_location,
                              sizeof(unsigned long long));
    restore_file.read((char*)&run_var.my_particles, sizeof(int));
    restore_file.read((char*)&run_var.visible_particles, sizeof(int));
    run_var.particle_array.resize(run_var.visible_particles);
    for (unsigned int i = 0; i < run_var.visible_particles; ++i) {
      run_var.particle_array[i].restore(restore_file);
      run_var.particle_index[run_var.particle_array[i].index] = i;
    }
    restore_file.close();
    
    run_var.output_file.open(output_string.c_str(),
                             std::ios_base::out | std::ios_base::app);
    if (!run_var.output_file.is_open()) {
      std::cerr << "Unable to open output file \"" << output_string <<
                                                     "\"! (Aborting)\n";
      crash(2);
    }
    if (truncate(output_string.c_str(),run_var.output_location)) {
      std::cerr << "Failed to truncate file \"" << output_string <<
                                                     "\"! (Aborting)\n";
      crash(2);
    }
    run_var.output_file.seekp(run_var.output_location);
  }
  
  //seed rng for run
  if (seed_override) generator.rseed(seed_override * (MPI::rank + 1));
  else generator.rseed((MPI::rank + 1) * time(NULL) * getpid());
  
  run_var.init();
  run_algorithm(generator);
  
  // backup on exit
  backup(parameters::total_steps);
#ifdef DETERMINE_VISCOSITY
  run_var.viscosity_print();
#endif
  // print timer stats if requested
  if (show_cycles) MPI::clock_assembly();
  MPI::end();
  
  return 0;
}
