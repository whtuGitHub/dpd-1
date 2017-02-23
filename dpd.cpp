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

// std::ofstream extension
class kofstream: public std::ofstream {
public:
  void open(const char *, std::ios_base::openmode);
  void close();
};

// vector math operations
namespace vmath {
  void zero(kfloat *v) {
    for (int i = 0; i < D; ++i) v[i] = 0;
  }
  
  // translate vector into periodic boundary conditions
  void pbc(kfloat *v) {
    for (int i = 0; i < D; ++i) v[i] -=
      std::floor(v[i] / parameters::box_x[i]) * parameters::box_x[i];
  }
  
  // (pbc) distance between i components
  kfloat distance(kfloat *v, kfloat *u, int i) {
    return v[i] - u[i] - std::floor((v[i] - u[i]) /
           parameters::box_x[i] + 0.5) * parameters::box_x[i];
  }
  
  // absolute value of the distance
  kfloat abs_distance(kfloat *v, kfloat *u, int i) {
    return std::abs(distance(v, u, i));
  }
  
  // distance squared between vectors
  kfloat distance_squared(kfloat *v, kfloat *u) {
    kfloat d2 = 0;
    for (int i = 0; i < D; ++i) {
      kfloat d = distance(v, u, i);
      d2 += d * d;
    }
    return d2;
  }
  
  // set v to a x b
  void set_cross(kfloat *v, kfloat *a, kfloat *b) {
    v[0] = a[1] * b[2] - a[2] * b[1];
    v[1] = a[2] * b[0] - a[0] * b[2];
    v[2] = a[0] * b[1] - a[1] * b[0];
  }
  
  // set v to a - b (pbc)
  void set_sub_pbc(kfloat *v, kfloat *a, kfloat *b) {
    for (int i = 0; i < D; ++i) v[i] = distance(a, b, i);
  }
  
  // set v to a - b (no pbc)
  void set_sub(kfloat *v, kfloat *a, kfloat *b) {
    for (int i = 0; i < D; ++i) v[i] = a[i] - b[i];
  }
  
  // set v to a / b
  void set_div(kfloat *v, kfloat *a, kfloat b) {
    for (int i = 0; i < D; ++i) v[i] = a[i] / b;
  }
  
  // set v to a * b (scalar)
  void set_mul(kfloat *v, kfloat *a, kfloat b) {
    for (int i = 0; i < D; ++i) v[i] = a[i] * b;
  }
  
  // v += a * b (scalar)
  void add_mul(kfloat *v, kfloat *a, kfloat b) {
    for (int i = 0; i < D; ++i) v[i] += a[i] * b;
  }
  
  // v += a
  void add(kfloat *v, kfloat *a) {
    for (int i = 0; i < D; ++i) v[i] += a[i];
  }
  
  // v -= a
  void sub(kfloat *v, kfloat *a) {
    for (int i = 0; i < D; ++i) v[i] -= a[i];
  }
  
  // v /= a
  void div(kfloat *v, kfloat a) {
    for (int i = 0; i < D; ++i) v[i] /= a;
  }
  
  // 3d rotation about angles a
  void rotate(kfloat *v, kfloat *a) {
    kfloat c[3];
    kfloat s[3];
    kfloat v_[3];
    for (int i = 0; i < D; ++i) {
      c[i]  = std::cos(a[i]);
      s[i]  = std::sin(a[i]);
      v_[i] = v[i];
    }
    v[0] = c[1] * c[2] * v_[0] - c[1] * s[2] * v_[1] + s[1] * v_[2];
    v[1] = (c[0] * s[2] + c[2] * s[0] * s[1]) * v_[0] + (c[0] * c[2] -
            s[0] * s[1] * s[2]) * v_[1] - c[1] * s[0] * v_[2];
    v[2] = (s[0] * s[2] - c[0] * c[2] * s[1]) * v_[0] + (c[2] * s[0] +
            c[0] * s[1] * s[2]) * v_[1] + c[0] * c[1] * v_[2];
  }
  
  // v dot u
  kfloat dot(kfloat *v, kfloat *u) {
    return v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
  }
  
  // vector length
  kfloat magnitude(kfloat *v) {
    return std::sqrt(dot(v, v));
  }
  
  // gaussian distributed zero-mean unit variance matrix
  void randomize_matrix(kfloat v[][D], rngSource &gen) {
    for (int i = 0; i < D; ++i)
      for (int j = 0; j < D; ++j) v[i][j] = gen.rNormFloat64();
  }
  
  // write vector into buffer
  void to_buffer(char *&buff, kfloat *v) {
    for (int i = 0; i < D; ++i) on_buffer(buff, v[i]);
  }
  
  // read vector from buffer
  void from_buffer(char *&buff, kfloat *v) {
    for (int i = 0; i < D; ++i) off_buffer(buff, v[i]);
  }
}

kfloat (*weight_function)(const kfloat);

kfloat weight_function_0_125(const kfloat r) {
  return std::sqrt(std::sqrt(std::sqrt(1.-r/parameters::cutoff)));
}

kfloat weight_function_0_25(const kfloat r) {
  return std::sqrt(std::sqrt(1.-r/parameters::cutoff));
}

kfloat weight_function_0_50(const kfloat r) {
  return std::sqrt(1.-r/parameters::cutoff);
}

kfloat weight_function_1_00(const kfloat r) {
  return 1.-r/parameters::cutoff;
}

kfloat weight_function_2_00(const kfloat r) {
  return (1.-r/parameters::cutoff)*(1.-r/parameters::cutoff);
}

kfloat force_t_a(const kfloat r,const kfloat a,const kfloat b,const kfloat c) {
  if (r>1.) return 0;
  return a*(1.-r);
}

kfloat force_t_b(const kfloat r,const kfloat a,const kfloat b,const kfloat c) {
  if (r>1.) return 0;
  return a*(std::exp(-b*r)-c)/(1.-c);
}

#ifdef FORCE_TEST
kfloat tot_velocity=0;
#endif
class particle {
public:
  short type;
  int index;
  int rank;
  kfloat r[D];
  kfloat v[D];
  kfloat v_[D];
  kfloat f[D];
  kfloat a[D];
  kfloat o[D];
  kfloat o_[D];
  kfloat t[D];
  
  static const int force_send_size=3*sizeof(kfloat)+sizeof(int);
  static const int position_send_size=12*sizeof(kfloat)+sizeof(int);
  static const int search_size=3*sizeof(r[0])+sizeof(index)+sizeof(type)+sizeof(rank);
  static const int transfer_size=18*sizeof(r[0])+sizeof(index)+sizeof(type);
  
  void transfer_to_buffer(char *&buff) {
    on_buffer(buff,type);
    on_buffer(buff,index);
    vmath::to_buffer(buff,r);
    vmath::to_buffer(buff,v);
    vmath::to_buffer(buff,v_);
    vmath::to_buffer(buff,o);
    vmath::to_buffer(buff,o_);
    vmath::to_buffer(buff,a);
  }
  
  void transfer_from_buffer(char *&buff) {
    off_buffer(buff,type);
    off_buffer(buff,index);
    vmath::from_buffer(buff,r);
    vmath::from_buffer(buff,v);
    vmath::from_buffer(buff,v_);
    vmath::from_buffer(buff,o);
    vmath::from_buffer(buff,o_);
    vmath::from_buffer(buff,a);
    rank=MPI::rank;
  }
  
  void search_to_buffer(char *&buff) {
    on_buffer(buff,type);
    on_buffer(buff,index);
    on_buffer(buff,rank);
    vmath::to_buffer(buff,r);
  }
  
  void search_from_buffer(char *&buff) {
    off_buffer(buff,type);
    off_buffer(buff,index);
    off_buffer(buff,rank);
    vmath::from_buffer(buff,r);
  }
  
  void force_to_buffer(char *&buff) {
    on_buffer(buff,index);
    vmath::to_buffer(buff,f);
    vmath::zero(f);
  }
  
  void force_from_buffer(char *&buff) {
    kfloat temp[D];
    vmath::from_buffer(buff,temp);
    vmath::add(f,temp);
  }
  
  void position_to_buffer(char *&buff) {
    on_buffer(buff,index);
    vmath::to_buffer(buff,r);
    vmath::to_buffer(buff,v_);
    vmath::to_buffer(buff,a);
    vmath::to_buffer(buff,o_);
  }
  
  void position_from_buffer(char *&buff) {
    vmath::from_buffer(buff,r);
    vmath::from_buffer(buff,v_);
    vmath::from_buffer(buff,a);
    vmath::from_buffer(buff,o_);
  }
  
  particle() {
    type=0;
    index=0;
    for (int i=0;i<D;++i) {
      r[i]=0;
      v[i]=0;
      f[i]=0;
      a[i]=0;
      o[i]=0;
      t[i]=0;
    }
    for (int i=0;i<D;++i) v_[i]=v[i];
    for (int i=0;i<D;++i) o_[i]=o[i];
  }
  
  particle(kfloat x,kfloat y,kfloat z,int mtype,int mindex,rngSource &gen) {
    type=mtype;
    index=mindex;
    kfloat temp[3];
    for (int i=0;i<D;++i) {
      v[i]=0;
      f[i]=0;
      temp[i]=2.*PI*gen.rFloat64();
      a[i]=0;
      o[i]=0;
      t[i]=0;
    }
    a[0]=1.;
    a[1]=0.;
    a[2]=0.;
    vmath::rotate(a,temp);
    r[0]=x;
    r[1]=y;
    r[2]=z;
    for (int i=0;i<D;++i) v_[i]=v[i];
    for (int i=0;i<D;++i) o_[i]=o[i];
  }
  
  void write(kofstream &file) {
    file << index << " " << r[0] << " " << r[1] << " " << r[2] << "\n";
  }
  
  void write_o(kofstream &file) {
    file << r[0] << " " << r[1] << " " << r[2] << "\n";
  }
  
  void backup(std::ofstream &file) {
    file.write((char*)r,D*sizeof(kfloat));
    file.write((char*)v,D*sizeof(kfloat));
    file.write((char*)v_,D*sizeof(kfloat));
    file.write((char*)a,D*sizeof(kfloat));
    file.write((char*)o,D*sizeof(kfloat));
    file.write((char*)o_,D*sizeof(kfloat));
    file.write((char*)&index,sizeof(int));
    file.write((char*)&rank,sizeof(int));
    file.write((char*)&type,sizeof(short));
  }
  
  void restore(std::ifstream &file) {
    file.read((char*)r,D*sizeof(kfloat));
    file.read((char*)v,D*sizeof(kfloat));
    file.read((char*)v_,D*sizeof(kfloat));
    file.read((char*)a,D*sizeof(kfloat));
    file.read((char*)o,D*sizeof(kfloat));
    file.read((char*)o_,D*sizeof(kfloat));
    file.read((char*)&index,sizeof(int));
    file.read((char*)&rank,sizeof(int));
    file.read((char*)&type,sizeof(short));
  }
  
  void step() {
    particle_type *ptype=&parameters::ptype_map[type];
#ifdef TORQUE_TEST
    if (type==1) {
      o[0]=0.1;
      o_[0]=0.1;
      std::cout << t[0] << "\n";
      for (int i=0;i<D;++i) f[i]=t[i]=0;
      return;
    }
#endif
#ifdef FORCE_TEST
    if (type==1) {
      v[0]=0;
      v_[0]=0;
      o[0]=0;
      o_[0]=0;
      std::cout << f[0] << " " << tot_velocity << "\n";
      for (int i=0;i<D;++i) f[i]=t[i]=0;
      return;
    }
    else {
      f[0]+=1e-4;
    }
#endif
#ifdef DETERMINE_VISCOSITY
    if (r[0]<(kfloat)parameters::box_x[0]/2) f[1]+=parameters::pforce; else f[1]-=parameters::pforce;
#endif
    for (int i=0;i<D;++i) {
      v[i]+=f[i]*parameters::delta_t/ptype->mass;
      v_[i]=v[i]+PREDICTOR_LAMBDA*f[i]*parameters::delta_t/ptype->mass;
      r[i]+=v[i]*parameters::delta_t;
      vmath::pbc(r);
      f[i]=0;
      o[i]+=t[i]*parameters::delta_t*ptype->inverse_moment[i];
      o_[i]=o[i]+PREDICTOR_LAMBDA*t[i]*parameters::delta_t*ptype->inverse_moment[i];
      t[i]=0;
    }
    if (ptype->activity) {
      kfloat temp[D];
      for (int i=0;i<D;++i) temp[i]=o[i]*parameters::delta_t;
      vmath::rotate(a,temp);
    }
  }
  
  kfloat temperature() {
    return vmath::dot(v,v)*parameters::ptype_map[type].mass/(3.*parameters::kT);
  }
  
  kfloat temperature_a() {
    return vmath::dot(o,o)/(parameters::ptype_map[type].inverse_moment[0]*3.*parameters::kT);
  }
  
  bool overlaps(particle &other) {
    return vmath::distance_squared(r,other.r)<1.;
  }
  
  bool interact(particle &other,rngSource &gen,kfloat reject_limit=10.) {
    kfloat magnitude_squared=vmath::distance_squared(r,other.r);
    if (magnitude_squared>parameters::cutoff_squared) {
      if (magnitude_squared>reject_limit) return true;
      return false;
    }

    kfloat rij[D];
    vmath::set_sub_pbc(rij,r,other.r);
    kfloat rij_magnitude=sqrt(magnitude_squared);
    kfloat unit[D];
    vmath::set_div(unit,rij,rij_magnitude);
    kfloat wf=(*weight_function)(rij_magnitude);
    kfloat wf2=wf*wf;
    interaction_type *itype=&parameters::itype_map[type][other.type];
    kfloat vij[D];
    vmath::set_sub(vij,v_,other.v_);
    kfloat v_r=vmath::dot(vij,unit);
    kfloat force_ij[D]; vmath::zero(force_ij);
    
    //T & C
    kfloat gamma_s_wf2=itype->gamma_s*wf2;
    vmath::add_mul(force_ij,unit,(-itype->gamma_c*wf2+gamma_s_wf2)*v_r+(*itype->force_t)(rij_magnitude,itype->a,itype->b,itype->c));
    vmath::add_mul(force_ij,vij,-gamma_s_wf2);
    
    //R
    kfloat force_r[D];
    kfloat element_o[D];
    vmath::set_mul(element_o,o_,itype->lambda);
    vmath::add_mul(element_o,other.o_,1.-itype->lambda);
    vmath::set_cross(force_r,rij,element_o);
    vmath::add_mul(force_ij,force_r,-gamma_s_wf2);
    
    //B
    wf/=sqrt(parameters::delta_t);
    kfloat W[D][D];
    vmath::randomize_matrix(W,gen);
    kfloat anti_factor=wf*sqrt2*itype->sigma_s;
    kfloat W_A_01=(W[0][1]-W[1][0])/2.;
    kfloat W_A_02=(W[0][2]-W[2][0])/2.;
    kfloat W_A_12=(W[1][2]-W[2][1])/2.;
    force_ij[0]+=anti_factor*(W_A_01*unit[1]+W_A_02*unit[2]);
    force_ij[1]+=anti_factor*(-W_A_01*unit[0]+W_A_12*unit[2]);
    force_ij[2]+=anti_factor*(-W_A_02*unit[0]-W_A_12*unit[1]);
    vmath::add_mul(force_ij,unit,wf*sqrt1d3*itype->sigma_c*(W[0][0]+W[1][1]+W[2][2]));
    
    vmath::add(f,force_ij);
    vmath::sub(other.f,force_ij);
    
    kfloat torque[D];
    vmath::set_cross(torque,force_ij,rij);
    vmath::add_mul(t,torque,itype->lambda);
    vmath::add_mul(other.t,torque,1.-itype->lambda);
    
    return false;
  }
};

bool compare_particle(const particle &i,const particle &j) {
  return i.r[0]<j.r[0];
}

class neighbor;

class run_variable {
public:
  std::list<kofstream*> open_file_list;
  std::vector<particle> particle_array;
  unsigned int *particle_index;
  unsigned int my_particles;
  unsigned int visible_particles;
  unsigned int neighbor_count;
  unsigned int search_point;
  kfloat my_bounds[2*D];
  neighbor *neighbors;
  std::vector< std::vector<int> > interaction_list;
  unsigned long long output_location;
  kofstream output_file;
  std::string backup_file_name;
  unsigned long long elapsed_steps;
#ifdef DETERMINE_VISCOSITY
  std::string backup_viscosity_name;
  kfloat v_division;
  kofstream viscosity_file;
  unsigned int v_bins;
  unsigned long long *v_count;
  kfloat *v_sum;
#endif
  std::map<int,int> rank_to_neighbor;
  char *buffer;
  std::vector<unsigned int> ***particle_box;
  
  run_variable() {
    buffer=NULL;
  }
  
  bool particle_is_mine(particle &p) {
    return (p.r[0]>=my_bounds[0] && p.r[0]<my_bounds[0+D] && p.r[1]>=my_bounds[1] && p.r[1]<my_bounds[1+D] && p.r[2]>=my_bounds[2] && p.r[2]<my_bounds[2+D]);
  }
  
  void init_particle_index() {
    for (unsigned int i=0;i<my_particles;++i) {
      particle_index[particle_array[i].index]=i;
    }
  }
  
  void add_particle(particle &p) {
    p.rank=MPI::rank;
    my_particles++;
    visible_particles++;
    particle_array.push_back(p);
  }
  
#ifdef DETERMINE_VISCOSITY
  void viscosity_print() {
    for (unsigned int i=0;i<v_bins;++i) {
      if (v_count[i]>0) viscosity_file << v_division*((kfloat)i+0.5) << " " << v_sum[i]/v_count[i] << "\n";
    }
  }
#endif
  
  void init() {
    init_particle_index();
    output_location=output_file.tellp();
    particle_box=new std::vector<unsigned int> **[parameters::box_x[0]];
    for (int i=0;i<parameters::box_x[0];++i) {
      particle_box[i]=new std::vector<unsigned int> *[parameters::box_x[1]];
      for (int j=0;j<parameters::box_x[1];++j) {
        particle_box[i][j]=new std::vector<unsigned int> [parameters::box_x[2]];
        for (int k=0;k<parameters::box_x[2];++k) {
          particle_box[i][j][k]=std::vector<unsigned int>();
        }
      }
    }
    if (!parameters::continuation) elapsed_steps=0;
#ifdef DETERMINE_VISCOSITY
    v_division=0.05;
    v_bins=(int)(parameters::box_x[0]/v_division+0.5);
    v_count=new unsigned long long[v_bins];
    v_sum=new kfloat[v_bins];
    for (unsigned int i=0;i<v_bins;++i) {
      v_count[i]=0;
      v_sum[i]=0;
    }
    if (parameters::continuation) {
      std::ifstream restorefile(backup_viscosity_name.c_str());
      int old_version;
      restorefile.read((char*)&old_version,sizeof(int));
      restorefile.read((char*)v_count,sizeof(unsigned long long)*v_bins);
      restorefile.read((char*)v_sum,sizeof(kfloat)*v_bins);
      restorefile.close();
    }
#endif
  }
  
  void empty_the_box() {
    for (int i=0;i<parameters::box_x[0];++i) {
      for (int j=0;j<parameters::box_x[1];++j) {
        for (int k=0;k<parameters::box_x[2];++k) {
          particle_box[i][j][k].clear();
        }
      }
    }
  }
  
  void fill_the_box() {
    for (unsigned int i=0;i<visible_particles;++i) {
      particle_box[(int)particle_array[i].r[0]][(int)particle_array[i].r[1]][(int)particle_array[i].r[2]].push_back(particle_array[i].index);
    }
  }
};
run_variable run_var;

void kofstream::open(const char *filename,std::ios_base::openmode mode=std::ios_base::out) {
  this->std::ofstream::open(filename,mode);
  run_var.open_file_list.push_back(&(*this));
}

void kofstream::close() {
  this->std::ofstream::close();
  run_var.open_file_list.remove(&(*this));
}

class neighbor {
public:
  connect_type connection;
  int rank;
  kfloat bounds[2*D];
  std::vector<unsigned int> search_out_list;
  int search_in_size;
  std::vector<unsigned int> transfer_out_list;
  int transfer_in_size;
  std::set<unsigned int> force_send_list;
  std::set<unsigned int> force_receive_list;
  int force_send_size;
  int force_receive_size;
  int position_send_size;
  int position_receive_size;
  char *force_receive_position;
  char *position_receive_position;
  char *search_receive_position;
  char *transfer_receive_position;
  
  void search_init() {
    search_out_list.clear();
    transfer_out_list.clear();
    force_send_list.clear();
    force_receive_list.clear();
    force_send_size=0;
    force_receive_size=0;
    position_send_size=0;
    position_receive_size=0;
  }
  
  bool transfer_check(particle *p,unsigned int i,unsigned int j) {
    if (p->r[0]>=bounds[0] && p->r[0]<bounds[0+D] && p->r[1]>=bounds[1] && p->r[1]<bounds[1+D] && p->r[2]>=bounds[2] && p->r[2]<bounds[2+D]) {
      particle temp=(*p);
      (*p)=run_var.particle_array[j];
      run_var.particle_array[j]=temp;
      run_var.particle_index[p->index]=i;
      transfer_out_list.push_back(j);
      run_var.particle_array[j].rank=rank;
      return true;
    }
    return false;
  }
  
  void transfer_process(int &slot) {
    int transfer_in_particles;
    off_buffer(transfer_receive_position,transfer_in_particles);
    
    run_var.my_particles+=transfer_in_particles;
    if (run_var.particle_array.size()<run_var.my_particles) run_var.particle_array.resize(run_var.my_particles);
    
    for (int i=0;i<transfer_in_particles;++i) {
      run_var.particle_array[slot].transfer_from_buffer(transfer_receive_position);
      run_var.particle_index[run_var.particle_array[slot].index]=slot;
      ++slot;
    }
  }
  
  void transfer_receive_particles(char *&buff,MPI_Request *req) {
    transfer_receive_position=buff;
    MPI::irecv(buff,rank,transfer_in_size,tag_transfer_particles,req);
    buff+=transfer_in_size;
  }
  
  void transfer_send_particles(char *&buff) {
    char *start=buff;
    int transfer_out_particles=transfer_out_list.size();
    on_buffer(buff,transfer_out_particles);
    int transfer_out_size=transfer_out_particles*particle::transfer_size+sizeof(int);
    for (std::vector<unsigned int>::iterator p=transfer_out_list.begin();p!=transfer_out_list.end();p++) {
      run_var.particle_array[*p].transfer_to_buffer(buff);
    }
    MPI::isend(start,rank,transfer_out_size,tag_transfer_particles,MPI::request(req_out));
    //if (MPI::rank==0) std::cout << (connection&1)+((connection&2)>>1)+((connection&4)>>2) << " " << (kfloat)transfer_out_size/transfer_in_size << " T\n";
  }
  
  void search_check(particle *p,unsigned int i) {
    switch (connection) {
      case c0:
        break;
      case cx:
        if (
          (vmath::abs_distance(p->r,bounds,0)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,0)<parameters::neighbor_cutoff)
        ) search_out_list.push_back(i);
        break;
      case cy:
        if (
          (vmath::abs_distance(p->r,bounds,1)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,1)<parameters::neighbor_cutoff)
        ) search_out_list.push_back(i);
        break;
      case cxy:
        if (
          (vmath::abs_distance(p->r,bounds,0)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,0)<parameters::neighbor_cutoff) && 
          (vmath::abs_distance(p->r,bounds,1)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,1)<parameters::neighbor_cutoff)
        ) search_out_list.push_back(i);
        break;
      case cz:
        if (
          (vmath::abs_distance(p->r,bounds,2)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,2)<parameters::neighbor_cutoff)
        ) search_out_list.push_back(i);
        break;
      case cxz:
        if (
          (vmath::abs_distance(p->r,bounds,0)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,0)<parameters::neighbor_cutoff) && 
          (vmath::abs_distance(p->r,bounds,2)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,2)<parameters::neighbor_cutoff)
        ) search_out_list.push_back(i);
        break;
      case cyz:
        if (
          (vmath::abs_distance(p->r,bounds,2)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,2)<parameters::neighbor_cutoff) && 
          (vmath::abs_distance(p->r,bounds,1)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,1)<parameters::neighbor_cutoff)
        ) search_out_list.push_back(i);
        break;
      case cxyz:
        if (
          (vmath::abs_distance(p->r,bounds,0)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,0)<parameters::neighbor_cutoff) && 
          (vmath::abs_distance(p->r,bounds,1)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,1)<parameters::neighbor_cutoff) && 
          (vmath::abs_distance(p->r,bounds,2)<parameters::neighbor_cutoff || vmath::abs_distance(p->r,bounds+D,2)<parameters::neighbor_cutoff)
        ) search_out_list.push_back(i);
        break;
    }
  }
  
  void search_process(int &slot) {
    int search_in_particles;
    off_buffer(search_receive_position,search_in_particles);
    
    run_var.visible_particles+=search_in_particles;
    if (run_var.particle_array.size()<run_var.visible_particles) run_var.particle_array.resize(run_var.visible_particles);
    
    for (int i=0;i<search_in_particles;++i) {
      run_var.particle_array[slot].search_from_buffer(search_receive_position);
      run_var.particle_index[run_var.particle_array[slot].index]=slot;
      //if (MPI::rank==21) std::cout << MPI::rank << " " << run_var.particle_array[slot].r[0] << " " << run_var.particle_array[slot].r[1] << "\n";
      ++slot;
    }
  }
  
  void search_receive_particles(char *&buff,MPI_Request *req) {
    search_receive_position=buff;
    MPI::irecv(buff,rank,search_in_size,tag_search_particles,req);
    buff+=search_in_size;
  }
  
  void search_send_particles(char *&buff) {
    char *start=buff;
    int search_out_particles=search_out_list.size();
    on_buffer(buff,search_out_particles);
    int search_out_size=search_out_particles*particle::search_size+sizeof(int);
    for (std::vector<unsigned int>::iterator p=search_out_list.begin();p!=search_out_list.end();p++) {
      run_var.particle_array[*p].search_to_buffer(buff);
    }
    MPI::isend(start,rank,search_out_size,tag_search_particles,MPI::request(req_out));
    //if (MPI::rank==0) std::cout << (connection&1)+((connection&2)>>1)+((connection&4)>>2) << " " << (kfloat)search_out_size/search_in_size << " S\n";
  }
  
  void force_process() {
    int index;
    for (unsigned int i=0;i<force_receive_list.size();++i) {
      off_buffer(force_receive_position,index);
      run_var.particle_array[run_var.particle_index[index]].force_from_buffer(force_receive_position);
    }
  }
  
  void force_receive(char *&buff,MPI_Request *req) {
    force_receive_position=buff;
    MPI::irecv(buff,rank,force_receive_size,tag_force,req);
    buff+=force_receive_size;
  }
  
  void force_send(char *&buff) {
    char *start=buff;
    for (std::set<unsigned int>::iterator p=force_send_list.begin();p!=force_send_list.end();p++) {
      run_var.particle_array[*p].force_to_buffer(buff);
    }
    MPI::isend(start,rank,force_send_size,tag_force,MPI::request(req_out));
  }
  
  void position_process() {
    int index;
    for (unsigned int i=0;i<force_send_list.size();++i) {
      off_buffer(position_receive_position,index);
      run_var.particle_array[run_var.particle_index[index]].position_from_buffer(position_receive_position);
    }
  }
  
  void position_receive(char *&buff,MPI_Request *req) {
    position_receive_position=buff;
    MPI::irecv(buff,rank,position_receive_size,tag_position,req);
    buff+=position_receive_size;
  }
  
  void position_send(char *&buff) {
    char *start=buff;
    for (std::set<unsigned int>::iterator p=force_receive_list.begin();p!=force_receive_list.end();p++) {
      run_var.particle_array[*p].position_to_buffer(buff);
    }
    MPI::isend(start,rank,position_send_size,tag_position,MPI::request(req_out));
  }
};

void crash(int i) {
  std::cerr << "CRASH: Exit Code " << i << "!\n";
  std::cerr.flush(); std::cout.flush();
  for (std::list<kofstream*>::iterator p=run_var.open_file_list.begin(); p!=run_var.open_file_list.end(); p++) {
    (*p)->close();
  }
  MPI::abort(i);
  std::exit(i);
}

unsigned long long input_unit_convert(kfloat t,std::string &unit) {
  switch (unit[0]) {
    case 'u':
      return kround(t/parameters::delta_t);
      break;
    case 's':
      return kround(t);
      break;
    default:
      std::cerr << "Unknown unit type \"" << unit << "\"! (Aborting)\n";
      crash(1);
  }
  return 0;
}

void read_interaction_type(std::ifstream &file,interaction_type *itype,interaction_type *jtype) {
  std::string empty;
  int force_type=0;
  file >> empty >> itype->lambda >> itype->gamma_s >> itype->gamma_c >> itype->a >> itype->b >> force_type;
  itype->a*=parameters::kT;
  itype->sigma_s=sqrt(2.*parameters::kT*itype->gamma_s);
  itype->sigma_c=sqrt(2.*parameters::kT*itype->gamma_c);
  itype->c=exp(-itype->b);
  
  switch (force_type) {
    case 0:
      itype->force_t=&force_t_a;
      break;
    case 1:
      itype->force_t=&force_t_b;
      break;
    default:
      std::cerr << "Unknown interaction force type \"" << force_type << "\"! (Aborting)\n";
      crash(1);
  }
  
  jtype->lambda=1.-itype->lambda;
  jtype->gamma_s=itype->gamma_s;
  jtype->gamma_c=itype->gamma_c;
  jtype->sigma_s=itype->sigma_s;
  jtype->sigma_c=itype->sigma_c;
  jtype->a=itype->a;
  jtype->b=itype->b;
  jtype->c=itype->c;
  jtype->force_t=itype->force_t;
}

void read_particle_type(std::ifstream &file,particle_type *ptype) {
  file >> ptype->name >> ptype->type >> ptype->write;
  file >> ptype->radius >> ptype->mass;
  file >> ptype->activity;
  for (int i=0;i<D;++i) ptype->inverse_moment[i]=1./(2.*ptype->mass*ptype->radius*ptype->radius/5.);
}

void determine_boundaries() {
  kfloat x_per_rank=(kfloat)parameters::box_x[0]/parameters::x_ranks;
  kfloat y_per_rank=(kfloat)parameters::box_x[1]/parameters::y_ranks;
  kfloat z_per_rank=(kfloat)parameters::box_x[2]/parameters::z_ranks;
  int x_neighbors=std::min(parameters::x_ranks,3);
  int y_neighbors=std::min(parameters::y_ranks,3);
  int z_neighbors=std::min(parameters::z_ranks,3);
  run_var.neighbor_count=x_neighbors*y_neighbors*z_neighbors-1;
  run_var.neighbors=new neighbor[run_var.neighbor_count];
  for (unsigned int i=0;i<run_var.neighbor_count;++i) run_var.neighbors[i].rank=-1;
  
  int z=MPI::rank/(parameters::x_ranks*parameters::y_ranks);
  int y=(MPI::rank-z*parameters::x_ranks*parameters::y_ranks)/parameters::x_ranks;
  int x=MPI::rank-z*parameters::x_ranks*parameters::y_ranks-y*parameters::x_ranks;
  run_var.my_bounds[0]=x*x_per_rank;
  run_var.my_bounds[1]=y*y_per_rank;
  run_var.my_bounds[2]=z*z_per_rank;
  run_var.my_bounds[0+D]=(x+1)*x_per_rank;
  run_var.my_bounds[1+D]=(y+1)*y_per_rank;
  run_var.my_bounds[2+D]=(z+1)*z_per_rank;
  
  int index=0;
  for (int dx=-1;dx<2;++dx) for (int dy=-1;dy<2;++dy) for (int dz=-1;dz<2;++dz) {
    if (dx==0 && dy==0 && dz==0) continue;
    int ix=x+dx; if (ix==parameters::x_ranks) ix=0; else if (ix<0) ix=parameters::x_ranks-1;
    int iy=y+dy; if (iy==parameters::y_ranks) iy=0; else if (iy<0) iy=parameters::y_ranks-1;
    int iz=z+dz; if (iz==parameters::z_ranks) iz=0; else if (iz<0) iz=parameters::z_ranks-1;
    int this_rank=ix+iy*parameters::x_ranks+iz*parameters::x_ranks*parameters::y_ranks;
    if (this_rank==MPI::rank) continue;
    
    bool reject=false;
    for (unsigned int i=0;i<run_var.neighbor_count;++i) {
      if (run_var.neighbors[i].rank==this_rank) {
        reject=true;
        break;
      }
    }
    if (reject) continue;
    
    run_var.rank_to_neighbor[this_rank]=index;
    run_var.neighbors[index].rank=this_rank;
    run_var.neighbors[index].bounds[0]=ix*x_per_rank;
    run_var.neighbors[index].bounds[1]=iy*y_per_rank;
    run_var.neighbors[index].bounds[2]=iz*z_per_rank;
    run_var.neighbors[index].bounds[0+D]=(ix+1)*x_per_rank;
    run_var.neighbors[index].bounds[1+D]=(iy+1)*y_per_rank;
    run_var.neighbors[index].bounds[2+D]=(iz+1)*z_per_rank;
    
    bool connect_x=false;
    bool connect_y=false;
    bool connect_z=false;

    if ((vmath::distance(run_var.neighbors[index].bounds,run_var.my_bounds+D,0)==0 && vmath::distance(run_var.my_bounds+D,run_var.my_bounds,0)!=0) ||
      (vmath::distance(run_var.my_bounds+D,run_var.my_bounds,0)!=0 && vmath::distance(run_var.neighbors[index].bounds+D,run_var.my_bounds,0)==0)) connect_x=true;
    if ((vmath::distance(run_var.neighbors[index].bounds,run_var.my_bounds+D,1)==0 && vmath::distance(run_var.my_bounds+D,run_var.my_bounds,1)!=0) ||
      (vmath::distance(run_var.my_bounds+D,run_var.my_bounds,1)!=0 && vmath::distance(run_var.neighbors[index].bounds+D,run_var.my_bounds,1)==0)) connect_y=true;
    if ((vmath::distance(run_var.neighbors[index].bounds,run_var.my_bounds+D,2)==0 && vmath::distance(run_var.my_bounds+D,run_var.my_bounds,2)!=0) ||
      (vmath::distance(run_var.my_bounds+D,run_var.my_bounds,2)!=0 && vmath::distance(run_var.neighbors[index].bounds+D,run_var.my_bounds,2)==0)) connect_z=true;
    
    const kfloat F=3;
    const kfloat F2=3;
    const kfloat F3=3;
    const kfloat T1=0.1;
    const kfloat T2=0.1;
    const kfloat T3=0.1;
    if (connect_x && connect_y && connect_z) {
      run_var.neighbors[index].connection=cxyz;
      run_var.neighbors[index].search_in_size=(int)std::ceil(parameters::neighbor_cutoff*parameters::neighbor_cutoff*parameters::neighbor_cutoff*parameters::density*F3*particle::search_size);
      run_var.neighbors[index].transfer_in_size=(int)std::ceil(T3*parameters::density*F*particle::transfer_size);
      if (parameters::x_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
      if (parameters::y_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
      if (parameters::z_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
    }
    else if (connect_x && connect_y) {
      run_var.neighbors[index].connection=cxy;
      run_var.neighbors[index].search_in_size=(int)std::ceil(parameters::neighbor_cutoff*parameters::neighbor_cutoff*z_per_rank*parameters::density*F2*particle::search_size);
      run_var.neighbors[index].transfer_in_size=(int)std::ceil(T2*z_per_rank*parameters::density*F*particle::transfer_size);
      if (parameters::x_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
      if (parameters::y_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
    }
    else if (connect_y && connect_z) {
      run_var.neighbors[index].connection=cyz;
      run_var.neighbors[index].search_in_size=(int)std::ceil(parameters::neighbor_cutoff*parameters::neighbor_cutoff*x_per_rank*parameters::density*F2*particle::search_size);
      run_var.neighbors[index].transfer_in_size=(int)std::ceil(T2*x_per_rank*parameters::density*F*particle::transfer_size);
      if (parameters::y_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
      if (parameters::z_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
    }
    else if (connect_x && connect_z) {
      run_var.neighbors[index].connection=cxz;
      run_var.neighbors[index].search_in_size=(int)std::ceil(parameters::neighbor_cutoff*parameters::neighbor_cutoff*y_per_rank*parameters::density*F2*particle::search_size);
      run_var.neighbors[index].transfer_in_size=(int)std::ceil(T2*y_per_rank*parameters::density*F*particle::transfer_size);
      if (parameters::x_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
      if (parameters::z_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
    }
    else if (connect_x) {
      run_var.neighbors[index].connection=cx;
      run_var.neighbors[index].search_in_size=(int)std::ceil(parameters::neighbor_cutoff*y_per_rank*z_per_rank*parameters::density*F*particle::search_size);
      run_var.neighbors[index].transfer_in_size=(int)std::ceil(T1*y_per_rank*z_per_rank*parameters::density*F*particle::transfer_size);
      if (parameters::x_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
    }
    else if (connect_y) {
      run_var.neighbors[index].connection=cy;
      run_var.neighbors[index].search_in_size=(int)std::ceil(parameters::neighbor_cutoff*x_per_rank*z_per_rank*parameters::density*F*particle::search_size);
      run_var.neighbors[index].transfer_in_size=(int)std::ceil(T1*x_per_rank*z_per_rank*parameters::density*F*particle::transfer_size);
      if (parameters::y_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
    }
    else if (connect_z) {
      run_var.neighbors[index].connection=cz;
      run_var.neighbors[index].search_in_size=(int)std::ceil(parameters::neighbor_cutoff*x_per_rank*y_per_rank*parameters::density*F*particle::search_size);
      run_var.neighbors[index].transfer_in_size=(int)std::ceil(T1*x_per_rank*y_per_rank*parameters::density*F*particle::transfer_size);
      if (parameters::z_ranks==2) {
        run_var.neighbors[index].search_in_size*=2;
        run_var.neighbors[index].transfer_in_size*=2;
      }
    }
    else {
      run_var.neighbors[index].connection=c0;
      std::cerr << "Neighbor with no connection? (Aborting)\n";
      crash(1);
    }
    
    ++index;
  }
}

void write_structure(kofstream &file) {
  for (int i=0;i<parameters::ptype_count;++i) {
    particle_type *ptype=&parameters::ptype_map[i];
    if (ptype->write) {
      file << "atom " << ptype->first << ":" << ptype->last << " name " << ptype->name << " type " << ptype->type << " radius " << ptype->radius << "\n";
    }
  }
  file << "pbc " << parameters::box_x[0] << " " << parameters::box_x[1] << " " << parameters::box_x[2] << " 90 90 90\n";
  file << "t\n";
}

void read_parameters(std::ifstream &file,rngSource &gen) {
  std::string empty;
  std::string unit;
  kfloat temp;
  file >> empty >> parameters::cutoff;
  parameters::cutoff_squared=parameters::cutoff*parameters::cutoff;
  
  char s;
  file >> empty >> s;
  switch (s) {
    case '4':
      weight_function=&weight_function_0_125;
      break;
    case '3':
      weight_function=&weight_function_0_25;
      break;
    case '2':
      weight_function=&weight_function_0_50;
      break;
    case '1':
      weight_function=&weight_function_1_00;
      break;
    case '0':
      weight_function=&weight_function_2_00;
      break;
    default:
      std::cerr << "Unknown weight function type \"" << s << "\"! (Aborting)\n";
      crash(1);
  }
  
  file >> empty >> parameters::delta_t;
  file >> empty >> parameters::kT;
  file >> empty >> parameters::box_x[0] >> parameters::box_x[1] >> parameters::box_x[2];
  file >> empty >> parameters::density;
  parameters::volume=parameters::box_x[0]*parameters::box_x[1]*parameters::box_x[2];
  file >> empty >> temp >> unit;
  parameters::total_steps=input_unit_convert(temp,unit);
  file >> empty >> temp >> unit;
  parameters::write_time=input_unit_convert(temp,unit);
  file >> empty >> temp >> unit;
  parameters::backup_time=input_unit_convert(temp,unit);
  file >> empty >> temp >> unit;
  parameters::neighbor_search_time=input_unit_convert(temp,unit);
  file >> empty >> parameters::neighbor_cutoff;
  parameters::neighbor_cutoff_squared=parameters::neighbor_cutoff*parameters::neighbor_cutoff;
  parameters::neighbor_box_cutoff=(int)std::ceil(parameters::neighbor_cutoff);
  
  precomputed_interaction_cutoff=new kfloat[parameters::neighbor_search_time];
  for (unsigned int i=0;i<parameters::neighbor_search_time;++i) {
    precomputed_interaction_cutoff[i]=(parameters::neighbor_search_time-(kfloat)i)*(parameters::neighbor_cutoff-1.)/parameters::neighbor_search_time+1.;
    precomputed_interaction_cutoff[i]*=precomputed_interaction_cutoff[i];
  }
  
  if (MPI::rank==0) {
    std::cerr << " Write Interval = " << parameters::write_time*parameters::delta_t << "\n";
    std::cerr << "Backup Interval = " << parameters::backup_time*parameters::delta_t << "\n";
    std::cerr << "Search Interval = " << parameters::neighbor_search_time*parameters::delta_t << "\n";
    std::cerr << "       Run Time = " << parameters::total_steps*parameters::delta_t << "\n";
  }
  
  file >> empty >> parameters::ptype_count;
  parameters::ptype_map=new particle_type[parameters::ptype_count];
  parameters::itype_map=new interaction_type *[parameters::ptype_count];
  for (int i=0;i<parameters::ptype_count;++i) {
    parameters::itype_map[i]=new interaction_type[parameters::ptype_count];
    read_particle_type(file,&parameters::ptype_map[i]);
  }
  for (int i=0;i<parameters::ptype_count;++i) for (int j=i;j<parameters::ptype_count;++j) {
    read_interaction_type(file,&parameters::itype_map[i][j],&parameters::itype_map[j][i]);
  }
  
  determine_boundaries();
  
  kfloat volume_correction=0;
  int number_correction=0;
  std::vector<exact_number> exact_numbers;
  while (1) {
    file >> empty;
    if (file.fail() || file.eof()) {
      std::cerr << "Unspecified input format! (Aborting)\n";
      crash(1);
    }
    if (empty=="INIT_FORMAT") {
      run_var.my_particles=0;
      run_var.visible_particles=0;
      int format;
      file >> format;
      switch (format) {
        case 0:
        {
          particle temp;
          unsigned int index=0;
          unsigned int total=int(parameters::density*(parameters::volume-volume_correction))+number_correction;
          parameters::ptype_map[0].first=number_correction;
          parameters::ptype_map[0].last=total-1;
          parameters::particle_count=total;
          run_var.particle_index=new unsigned int[total];
          
          if (parameters::continuation) break;
          
          if (!parameters::continuation && MPI::rank==0) write_structure(run_var.output_file);
          
          for (unsigned int n=0;n<exact_numbers.size();++n) {
            for (int i=0;i<exact_numbers[n].number;++i) {
              temp=particle(gen.rFloat64()*parameters::box_x[0],gen.rFloat64()*parameters::box_x[1],gen.rFloat64()*parameters::box_x[2],exact_numbers[n].type,index,gen);
              if (run_var.particle_is_mine(temp)) run_var.add_particle(temp);
              if (!parameters::continuation && MPI::rank==0 && parameters::ptype_map[exact_numbers[n].type].write) temp.write_o(run_var.output_file);
              ++index;
            }
          }
          
          for (unsigned int i=number_correction;i<total;++i) {
            temp=particle(gen.rFloat64()*parameters::box_x[0],gen.rFloat64()*parameters::box_x[1],gen.rFloat64()*parameters::box_x[2],0,index,gen);
            if (run_var.particle_is_mine(temp)) run_var.add_particle(temp);
            if (!parameters::continuation && MPI::rank==0 && parameters::ptype_map[0].write) temp.write_o(run_var.output_file);
            ++index;
          }
          
          break;
        }
        default:
          std::cerr << "Unknown input format \"" << format << "\"! (Aborting)\n";
          crash(1);
      }
      break;
    }
    else if (empty=="EXACT_NUMBER") {
      exact_number n;
      file >> n.type >> n.number >> n.density;
      exact_numbers.push_back(n);
      volume_correction+=n.number/n.density;
      parameters::ptype_map[n.type].first=number_correction;
      parameters::ptype_map[n.type].last=number_correction+n.number-1;
      number_correction+=n.number;
    }
    else if (empty=="EXACT_FRACTION") {
      exact_number n;
      kfloat fraction;
      file >> n.type >> fraction >> n.density;
      n.number=fraction*parameters::volume*n.density;
      exact_numbers.push_back(n);
      volume_correction+=n.number/n.density;
      parameters::ptype_map[n.type].first=number_correction;
      parameters::ptype_map[n.type].last=number_correction+n.number-1;
      number_correction+=n.number;
    }
  }
  
#ifdef DETERMINE_VISCOSITY
  kfloat fff=0.05;
  switch (s) {
    case '4':
      fff=0.10;
      break;
    case '3':
      fff=0.05;
      break;
    case '2':
      fff=0.025;
      break;
    case '1':
      fff=0.0125;
      break;
    case '0':
      fff=0.00625;
      break;
    default:
      std::cerr << "Unknown weight function type \"" << s << "\"! (Aborting)\n";
      crash(1);
  }
  parameters::pforce=fff*(parameters::itype_map[0][0].gamma_c/4.5)*pow(parameters::cutoff,5.0);
  if (MPI::rank==0) std::cerr << "PFORCE = " << parameters::pforce << "\n";
#endif
}

void write_coordinates(kofstream &file) {
  file << "i\n";
  for (unsigned int i=0;i<run_var.my_particles;++i) {
    particle_type *ptype=&parameters::ptype_map[run_var.particle_array[i].type];
    if (ptype->write) run_var.particle_array[i].write(file);
  }
}

bool copyFile(std::string src,std::string dst) {
  std::ifstream in(src.c_str(),std::ios_base::binary);
  std::ofstream out(dst.c_str(),std::ios_base::binary);
  
  if (!in.is_open() || !out.is_open()) return false;

  out << in.rdbuf();
  
  return true;
}

void backup(unsigned long long t) {
  std::ifstream check(run_var.backup_file_name.c_str());
  if (check.is_open()) {
    check.close();
    if (!copyFile(run_var.backup_file_name,run_var.backup_file_name+"~")) {
      std::cerr << "Failure while backing up file \"" << run_var.backup_file_name << "\"! (Aborting)\n";
      crash(2);
    }
  }
  
  std::ofstream backupfile;
  backupfile.open(run_var.backup_file_name.c_str(),std::ios_base::trunc|std::ios_base::out);
  backupfile.write((char*)&VERSION,sizeof(int));
  t+=run_var.elapsed_steps;
  backupfile.write((char*)&t,sizeof(unsigned long long));
  backupfile.write((char*)&run_var.output_location,sizeof(unsigned long long));
  backupfile.write((char*)&run_var.my_particles,sizeof(int));
  backupfile.write((char*)&run_var.visible_particles,sizeof(int));
  for (unsigned int i=0;i<run_var.visible_particles;++i) {
    run_var.particle_array[i].backup(backupfile);
  }
  
  if (backupfile.bad()) {
    std::cerr << "Bad file detected during backup! (Aborting)\n";
    crash(2);
  }
  
  backupfile.close();
  
#ifdef DETERMINE_VISCOSITY
  check.open(run_var.backup_viscosity_name.c_str());
  if (check.is_open()) {
    check.close();
    if (!copyFile(run_var.backup_viscosity_name,run_var.backup_viscosity_name+"~")) {
      std::cerr << "Failure while backing up file \"" << run_var.backup_viscosity_name << "\"! (Aborting)\n";
      crash(2);
    }
  }
  
  backupfile.open(run_var.backup_viscosity_name.c_str(),std::ios_base::trunc|std::ios_base::out);
  backupfile.write((char*)&VERSION,sizeof(int));
  backupfile.write((char*)run_var.v_count,sizeof(unsigned long long)*run_var.v_bins);
  backupfile.write((char*)run_var.v_sum,sizeof(kfloat)*run_var.v_bins);
  
  if (backupfile.bad()) {
    std::cerr << "Bad file detected during backup! (Aborting)\n";
    crash(2);
  }
  
  backupfile.close();
#endif
}

void particle_exchange() {
  //determine particle destinations
  unsigned int end_index=run_var.my_particles-1;
  for (unsigned int i=0;i<run_var.my_particles;) {
    bool do_transfer=false;
    for (unsigned int j=0;j<run_var.neighbor_count;++j) {
      if (run_var.neighbors[j].transfer_check(&run_var.particle_array[i],i,end_index)) {
        end_index--;
        run_var.my_particles--;
        do_transfer=true;
        break;
      }
    }
    if (!do_transfer) {
      for (unsigned int j=0;j<run_var.neighbor_count;++j) {
        run_var.neighbors[j].search_check(&run_var.particle_array[i],i);
      }
      ++i;
    }
  }
  
  int out_buffer_size=0;
  int in_buffer_size=0;
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    out_buffer_size+=run_var.neighbors[i].transfer_out_list.size()*particle::transfer_size+sizeof(int);
    in_buffer_size+=run_var.neighbors[i].transfer_in_size;
  }
  
  //send & receive particle information
  int buffer_size=out_buffer_size+in_buffer_size;
  char *particle_buffer=new char[buffer_size];
  char *particle_out_buffer=particle_buffer;
  char *particle_in_buffer=particle_buffer+out_buffer_size;
  MPI_Request *req=new MPI_Request[run_var.neighbor_count];
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    run_var.neighbors[i].transfer_receive_particles(particle_in_buffer,&req[i]);
  }
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    run_var.neighbors[i].transfer_send_particles(particle_out_buffer);
  }
  
  int index;
  int slot=run_var.my_particles;
  run_var.search_point=slot;
  //put incoming particles into vector
  for (unsigned int i=0;i<run_var.neighbor_count;++i) {
    MPI::wait_any(run_var.neighbor_count,req,&index);
    run_var.neighbors[index].transfer_process(slot);
  }

  MPI::empty_all();
  delete [] req;
  delete [] particle_buffer;
}

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
      for (int i=0;i<run_var.my_particles;++i) {
        vmath::add(total_v,run_var.particle_array[i].v);
      }
      kfloat rtotal_v[D];
      MPI::allred(&total_v,&rtotal_v,D,MPI_KFLOAT,MPI_SUM);
      vmath::div(rtotal_v,parameters::particle_count);
      for (int i=0;i<run_var.my_particles;++i) {
        vmath::sub(run_var.particle_array[i].v,rtotal_v);
        vmath::sub(run_var.particle_array[i].v_,rtotal_v);
      }
      static unsigned long long t_start=(unsigned long long)(100./parameters::delta_t);
      if (t+run_var.elapsed_steps>t_start) for (int i=0;i<run_var.my_particles;++i) {
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
    tot_velocity=0;
    for (unsigned int i=0;i<run_var.my_particles;++i) {
      if (run_var.particle_array[i].type==0) {
        tot_velocity+=run_var.particle_array[i].v[0];
      }
    }
    tot_velocity/=(run_var.my_particles-1);
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
      for (int i=0;i<run_var.my_particles;++i) {
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
    std::ifstream restorefile(run_var.backup_file_name.c_str());
    int old_version;
    restorefile.read((char*)&old_version,sizeof(int));
    restorefile.read((char*)&run_var.elapsed_steps,sizeof(unsigned long long));
    restorefile.read((char*)&run_var.output_location,sizeof(unsigned long long));
    restorefile.read((char*)&run_var.my_particles,sizeof(int));
    restorefile.read((char*)&run_var.visible_particles,sizeof(int));
    run_var.particle_array.resize(run_var.visible_particles);
    for (unsigned int i=0;i<run_var.visible_particles;++i) {
      run_var.particle_array[i].restore(restorefile);
      run_var.particle_index[run_var.particle_array[i].index]=i;
    }
    restorefile.close();
    
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
