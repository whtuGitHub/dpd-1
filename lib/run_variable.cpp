class neighbor;

// class for various run variables, for *this* MPI process
class run_variable {
public:
  std::list<kofstream*> open_file_list;
  std::vector<particle> particle_array;
  unsigned int *particle_index;
  unsigned int my_particles;
  unsigned int visible_particles;
  unsigned int neighbor_count;
  unsigned int search_point;
  kfloat my_bounds[2 * D];
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
  std::map<int, int> rank_to_neighbor;
  char *buffer;
  std::vector<unsigned int> ***particle_box;
  
  run_variable() {
    buffer=NULL;
  }
  
  // return true if the particle is within my bounds
  bool particle_is_mine(particle &p) {
    return (p.r[0] >= my_bounds[0] && p.r[0] < my_bounds[0 + D] &&
            p.r[1] >= my_bounds[1] && p.r[1] < my_bounds[1 + D] &&
            p.r[2] >= my_bounds[2] && p.r[2] < my_bounds[2 + D]);
  }
  
  // set the particle indices
  void init_particle_index() {
    for (unsigned int i = 0; i < my_particles; ++i)
      particle_index[particle_array[i].index] = i;
  }
  
  // add particle to my particle list (particles in my domain)
  void add_particle(particle &p) {
    p.rank = MPI::rank;
    my_particles++;
    visible_particles++;
    particle_array.push_back(p);
  }
  
#ifdef DETERMINE_VISCOSITY
  // print pertinent info to viscosity determination
  void viscosity_print() {
    for (unsigned int i = 0; i < v_bins; ++i) {
      if (v_count[i] > 0) viscosity_file << v_division *
              ((kfloat)i + 0.5) << " " << v_sum[i] / v_count[i] << "\n";
    }
  }
#endif
  
  // initialize this process's variables
  void init() {
    init_particle_index();
    output_location = output_file.tellp();
    particle_box = new std::vector<unsigned int> **[parameters::box_x[0]];
    for (int i = 0; i < parameters::box_x[0]; ++i) {
      particle_box[i] = new std::vector<unsigned int> *[parameters::box_x[1]];
      for (int j = 0;j < parameters::box_x[1]; ++j) {
        particle_box[i][j] = new std::vector<unsigned int> [parameters::box_x[2]];
        for (int k = 0;k < parameters::box_x[2]; ++k) {
          particle_box[i][j][k] = std::vector<unsigned int>();
        }
      }
    }
    if (!parameters::continuation) elapsed_steps = 0;
#ifdef DETERMINE_VISCOSITY
    v_division = 0.05;
    v_bins  = (int)(parameters::box_x[0] / v_division + 0.5);
    v_count = new unsigned long long[v_bins];
    v_sum = new kfloat[v_bins];
    for (unsigned int i = 0; i < v_bins; ++i) {
      v_count[i] = 0;
      v_sum[i]   = 0;
    }
    // continuation means we are restoring an old state
    if (parameters::continuation) {
      std::ifstream restore_file(backup_viscosity_name.c_str());
      int old_version;
      restore_file.read((char*)&old_version, sizeof(int));
      restore_file.read((char*)v_count,
                        sizeof(unsigned long long) * v_bins);
      restore_file.read((char*)v_sum, sizeof(kfloat) * v_bins);
      restore_file.close();
    }
#endif
  }
  
  // empty box of particle indices
  void empty_the_box() {
    for (int i = 0; i < parameters::box_x[0]; ++i)
      for (int j = 0; j < parameters::box_x[1]; ++j)
        for (int k = 0; k < parameters::box_x[2]; ++k)
          particle_box[i][j][k].clear();
  }
  
  // place particle indices into boxes
  void fill_the_box() {
    for (unsigned int i = 0; i < visible_particles; ++i) {
      particle *p = &particle_array[i];
      particle_box[(int)p->r[0]][(int)p->r[1]][(int)p->r[2]].push_back(p->index);
    }
  }
};
run_variable run_var;
