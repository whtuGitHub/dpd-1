
// dpd particle class
class particle {
public:
  short type;
  int index;
  int rank;
  // v/o_ is the predicted v/o
  kfloat r[D];
  kfloat v[D];
  kfloat v_[D];
  kfloat f[D];
  kfloat a[D];
  kfloat o[D];
  kfloat o_[D];
  kfloat t[D];
  
  // memory sizes for MPI communication
  static const int force_send_size = 3 * sizeof(kfloat) + sizeof(int);
  static const int position_send_size = 12 * sizeof(kfloat) +
                                             sizeof(int);
  static const int search_size = 3 * sizeof(r[0]) + sizeof(index) +
                                     sizeof(type) + sizeof(rank);
  static const int transfer_size = 18 * sizeof(r[0]) + sizeof(index) +
                                                       sizeof(type);
  
  // write partice to buffer
  void transfer_to_buffer(char *&buff) {
    on_buffer(buff, type);
    on_buffer(buff, index);
    vmath::to_buffer(buff, r);
    vmath::to_buffer(buff, v);
    vmath::to_buffer(buff, v_);
    vmath::to_buffer(buff, o);
    vmath::to_buffer(buff, o_);
    vmath::to_buffer(buff, a);
  }
  
  // read particle from buffer
  void transfer_from_buffer(char *&buff) {
    off_buffer(buff, type);
    off_buffer(buff, index);
    vmath::from_buffer(buff, r);
    vmath::from_buffer(buff, v);
    vmath::from_buffer(buff, v_);
    vmath::from_buffer(buff, o);
    vmath::from_buffer(buff, o_);
    vmath::from_buffer(buff, a);
    rank = MPI::rank;
  }
  
  // write search info to buffer
  void search_to_buffer(char *&buff) {
    on_buffer(buff, type);
    on_buffer(buff, index);
    on_buffer(buff, rank);
    vmath::to_buffer(buff, r);
  }
  
  // read search info from buffer
  void search_from_buffer(char *&buff) {
    off_buffer(buff, type);
    off_buffer(buff, index);
    off_buffer(buff, rank);
    vmath::from_buffer(buff, r);
  }
  
  // write force on this particle to buffer
  void force_to_buffer(char *&buff) {
    on_buffer(buff, index);
    vmath::to_buffer(buff, f);
    vmath::zero(f);
  }
  
  // read force on this particle from buffer
  void force_from_buffer(char *&buff) {
    kfloat temp[D];
    vmath::from_buffer(buff, temp);
    vmath::add(f, temp);
  }
  
  // write position to buffer
  void position_to_buffer(char *&buff) {
    on_buffer(buff, index);
    vmath::to_buffer(buff, r);
    vmath::to_buffer(buff, v_);
    vmath::to_buffer(buff, a);
    vmath::to_buffer(buff, o_);
  }
  
  // read position from buffer
  void position_from_buffer(char *&buff) {
    vmath::from_buffer(buff, r);
    vmath::from_buffer(buff, v_);
    vmath::from_buffer(buff, a);
    vmath::from_buffer(buff, o_);
  }
  
  // zero-intialized particle
  particle() {
    type  = 0;
    index = 0;
    for (int i = 0; i < D; ++i) {
      r[i] = 0;
      v[i] = 0;
      f[i] = 0;
      a[i] = 0;
      o[i] = 0;
      t[i] = 0;
    }
    for (int i = 0; i < D; ++i) v_[i] = v[i];
    for (int i = 0; i < D; ++i) o_[i] = o[i];
  }
  
  // particle with position and type / index given, angle randomized
  particle(kfloat x, kfloat y, kfloat z, int mtype, int mindex,
                                                    rngSource &gen) {
    type  = mtype;
    index = mindex;
    kfloat temp[3];
    for (int i = 0; i < D; ++i) {
      v[i] = 0;
      f[i] = 0;
      temp[i] = 2. * PI * gen.rFloat64();
      a[i] = 0;
      o[i] = 0;
      t[i] = 0;
    }
    a[0] = 1.; // facing angle is normalized
    a[1] = 0.;
    a[2] = 0.;
    vmath::rotate(a, temp);
    r[0] = x;
    r[1] = y;
    r[2] = z;
    for (int i = 0; i < D; ++i) v_[i] = v[i];
    for (int i = 0; i < D; ++i) o_[i] = o[i];
  }
  
  void write(kofstream &file) {
    file << index << " " << r[0] << " " << r[1] << " " << r[2] << "\n";
  }
  
  void write_o(kofstream &file) {
    file << r[0] << " " << r[1] << " " << r[2] << "\n";
  }
  
  void backup(std::ofstream &file) {
    file.write((char*)r, D * sizeof(kfloat));
    file.write((char*)v, D * sizeof(kfloat));
    file.write((char*)v_, D * sizeof(kfloat));
    file.write((char*)a, D * sizeof(kfloat));
    file.write((char*)o, D * sizeof(kfloat));
    file.write((char*)o_, D * sizeof(kfloat));
    file.write((char*)&index, sizeof(int));
    file.write((char*)&rank, sizeof(int));
    file.write((char*)&type, sizeof(short));
  }
  
  void restore(std::ifstream &file) {
    file.read((char*)r, D * sizeof(kfloat));
    file.read((char*)v, D * sizeof(kfloat));
    file.read((char*)v_, D * sizeof(kfloat));
    file.read((char*)a, D * sizeof(kfloat));
    file.read((char*)o, D * sizeof(kfloat));
    file.read((char*)o_, D * sizeof(kfloat));
    file.read((char*)&index, sizeof(int));
    file.read((char*)&rank, sizeof(int));
    file.read((char*)&type, sizeof(short));
  }
  
  // move the particle forward one time step
  void step() {
    particle_type *ptype = &parameters::ptype_map[type];
    // torque test requires particle (1) with fixed angular velocity
#ifdef TORQUE_TEST
    if (type == 1) {
      o[0]  = 0.1;
      o_[0] = 0.1;
      std::cout << t[0] << "\n"; // print torque
      for (int i = 0; i < D; ++i) f[i] = t[i] = 0;
      return;
    }
#endif
    // force test requires particle (1) with fixed velocity (zero)
#ifdef FORCE_TEST
    if (type == 1) {
      v[0]  = 0;
      v_[0] = 0;
      o[0]  = 0;
      o_[0] = 0;
      std::cout << f[0] << " " << total_velocity << "\n";
      for (int i = 0; i < D; ++i) f[i] = t[i] = 0;
      return;
    }
    else {
      f[0] += 1e-4; // creates fixed flow rate
    }
#endif
    // periodic poiseuille flow force
#ifdef DETERMINE_VISCOSITY
    if (r[0] < (kfloat)parameters::box_x[0] / 2)
      f[1] += parameters::pforce;
    else f[1] -= parameters::pforce;
#endif
    // Eqs. of motion
    for (int i = 0; i < D; ++i) {
      v[i] += f[i] * parameters::delta_t / ptype->mass;
      v_[i] = v[i] + PREDICTOR_LAMBDA * f[i] *
                     parameters::delta_t / ptype->mass;
      r[i] += v[i] * parameters::delta_t;
      vmath::pbc(r); // wrap particle position in pbc
      f[i] = 0;
      o[i] += t[i] * parameters::delta_t * ptype->inverse_moment[i];
      o_[i] = o[i] + PREDICTOR_LAMBDA * t[i] * 
                     parameters::delta_t * ptype->inverse_moment[i];
      t[i] = 0;
    }
    // the facing angle only matters for active particles
    if (ptype->activity) {
      kfloat temp[D];
      for (int i = 0; i < D; ++i) temp[i] = o[i] * parameters::delta_t;
      vmath::rotate(a, temp);
    }
  }
  
  // particle temperature (translation)
  kfloat temperature() {
    return vmath::dot(v, v) * parameters::ptype_map[type].mass /
                                                  (3. * parameters::kT);
  }
  
  // particle temperature (rotation)
  kfloat temperature_a() {
    return vmath::dot(o, o) /
      (parameters::ptype_map[type].inverse_moment[0] *
                                                   3. * parameters::kT);
  }
  
  // return true if the particles are overlapping
  bool overlaps(particle &other) {
    return vmath::distance_squared(r, other.r) < 1.;
  }
  
  // interact with other particle via dpd forces
  bool interact(particle &other, rngSource &gen, kfloat reject_limit=10.) {
    kfloat magnitude_squared = vmath::distance_squared(r, other.r);
    if (magnitude_squared > parameters::cutoff_squared) {
      if (magnitude_squared > reject_limit) return true;
      return false;
    }

    kfloat rij[D];
    vmath::set_sub_pbc(rij, r, other.r);
    kfloat rij_magnitude = sqrt(magnitude_squared);
    kfloat unit[D];
    vmath::set_div(unit, rij, rij_magnitude);
    kfloat wf  = (*weight_function)(rij_magnitude);
    kfloat wf2 = wf * wf;
    interaction_type *itype = &parameters::itype_map[type][other.type];
    kfloat vij[D];
    vmath::set_sub(vij, v_, other.v_);
    kfloat v_r = vmath::dot(vij, unit);
    kfloat force_ij[D]; vmath::zero(force_ij);
    
    // translational and conservative
    kfloat gamma_s_wf2 = itype->gamma_s * wf2;
    vmath::add_mul(force_ij, unit,
      (-itype->gamma_c * wf2 + gamma_s_wf2) * v_r +
      (*itype->force_t)(rij_magnitude, itype->a, itype->b, itype->c));
    vmath::add_mul(force_ij, vij, -gamma_s_wf2);
    
    // rotational
    kfloat force_r[D];
    kfloat element_o[D];
    vmath::set_mul(element_o, o_, itype->lambda);
    vmath::add_mul(element_o, other.o_, 1. - itype->lambda);
    vmath::set_cross(force_r, rij, element_o);
    vmath::add_mul(force_ij, force_r, -gamma_s_wf2);
    
    // brownian
    wf /= std::sqrt(parameters::delta_t);
    kfloat W[D][D];
    vmath::randomize_matrix(W, gen);
    kfloat anti_factor = wf * sqrt2 * itype->sigma_s;
    kfloat W_A_01 = (W[0][1] - W[1][0])/2.;
    kfloat W_A_02 = (W[0][2] - W[2][0])/2.;
    kfloat W_A_12 = (W[1][2] - W[2][1])/2.;
    force_ij[0] += anti_factor * (W_A_01 * unit[1] + W_A_02 * unit[2]);
    force_ij[1] += anti_factor * (-W_A_01 * unit[0] + W_A_12 * unit[2]);
    force_ij[2] += anti_factor * (-W_A_02 * unit[0] - W_A_12 * unit[1]);
    vmath::add_mul(force_ij, unit, wf * sqrt1d3 * itype->sigma_c *
                                        (W[0][0] + W[1][1] + W[2][2]));
    
    // apply force and torque on particles
    vmath::add(f, force_ij);
    vmath::sub(other.f, force_ij);
    
    kfloat torque[D];
    vmath::set_cross(torque, force_ij, rij);
    vmath::add_mul(t, torque, itype->lambda);
    vmath::add_mul(other.t, torque, 1. - itype->lambda);
    
    return false;
  }
};
