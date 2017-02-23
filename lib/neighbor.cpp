// class storing data and communication info for each MPI neighbor
class neighbor {
public:
  connect_type connection;
  int rank;
  kfloat bounds[2 * D];
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
  
  // wipe and zero related fields
  void search_init() {
    search_out_list.clear();
    transfer_out_list.clear();
    force_send_list.clear();
    force_receive_list.clear();
    force_send_size = 0;
    force_receive_size = 0;
    position_send_size = 0;
    position_receive_size = 0;
  }
  
  // return true if the particle is transferred
  bool transfer_check(particle *p, unsigned int i, unsigned int j) {
    if (p->r[0] >= bounds[0] && p->r[0] < bounds[0 + D] &&
        p->r[1] >= bounds[1] && p->r[1] < bounds[1 + D] &&
        p->r[2] >= bounds[2] && p->r[2] < bounds[2 + D]) {
      // swap the particle out of my particle range
      particle temp = (*p);
      (*p) = run_var.particle_array[j];
      run_var.particle_array[j] = temp;
      run_var.particle_index[p->index] = i;
      transfer_out_list.push_back(j);
      run_var.particle_array[j].rank = rank;
      return true;
    }
    return false;
  }
  
  // merge incoming particles into particle array
  void transfer_process(int &slot) {
    int transfer_in_particles;
    off_buffer(transfer_receive_position, transfer_in_particles);
    
    run_var.my_particles += transfer_in_particles;
    if (run_var.particle_array.size() < run_var.my_particles)
      run_var.particle_array.resize(run_var.my_particles);
    
    for (int i = 0; i < transfer_in_particles; ++i) {
      run_var.particle_array[slot].transfer_from_buffer(transfer_receive_position);
      run_var.particle_index[run_var.particle_array[slot].index] = slot;
      ++slot;
    }
  }
  
  // open irecv for particle transfer info
  void transfer_receive_particles(char *&buff, MPI_Request *req) {
    transfer_receive_position = buff;
    MPI::irecv(buff, rank, transfer_in_size, tag_transfer_particles,
                                                                   req);
    buff += transfer_in_size;
  }
  
  // open isend for particle transfer info
  void transfer_send_particles(char *&buff) {
    char *start = buff;
    int transfer_out_particles = transfer_out_list.size();
    on_buffer(buff, transfer_out_particles);
    int transfer_out_size = transfer_out_particles *
                            particle::transfer_size + sizeof(int);
    for (std::vector<unsigned int>::iterator p =
         transfer_out_list.begin(); p != transfer_out_list.end(); p++) {
      run_var.particle_array[*p].transfer_to_buffer(buff);
    }
    MPI::isend(start, rank, transfer_out_size, tag_transfer_particles,
                                                 MPI::request(req_out));
  }
  
  // apply connection-relevant bounds for neighbor search conditions
  void search_check(particle *p, unsigned int i) {
    switch (connection) {
      case c0:
        break;
      case cx:
        if (vmath::abs_distance(p->r, bounds, 0) <
            parameters::neighbor_cutoff ||
            vmath::abs_distance(p->r, bounds + D, 0) <
            parameters::neighbor_cutoff) search_out_list.push_back(i);
        break;
      case cy:
        if (vmath::abs_distance(p->r, bounds, 1) <
            parameters::neighbor_cutoff ||
            vmath::abs_distance(p->r, bounds + D, 1) <
            parameters::neighbor_cutoff) search_out_list.push_back(i);
        break;
      case cxy:
        if ((vmath::abs_distance(p->r, bounds, 0) <
             parameters::neighbor_cutoff ||
             vmath::abs_distance(p->r, bounds + D, 0) <
             parameters::neighbor_cutoff) && 
            (vmath::abs_distance(p->r, bounds, 1) <
             parameters::neighbor_cutoff ||
             vmath::abs_distance(p->r, bounds + D, 1) <
             parameters::neighbor_cutoff)) search_out_list.push_back(i);
        break;
      case cz:
        if (vmath::abs_distance(p->r, bounds, 2) <
            parameters::neighbor_cutoff ||
            vmath::abs_distance(p->r, bounds + D, 2) <
            parameters::neighbor_cutoff) search_out_list.push_back(i);
        break;
      case cxz:
        if ((vmath::abs_distance(p->r, bounds, 0) <
             parameters::neighbor_cutoff ||
             vmath::abs_distance(p->r, bounds + D, 0) <
             parameters::neighbor_cutoff) && 
            (vmath::abs_distance(p->r, bounds, 2) <
             parameters::neighbor_cutoff ||
             vmath::abs_distance(p->r, bounds + D, 2) <
             parameters::neighbor_cutoff)) search_out_list.push_back(i);
        break;
      case cyz:
        if ((vmath::abs_distance(p->r, bounds, 2) <
             parameters::neighbor_cutoff ||
             vmath::abs_distance(p->r, bounds + D, 2) <
             parameters::neighbor_cutoff) && 
            (vmath::abs_distance(p->r, bounds, 1) <
             parameters::neighbor_cutoff ||
             vmath::abs_distance(p->r, bounds + D, 1) <
             parameters::neighbor_cutoff)) search_out_list.push_back(i);
        break;
      case cxyz:
        if ((vmath::abs_distance(p->r, bounds, 0) <
             parameters::neighbor_cutoff ||
             vmath::abs_distance(p->r, bounds + D, 0) <
             parameters::neighbor_cutoff) && 
            (vmath::abs_distance(p->r, bounds, 1) <
             parameters::neighbor_cutoff ||
             vmath::abs_distance(p->r, bounds + D, 1) <
             parameters::neighbor_cutoff) && 
            (vmath::abs_distance(p->r, bounds, 2) <
             parameters::neighbor_cutoff ||
             vmath::abs_distance(p->r, bounds + D, 2) <
             parameters::neighbor_cutoff)) search_out_list.push_back(i);
        break;
    }
  }
  
  // merge search particles into visible particle array
  void search_process(int &slot) {
    int search_in_particles;
    off_buffer(search_receive_position, search_in_particles);
    
    run_var.visible_particles += search_in_particles;
    if (run_var.particle_array.size() < run_var.visible_particles)
      run_var.particle_array.resize(run_var.visible_particles);
    
    for (int i = 0; i < search_in_particles; ++i) {
      run_var.particle_array[slot].search_from_buffer(search_receive_position);
      run_var.particle_index[run_var.particle_array[slot].index] = slot;
      ++slot;
    }
  }
  
  // prepare irecv for particle search info
  void search_receive_particles(char *&buff, MPI_Request *req) {
    search_receive_position = buff;
    MPI::irecv(buff, rank, search_in_size, tag_search_particles, req);
    buff += search_in_size;
  }
  
  // prepare isend for particle search info
  void search_send_particles(char *&buff) {
    char *start = buff;
    int search_out_particles = search_out_list.size();
    on_buffer(buff, search_out_particles);
    int search_out_size = search_out_particles *
                          particle::search_size + sizeof(int);
    for (std::vector<unsigned int>::iterator p =
             search_out_list.begin(); p != search_out_list.end(); p++) {
      run_var.particle_array[*p].search_to_buffer(buff);
    }
    MPI::isend(start, rank, search_out_size, tag_search_particles,
                                                 MPI::request(req_out));
  }
  
  // merge incoming force info into particle array
  void force_process() {
    int index;
    for (unsigned int i = 0; i < force_receive_list.size(); ++i) {
      off_buffer(force_receive_position, index);
      run_var.particle_array[run_var.particle_index[index]].force_from_buffer(force_receive_position);
    }
  }
  
  // prepare irecv for force information
  void force_receive(char *&buff, MPI_Request *req) {
    force_receive_position = buff;
    MPI::irecv(buff, rank, force_receive_size, tag_force, req);
    buff += force_receive_size;
  }
  
  // prepare isend for force information
  void force_send(char *&buff) {
    char *start = buff;
    for (std::set<unsigned int>::iterator p = force_send_list.begin();
                                       p!= force_send_list.end(); p++) {
      run_var.particle_array[*p].force_to_buffer(buff);
    }
    MPI::isend(start, rank, force_send_size, tag_force,
                                             MPI::request(req_out));
  }
  
  // merge incoming position info into particle array
  void position_process() {
    int index;
    for (unsigned int i = 0; i < force_send_list.size(); ++i) {
      off_buffer(position_receive_position, index);
      run_var.particle_array[run_var.particle_index[index]].position_from_buffer(position_receive_position);
    }
  }
  
  // prepare irecv for position information
  void position_receive(char *&buff, MPI_Request *req) {
    position_receive_position = buff;
    MPI::irecv(buff, rank, position_receive_size, tag_position, req);
    buff += position_receive_size;
  }
  
  // prepare isend for position information
  void position_send(char *&buff) {
    char *start = buff;
    for (std::set<unsigned int>::iterator p =
       force_receive_list.begin(); p != force_receive_list.end(); p++) {
      run_var.particle_array[*p].position_to_buffer(buff);
    }
    MPI::isend(start, rank, position_send_size, tag_position,
                                                MPI::request(req_out));
  }
};
