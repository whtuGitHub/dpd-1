// neighbor searching for non-parallel runs
void neighbor_search_alone() {
  for (unsigned int i = 0; i < run_var.visible_particles; ++i)
    run_var.particle_index[run_var.particle_array[i].index] = i;
  run_var.empty_the_box();
  run_var.fill_the_box();
  particle temp;
  unsigned int fill_index = 0;
  // sort particles by location for cache optimization
  for (int i=0; i < parameters::box_x[0]; ++i)
    for (int j = 0;j < parameters::box_x[1]; ++j)
      for (int k = 0; k < parameters::box_x[2]; ++k) {
        std::vector<unsigned int> *box = &run_var.particle_box[i][j][k];
        for (int n = 0, max_n = box->size(); n < max_n; ++n) {
          temp = run_var.particle_array[fill_index];
          run_var.particle_array[fill_index] = 
              run_var.particle_array[run_var.particle_index[(*box)[n]]];
          run_var.particle_array[run_var.particle_index[(*box)[n]]] = temp;
          run_var.particle_index[temp.index] = 
              run_var.particle_index[(*box)[n]];
          run_var.particle_index[run_var.particle_array[fill_index].index] = fill_index;
          (*box)[n] = fill_index;
          ++fill_index;
        }
      }
  
  int ints=0;
  run_var.interaction_list.resize(run_var.my_particles);
  // turn on interactions for particles within the cutoff distance
  for (unsigned int i = 0; i < run_var.my_particles; ++i) {
    run_var.interaction_list[i].clear();
    run_var.particle_index[run_var.particle_array[i].index] = i;
    particle *p = &run_var.particle_array[i];
    for (int box_i = -parameters::neighbor_box_cutoff; box_i <= 
                      parameters::neighbor_box_cutoff; ++box_i)
      for (int box_j = -parameters::neighbor_box_cutoff; box_j <=
                        parameters::neighbor_box_cutoff; ++box_j)
        for (int box_k = -parameters::neighbor_box_cutoff; box_k <=
                          parameters::neighbor_box_cutoff; ++box_k) {
          int b_i = (int)p->r[0] + box_i;
          if (b_i < 0) b_i += parameters::box_x[0];
          else if (b_i >= parameters::box_x[0])
            b_i -= parameters::box_x[0];
          int b_j = (int)p->r[1] + box_j;
          if (b_j < 0) b_j += parameters::box_x[1];
          else if (b_j >= parameters::box_x[1])
            b_j -= parameters::box_x[1];
          int b_k = (int)p->r[2] + box_k;
          if (b_k < 0) b_k += parameters::box_x[2];
          else if (b_k >= parameters::box_x[2])
            b_k -= parameters::box_x[2];
          std::vector<unsigned int> *box = 
                                   &run_var.particle_box[b_i][b_j][b_k];
          for (int j = 0, max_j = box->size(); j < max_j; ++j) {
            particle *q = &run_var.particle_array[(*box)[j]];
            if (i < (*box)[j] && vmath::distance_squared(p->r,q->r) <
                                 parameters::neighbor_cutoff_squared) {
              ++ints;
              run_var.interaction_list[i].push_back((*box)[j]);
            }
          }
        }
  }
}

// neighbor searching algorithm
void neighbor_search() {
  run_var.interaction_list.clear();
  if (run_var.neighbor_count == 0) return neighbor_search_alone();
  
  for (unsigned int i = 0; i < run_var.neighbor_count; i++) {
    run_var.neighbors[i].search_init();
  }
  
  particle_exchange();
  
  //test particles for sending to neighbors
  for (unsigned int i = run_var.search_point; i < run_var.my_particles;
    ++i) for (unsigned int j = 0; j < run_var.neighbor_count; ++j)
    run_var.neighbors[j].search_check(&run_var.particle_array[i], i);
  
  // calculate buffer sizes
  int out_buffer_size = 0;
  int in_buffer_size = 0;
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i) {
    out_buffer_size += run_var.neighbors[i].search_out_list.size() *
                       particle::search_size + sizeof(int);
    in_buffer_size += run_var.neighbors[i].search_in_size;
  }
  int buffer_size = out_buffer_size + in_buffer_size;
  
  //send & receive particle information
  char *particle_buffer = new char[buffer_size];
  char *particle_out_buffer = particle_buffer;
  char *particle_in_buffer = particle_buffer + out_buffer_size;
  MPI_Request *req = new MPI_Request[run_var.neighbor_count];
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i)
    run_var.neighbors[i].search_receive_particles(particle_in_buffer, &req[i]);
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i)
    run_var.neighbors[i].search_send_particles(particle_out_buffer);
  
  int index;
  run_var.visible_particles = run_var.my_particles;
  int slot = run_var.my_particles;
  //put incoming particles into vector
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i) {
    MPI::wait_any(run_var.neighbor_count, req, &index);
    run_var.neighbors[index].search_process(slot);
  }
  
  run_var.empty_the_box();
  run_var.fill_the_box();
  particle temp;
  unsigned int fill_index = 0;
  unsigned int fill_index_n = run_var.my_particles;
  // sort particles for cache optimization, (mine, and visible)
  for (int i = 0; i < parameters::box_x[0]; ++i)
    for (int j = 0; j < parameters::box_x[1]; ++j)
      for (int k = 0; k < parameters::box_x[2]; ++k) {
        std::vector<unsigned int> *box = &run_var.particle_box[i][j][k];
        for (int n = 0, max_n = box->size(); n < max_n; ++n) {
          if (run_var.particle_array[run_var.particle_index[(*box)[n]]].rank == MPI::rank) {
            temp = run_var.particle_array[fill_index];
            run_var.particle_array[fill_index] =
              run_var.particle_array[run_var.particle_index[(*box)[n]]];
            run_var.particle_array[run_var.particle_index[(*box)[n]]] =
              temp;
            run_var.particle_index[temp.index] =
              run_var.particle_index[(*box)[n]];
            run_var.particle_index[run_var.particle_array[fill_index].index] = fill_index;
            (*box)[n] = fill_index;
            ++fill_index;
          }
          else {
            temp = run_var.particle_array[fill_index_n];
            run_var.particle_array[fill_index_n] =
              run_var.particle_array[run_var.particle_index[(*box)[n]]];
            run_var.particle_array[run_var.particle_index[(*box)[n]]] =
              temp;
            run_var.particle_index[temp.index] =
              run_var.particle_index[(*box)[n]];
            run_var.particle_index[run_var.particle_array[fill_index_n].index] = fill_index_n;
            (*box)[n] = fill_index_n;
            ++fill_index_n;
          }
        }
      }
  
  int ints = 0;
  run_var.interaction_list.resize(run_var.my_particles);
  // turn on interactions for particles within cutoff range
  for (unsigned int i = 0; i < run_var.my_particles; ++i) {
    run_var.interaction_list[i].clear();
    particle *p = &run_var.particle_array[i];
    for (int box_i = -parameters::neighbor_box_cutoff; box_i <=
                      parameters::neighbor_box_cutoff; ++box_i)
      for (int box_j = -parameters::neighbor_box_cutoff; box_j <=
                        parameters::neighbor_box_cutoff; ++box_j)
        for (int box_k = -parameters::neighbor_box_cutoff; box_k <=
                          parameters::neighbor_box_cutoff; ++box_k) {
          int b_i = (int)p->r[0] + box_i;
          if (b_i < 0) b_i += parameters::box_x[0];
          else if (b_i >= parameters::box_x[0])
            b_i -= parameters::box_x[0];
          int b_j = (int)p->r[1] + box_j;
          if (b_j < 0) b_j += parameters::box_x[1];
          else if (b_j >= parameters::box_x[1])
            b_j -= parameters::box_x[1];
          int b_k = (int)p->r[2] + box_k;
          if (b_k < 0) b_k += parameters::box_x[2];
          else if (b_k >= parameters::box_x[2])
            b_k -= parameters::box_x[2];
          std::vector<unsigned int> *box =
                                   &run_var.particle_box[b_i][b_j][b_k];
          
          for (unsigned int j = 0, max_j = box->size(); j < max_j; ++j) {
            unsigned int qi = (*box)[j];
            particle *q = &run_var.particle_array[qi];
            if (i < qi && vmath::distance_squared(p->r, q->r) <
                          parameters::neighbor_cutoff_squared) {
              if (q->rank == MPI::rank) {
                ++ints;
                run_var.interaction_list[i].push_back(qi);
              }
              else {
                neighbor *n =
                  &run_var.neighbors[run_var.rank_to_neighbor[q->rank]];
                if (((p->rank < q->rank) &&
                     (p->index + q->index) % 2 == 0) || 
                    ((p->rank > q->rank) &&
                     (p->index + q->index) % 2 == 1)) {
                  ++ints;
                  run_var.interaction_list[i].push_back(qi);
                  if (n->force_send_list.find(qi) ==
                      n->force_send_list.end()) {
                    n->force_send_list.insert(qi);
                    n->force_send_size += particle::force_send_size;
                    n->position_receive_size +=
                                          particle::position_send_size;
                  }
                }
                else {
                  if (n->force_receive_list.find(i) ==
                      n->force_receive_list.end()) {
                    n->force_receive_list.insert(i);
                    n->force_receive_size += particle::force_send_size;
                    n->position_send_size +=
                                          particle::position_send_size;
                  }
                }
              }
            }
          }
        }
  }
  
  // cleare mpi buffers and requests
  MPI::empty_all();
  delete [] req;
  delete [] particle_buffer;
}
