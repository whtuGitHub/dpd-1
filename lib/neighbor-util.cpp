// set up neighbor boundaries
void determine_boundaries() {
  kfloat x_per_rank =
    (kfloat)parameters::box_x[0] / parameters::x_ranks;
  kfloat y_per_rank =
    (kfloat)parameters::box_x[1] / parameters::y_ranks;
  kfloat z_per_rank =
    (kfloat)parameters::box_x[2] / parameters::z_ranks;
  int x_neighbors = std::min(parameters::x_ranks, 3);
  int y_neighbors = std::min(parameters::y_ranks, 3);
  int z_neighbors = std::min(parameters::z_ranks, 3);
  run_var.neighbor_count = x_neighbors * y_neighbors * z_neighbors -1;
  run_var.neighbors = new neighbor[run_var.neighbor_count];
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i)
    run_var.neighbors[i].rank = -1;
  
  int z =  MPI::rank /    (parameters::x_ranks * parameters::y_ranks);
  int y = (MPI::rank - z * parameters::x_ranks * parameters::y_ranks) /
                                                 parameters::x_ranks;
  int x =  MPI::rank - z * parameters::x_ranks * parameters::y_ranks -
                                             y * parameters::x_ranks;
  run_var.my_bounds[0] = x * x_per_rank;
  run_var.my_bounds[1] = y * y_per_rank;
  run_var.my_bounds[2] = z * z_per_rank;
  run_var.my_bounds[0 + D] = (x + 1) * x_per_rank;
  run_var.my_bounds[1 + D] = (y + 1) * y_per_rank;
  run_var.my_bounds[2 + D] = (z + 1) * z_per_rank;
  
  int index = 0;
  for (int dx = -1; dx < 2; ++dx) for (int dy = -1; dy < 2; ++dy)
    for (int dz = -1; dz < 2; ++dz) {
      if (dx == 0 && dy == 0 && dz == 0) continue;
      int ix = x + dx; if (ix == parameters::x_ranks) ix = 0;
      else if (ix < 0) ix = parameters::x_ranks - 1;
      int iy = y + dy; if (iy == parameters::y_ranks) iy = 0;
      else if (iy < 0) iy = parameters::y_ranks - 1;
      int iz = z + dz; if (iz == parameters::z_ranks) iz = 0;
      else if (iz < 0) iz = parameters::z_ranks - 1;
      int this_rank = ix + iy * parameters::x_ranks +
                      iz * parameters::x_ranks * parameters::y_ranks;
      // if this "neighbor" is myself, don't need to continue
      if (this_rank == MPI::rank) continue;
      
      bool reject = false;
      for (unsigned int i = 0; i < run_var.neighbor_count; ++i) {
        if (run_var.neighbors[i].rank == this_rank) {
          reject = true;
          break;
        }
      }
      if (reject) continue;
      
      run_var.rank_to_neighbor[this_rank] = index;
      run_var.neighbors[index].rank = this_rank;
      run_var.neighbors[index].bounds[0] = ix * x_per_rank;
      run_var.neighbors[index].bounds[1] = iy * y_per_rank;
      run_var.neighbors[index].bounds[2] = iz * z_per_rank;
      run_var.neighbors[index].bounds[0 + D] = (ix + 1) * x_per_rank;
      run_var.neighbors[index].bounds[1 + D] = (iy + 1) * y_per_rank;
      run_var.neighbors[index].bounds[2 + D] = (iz + 1) * z_per_rank;
      
      bool connect_x = false;
      bool connect_y = false;
      bool connect_z = false;

      if ((vmath::distance(run_var.neighbors[index].bounds, run_var.my_bounds + D, 0) == 0 &&
           vmath::distance(run_var.my_bounds + D, run_var.my_bounds, 0) != 0) ||
          (vmath::distance(run_var.my_bounds + D, run_var.my_bounds, 0) !=0 &&
           vmath::distance(run_var.neighbors[index].bounds + D, run_var.my_bounds, 0) == 0)) connect_x=true;
      if ((vmath::distance(run_var.neighbors[index].bounds, run_var.my_bounds + D, 1) == 0 &&
           vmath::distance(run_var.my_bounds + D, run_var.my_bounds, 1) != 0) ||
          (vmath::distance(run_var.my_bounds + D, run_var.my_bounds, 1) != 0 &&
           vmath::distance(run_var.neighbors[index].bounds + D, run_var.my_bounds, 1) == 0)) connect_y=true;
      if ((vmath::distance(run_var.neighbors[index].bounds, run_var.my_bounds + D, 2) == 0 &&
           vmath::distance(run_var.my_bounds + D, run_var.my_bounds, 2) != 0) ||
          (vmath::distance(run_var.my_bounds + D, run_var.my_bounds, 2) != 0 &&
           vmath::distance(run_var.neighbors[index].bounds + D, run_var.my_bounds, 2) == 0)) connect_z=true;
      
      const kfloat F  = 3;
      const kfloat F2 = 3;
      const kfloat F3 = 3;
      const kfloat T1 = 0.1;
      const kfloat T2 = 0.1;
      const kfloat T3 = 0.1;
      // set up connection information for neighbor searching
      if (connect_x && connect_y && connect_z) {
        run_var.neighbors[index].connection = cxyz;
        run_var.neighbors[index].search_in_size =
          (int)std::ceil(parameters::neighbor_cutoff *
                         parameters::neighbor_cutoff *
                         parameters::neighbor_cutoff *
                         parameters::density *
                         F3 * particle::search_size);
        run_var.neighbors[index].transfer_in_size = 
          (int)std::ceil(T3 * parameters::density *
                         F * particle::transfer_size);
        if (parameters::x_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *= 2;
        }
        if (parameters::y_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *= 2;
        }
        if (parameters::z_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *= 2;
        }
      }
      else if (connect_x && connect_y) {
        run_var.neighbors[index].connection = cxy;
        run_var.neighbors[index].search_in_size =
          (int)std::ceil(parameters::neighbor_cutoff * 
                         parameters::neighbor_cutoff *
                         z_per_rank * parameters::density *
                         F2 * particle::search_size);
        run_var.neighbors[index].transfer_in_size =
          (int)std::ceil(T2 * z_per_rank * parameters::density *
                         F * particle::transfer_size);
        if (parameters::x_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *= 2;
        }
        if (parameters::y_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *= 2;
        }
      }
      else if (connect_y && connect_z) {
        run_var.neighbors[index].connection = cyz;
        run_var.neighbors[index].search_in_size =
          (int)std::ceil(parameters::neighbor_cutoff *
                         parameters::neighbor_cutoff *
                         x_per_rank * parameters::density *
                         F2 * particle::search_size);
        run_var.neighbors[index].transfer_in_size =
          (int)std::ceil(T2 * x_per_rank * parameters::density *
                         F * particle::transfer_size);
        if (parameters::y_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *=2;
        }
        if (parameters::z_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *= 2;
        }
      }
      else if (connect_x && connect_z) {
        run_var.neighbors[index].connection = cxz;
        run_var.neighbors[index].search_in_size =
          (int)std::ceil(parameters::neighbor_cutoff *
                         parameters::neighbor_cutoff *
                         y_per_rank * parameters::density *
                         F2 * particle::search_size);
        run_var.neighbors[index].transfer_in_size =
          (int)std::ceil(T2 * y_per_rank * parameters::density *
                         F * particle::transfer_size);
        if (parameters::x_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *= 2;
        }
        if (parameters::z_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *= 2;
        }
      }
      else if (connect_x) {
        run_var.neighbors[index].connection = cx;
        run_var.neighbors[index].search_in_size =
          (int)std::ceil(parameters::neighbor_cutoff * y_per_rank *
                         z_per_rank * parameters::density *
                         F * particle::search_size);
        run_var.neighbors[index].transfer_in_size =
          (int)std::ceil(T1 * y_per_rank * z_per_rank *
                         parameters::density *
                         F * particle::transfer_size);
        if (parameters::x_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *= 2;
        }
      }
      else if (connect_y) {
        run_var.neighbors[index].connection = cy;
        run_var.neighbors[index].search_in_size =
          (int)std::ceil(parameters::neighbor_cutoff * x_per_rank *
                         z_per_rank * parameters::density *
                         F * particle::search_size);
        run_var.neighbors[index].transfer_in_size =
          (int)std::ceil(T1 * x_per_rank * z_per_rank *
                         parameters::density *
                         F * particle::transfer_size);
        if (parameters::y_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *= 2;
        }
      }
      else if (connect_z) {
        run_var.neighbors[index].connection = cz;
        run_var.neighbors[index].search_in_size =
          (int)std::ceil(parameters::neighbor_cutoff * x_per_rank *
                         y_per_rank * parameters::density *
                         F * particle::search_size);
        run_var.neighbors[index].transfer_in_size =
          (int)std::ceil(T1 * x_per_rank * y_per_rank *
                         parameters::density * 
                         F * particle::transfer_size);
        if (parameters::z_ranks == 2) {
          run_var.neighbors[index].search_in_size *= 2;
          run_var.neighbors[index].transfer_in_size *= 2;
        }
      }
      else {
        run_var.neighbors[index].connection = c0;
        std::cerr << "Neighbor with no connection? (Aborting)\n";
        crash(1);
      }
      
      ++index;
  }
}

// send and receive particles between neighbors
void particle_exchange() {
  //determine particle destinations
  unsigned int end_index = run_var.my_particles - 1;
  for (unsigned int i = 0; i < run_var.my_particles;) {
    bool do_transfer = false;
    for (unsigned int j = 0; j < run_var.neighbor_count; ++j)
      if (run_var.neighbors[j].transfer_check(&run_var.particle_array[i], i, end_index)) {
        end_index--;
        run_var.my_particles--;
        do_transfer = true;
        break;
      }
    if (!do_transfer) {
      for (unsigned int j = 0; j < run_var.neighbor_count; ++j)
        run_var.neighbors[j].search_check(&run_var.particle_array[i], i);
      ++i;
    }
  }
  
  // set up buffer sizes
  int out_buffer_size = 0;
  int in_buffer_size = 0;
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i) {
    out_buffer_size += run_var.neighbors[i].transfer_out_list.size() *
                       particle::transfer_size + sizeof(int);
    in_buffer_size  += run_var.neighbors[i].transfer_in_size;
  }
  
  //send & receive particle information
  int buffer_size = out_buffer_size + in_buffer_size;
  char *particle_buffer = new char[buffer_size];
  char *particle_out_buffer = particle_buffer;
  char *particle_in_buffer = particle_buffer + out_buffer_size;
  MPI_Request *req = new MPI_Request[run_var.neighbor_count];
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i) {
    run_var.neighbors[i].transfer_receive_particles(particle_in_buffer, &req[i]);
  }
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i) {
    run_var.neighbors[i].transfer_send_particles(particle_out_buffer);
  }
  
  int index;
  int slot = run_var.my_particles;
  run_var.search_point = slot;
  //put incoming particles into vector
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i) {
    MPI::wait_any(run_var.neighbor_count, req, &index);
    run_var.neighbors[index].transfer_process(slot);
  }

  // clear up mpi buffers and requests
  MPI::empty_all();
  delete [] req;
  delete [] particle_buffer;
}
