// dissipative particle dynamics algorithm
void run_algorithm(rngSource generator) {
  unsigned long long neighbor_search_t = 
                                       parameters::neighbor_search_time;
  for (unsigned long long t = 0; t < parameters::total_steps;) {
#ifdef NCUTOFF_TEST
    static bool once = true;
    static unsigned int num_s = 0;
    static kfloat **x = new kfloat*[run_var.my_particles];
    static kfloat d = 0;
#endif
    if (neighbor_search_t == parameters::neighbor_search_time) {
      // output time to log at each neighbor search
      if (MPI::rank == 0)
        std::cerr << (kfloat)t * parameters::delta_t << "\n";
      
      // neighbor search
      MPI::tsearch.start();
      neighbor_search();
      MPI::tsearch.stop();
      
      neighbor_search_t = 0;
#ifdef NCUTOFF_TEST
      ++num_s; // number of searches
      if (num_s > 4) {
        if (once) {
          once = false;
          for (unsigned int i = 0; i < run_var.my_particles; ++i) {
            x[i] = new kfloat[D];
            for (int j = 0; j < D; ++j) x[i][j] =
              run_var.particle_array[run_var.particle_index[i]].r[j];
          }
        }
        // store particle locations
        for (unsigned int i = 0; i < run_var.my_particles; ++i)
          for (int j=0;j<D;++j) x[i][j] = 
            run_var.particle_array[run_var.particle_index[i]].r[j];
      }
#endif
#ifdef DETERMINE_VISCOSITY
      // adjust velocity to be zero, on average
      kfloat total_v[D];
      vmath::zero(total_v);
      for (unsigned int i = 0; i < run_var.my_particles; ++i)
        vmath::add(total_v, run_var.particle_array[i].v);
      kfloat rtotal_v[D];
      MPI::allred(&total_v, &rtotal_v, D, MPI_KFLOAT, MPI_SUM);
      vmath::div(rtotal_v, parameters::particle_count);
      for (unsigned int i = 0; i < run_var.my_particles; ++i) {
        vmath::sub(run_var.particle_array[i].v,  rtotal_v);
        vmath::sub(run_var.particle_array[i].v_, rtotal_v);
      }
      
      // gather data for average velocity at each position
      static unsigned long long t_start =
            (unsigned long long)(100. / parameters::delta_t);
      if (t + run_var.elapsed_steps > t_start)
        for (unsigned int i = 0; i < run_var.my_particles; ++i) {
          int bin = (int)(run_var.particle_array[i].r[0] /
                          run_var.v_division);
          ++run_var.v_count[bin];
          run_var.v_sum[bin] += run_var.particle_array[i].v[1];
        }
#endif
    }
    ++neighbor_search_t;
    
    // communicate positions
    MPI::tposition.start();
    communicate_positions();
    MPI::tposition.stop();
    
    // calculate interactions
    MPI::tinteract.start();
    for (unsigned int i = 0; i < run_var.my_particles; ++i)
      for (unsigned int j = 0; j < run_var.interaction_list[i].size();) {
        if (run_var.particle_array[i].interact(run_var.particle_array[run_var.interaction_list[i][j]],
          generator, precomputed_interaction_cutoff[neighbor_search_t - 1])) {
          for (unsigned int k = j; k < run_var.interaction_list[i].size() - 1; ++k)
            run_var.interaction_list[i][k] = run_var.interaction_list[i][k + 1];
          run_var.interaction_list[i].resize(run_var.interaction_list[i].size() - 1);
        } else ++j;
      }
    MPI::tinteract.stop();
    
    // communicate forces
    MPI::tforce.start();
    communicate_forces();
    MPI::tforce.stop();
    
    // step particles forward one time step
    MPI::tstep.start();
    for (unsigned int i = 0; i < run_var.my_particles; ++i) {
      run_var.particle_array[i].step();
    }
    MPI::tstep.stop();
    
#ifdef FORCE_TEST
    // calculate average flow velocity
    total_velocity = 0;
    for (unsigned int i = 0; i < run_var.my_particles; ++i)
      if (run_var.particle_array[i].type == 0)
        total_velocity += run_var.particle_array[i].v[0];
    total_velocity /= (run_var.my_particles-1);
#endif
#ifdef NCUTOFF_TEST
    // max distance traveled by a particle between searches.
    // while this printout remains the same, the neighbor search cutoff
    // does not affect the accuracy of calculations
    if (num_s > 4)
      for (unsigned int i = 0; i < run_var.my_particles; ++i) {
        kfloat dd = vmath::distance_squared(x[i],
                    run_var.particle_array[run_var.particle_index[i]].r);
        if (dd > d) {
          d = dd;
          std::cerr << "MAX: " << sqrt(d) << "\n";
        }
      }
#endif
#ifdef TEMPERATURE_TEST
    // calculate and print average temperature
    if ((kfloat)t * parameters::delta_t > 0) {
      kfloat avg  = 0;
      kfloat avg2 = 0;
      unsigned int count = 0;
      for (unsigned int i = 0; i < run_var.my_particles; ++i)
        if (run_var.particle_array[i].type == 1) {
          ++count;
          avg  += run_var.particle_array[i].temperature();
          avg2 += run_var.particle_array[i].temperature_a();
        }
      std::cout << avg / count << " " << avg2 / count << "\n";
    }
#endif
    // write
    ++t;
    if (parameters::write_time &&
       (t + run_var.elapsed_steps) % parameters::write_time == 0) {
      write_coordinates(run_var.output_file);
      run_var.output_location=run_var.output_file.tellp();
    }
    
    // backup
    if (parameters::backup_time && t % parameters::backup_time == 0)
      backup(t);
  }
}
