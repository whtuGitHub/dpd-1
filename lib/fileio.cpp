// convert input file units
unsigned long long input_unit_convert(kfloat t, std::string &unit) {
  switch (unit[0]) {
    case 'u':
      return kround(t / parameters::delta_t);
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

// parse interaction type from input file
void read_interaction_type(std::ifstream &file,
                     interaction_type *itype, interaction_type *jtype) {
  std::string empty;
  int force_type = 0;
  file >> empty >> itype->lambda >> itype->gamma_s >> itype->gamma_c >>
                                    itype->a >> itype->b >> force_type;
  itype->a *= parameters::kT;
  itype->sigma_s = sqrt(2. * parameters::kT * itype->gamma_s);
  itype->sigma_c = sqrt(2. * parameters::kT * itype->gamma_c);
  itype->c = exp(-itype->b);
  
  // set force function
  switch (force_type) {
    case 0:
      itype->force_t = &force_t_a;
      break;
    case 1:
      itype->force_t = &force_t_b;
      break;
    default:
      std::cerr << "Unknown interaction force type \"" << force_type <<
                                                     "\"! (Aborting)\n";
      crash(1);
  }
  
  jtype->lambda = 1. - itype->lambda;
  jtype->gamma_s = itype->gamma_s;
  jtype->gamma_c = itype->gamma_c;
  jtype->sigma_s = itype->sigma_s;
  jtype->sigma_c = itype->sigma_c;
  jtype->a = itype->a;
  jtype->b = itype->b;
  jtype->c = itype->c;
  jtype->force_t = itype->force_t;
}

// parse particle type from input file
void read_particle_type(std::ifstream &file, particle_type *ptype) {
  file >> ptype->name >> ptype->type >> ptype->write;
  file >> ptype->radius >> ptype->mass;
  file >> ptype->activity;
  for (int i = 0; i < D; ++i) ptype->inverse_moment[i] = 1. /
    (2. * ptype->mass * ptype->radius * ptype->radius / 5.);
}

// write particle structure to file (for vmd)
void write_structure(kofstream &file) {
  for (int i = 0; i < parameters::ptype_count; ++i) {
    particle_type *ptype = &parameters::ptype_map[i];
    if (ptype->write) {
      file << "atom " << ptype->first << ":" << ptype->last <<
              " name " << ptype->name << " type " << ptype->type <<
              " radius " << ptype->radius << "\n";
    }
  }
  file << "pbc " << parameters::box_x[0] << " " <<
                    parameters::box_x[1] << " " <<
                    parameters::box_x[2] << " 90 90 90\n";
  file << "t\n";
}

// read parameters from input file
void read_parameters(std::ifstream &file, rngSource &gen) {
  std::string empty; // empty is just a dummy store for key words
  std::string unit;
  kfloat temp;
  file >> empty >> parameters::cutoff;
  parameters::cutoff_squared = parameters::cutoff * parameters::cutoff;
  
  char s;
  file >> empty >> s;
  switch (s) {
    case '4':
      weight_function = &weight_function_0_125;
      break;
    case '3':
      weight_function = &weight_function_0_25;
      break;
    case '2':
      weight_function = &weight_function_0_50;
      break;
    case '1':
      weight_function = &weight_function_1_00;
      break;
    case '0':
      weight_function = &weight_function_2_00;
      break;
    default:
      std::cerr << "Unknown weight function type \"" << s <<
                                                     "\"! (Aborting)\n";
      crash(1);
  }
  
  file >> empty >> parameters::delta_t;
  file >> empty >> parameters::kT;
  file >> empty >> parameters::box_x[0] >> parameters::box_x[1] >>
                                           parameters::box_x[2];
  file >> empty >> parameters::density;
  parameters::volume = parameters::box_x[0] * parameters::box_x[1] *
                                              parameters::box_x[2];
  file >> empty >> temp >> unit;
  parameters::total_steps = input_unit_convert(temp, unit);
  file >> empty >> temp >> unit;
  parameters::write_time = input_unit_convert(temp, unit);
  file >> empty >> temp >> unit;
  parameters::backup_time = input_unit_convert(temp, unit);
  file >> empty >> temp >> unit;
  parameters::neighbor_search_time = input_unit_convert(temp, unit);
  file >> empty >> parameters::neighbor_cutoff;
  parameters::neighbor_cutoff_squared = parameters::neighbor_cutoff *
                                        parameters::neighbor_cutoff;
  parameters::neighbor_box_cutoff =
    (int)std::ceil(parameters::neighbor_cutoff);
  
  precomputed_interaction_cutoff =
    new kfloat[parameters::neighbor_search_time];
  for (unsigned int i = 0; i < parameters::neighbor_search_time; ++i) {
    precomputed_interaction_cutoff[i] =
      (parameters::neighbor_search_time - (kfloat)i) *
      (parameters::neighbor_cutoff - 1.) /
       parameters::neighbor_search_time + 1.;
    precomputed_interaction_cutoff[i] *=
      precomputed_interaction_cutoff[i];
  }
  
  // only the head process writes log info
  if (MPI::rank == 0) {
    std::cerr << " Write Interval = " << parameters::write_time *
                                         parameters::delta_t << "\n";
    std::cerr << "Backup Interval = " << parameters::backup_time *
                                         parameters::delta_t << "\n";
    std::cerr << "Search Interval = " <<
      parameters::neighbor_search_time * parameters::delta_t << "\n";
    std::cerr << "       Run Time = " << parameters::total_steps *
                                         parameters::delta_t << "\n";
  }
  
  file >> empty >> parameters::ptype_count;
  parameters::ptype_map = new particle_type[parameters::ptype_count];
  parameters::itype_map = 
                     new interaction_type *[parameters::ptype_count];
  for (int i = 0; i < parameters::ptype_count; ++i) {
    parameters::itype_map[i] =
                     new interaction_type[parameters::ptype_count];
    read_particle_type(file, &parameters::ptype_map[i]);
  }
  for (int i = 0; i < parameters::ptype_count; ++i)
    for (int j = i; j < parameters::ptype_count; ++j) {
      read_interaction_type(file, &parameters::itype_map[i][j],
                                  &parameters::itype_map[j][i]);
  }
  
  determine_boundaries();
  
  kfloat volume_correction = 0;
  int number_correction = 0;
  std::vector<exact_number> exact_numbers;
  // parse system initialization format
  while (1) {
    file >> empty;
    if (file.fail() || file.eof()) {
      std::cerr << "Unspecified input format! (Aborting)\n";
      crash(1);
    }
    if (empty == "INIT_FORMAT") {
      run_var.my_particles = 0;
      run_var.visible_particles = 0;
      int format;
      file >> format;
      switch (format) {
        case 0:
        {
          particle temp;
          unsigned int index = 0;
          unsigned int total =
            int(parameters::density * (parameters::volume -
                            volume_correction)) + number_correction;
          parameters::ptype_map[0].first = number_correction;
          parameters::ptype_map[0].last = total - 1;
          parameters::particle_count = total;
          run_var.particle_index = new unsigned int[total];
          
          if (parameters::continuation) break;
          
          if (!parameters::continuation && MPI::rank == 0)
            write_structure(run_var.output_file);
          
          for (unsigned int n = 0; n < exact_numbers.size(); ++n) {
            for (int i = 0; i < exact_numbers[n].number; ++i) {
              temp = particle(gen.rFloat64() * parameters::box_x[0],
                              gen.rFloat64() * parameters::box_x[1],
                              gen.rFloat64() * parameters::box_x[2],
                              exact_numbers[n].type, index, gen);
              if (run_var.particle_is_mine(temp))
                run_var.add_particle(temp);
              if (!parameters::continuation && MPI::rank == 0 &&
                   parameters::ptype_map[exact_numbers[n].type].write)
                temp.write_o(run_var.output_file);
              ++index;
            }
          }
          
          for (unsigned int i = number_correction; i < total; ++i) {
            temp = particle(gen.rFloat64() * parameters::box_x[0],
                            gen.rFloat64() * parameters::box_x[1],
                            gen.rFloat64() * parameters::box_x[2],
                            0, index, gen);
            if (run_var.particle_is_mine(temp))
              run_var.add_particle(temp);
            if (!parameters::continuation && MPI::rank == 0 &&
                 parameters::ptype_map[0].write)
              temp.write_o(run_var.output_file);
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
    else if (empty == "EXACT_NUMBER") {
      exact_number n; // n of this type, then fill with solvent
      file >> n.type >> n.number >> n.density;
      exact_numbers.push_back(n);
      volume_correction += n.number/n.density;
      parameters::ptype_map[n.type].first = number_correction;
      parameters::ptype_map[n.type].last = 
                                         number_correction+n.number - 1;
      number_correction += n.number;
    }
    else if (empty=="EXACT_FRACTION") {
      exact_number n; // fraction of this type, then fill with solvent
      kfloat fraction;
      file >> n.type >> fraction >> n.density;
      n.number = fraction * parameters::volume * n.density;
      exact_numbers.push_back(n);
      volume_correction += n.number / n.density;
      parameters::ptype_map[n.type].first = number_correction;
      parameters::ptype_map[n.type].last =
                                       number_correction + n.number - 1;
      number_correction += n.number;
    }
  }
  
#ifdef DETERMINE_VISCOSITY
  kfloat fff=0.05;
  switch (s) {
    case '4':
      fff = 0.10;
      break;
    case '3':
      fff = 0.05;
      break;
    case '2':
      fff = 0.025;
      break;
    case '1':
      fff = 0.0125;
      break;
    case '0':
      fff = 0.00625;
      break;
    default:
      std::cerr << "Unknown weight function type \"" << s <<
                                                     "\"! (Aborting)\n";
      crash(1);
  }
  parameters::pforce = fff *
    (parameters::itype_map[0][0].gamma_c / 4.5) *
                                           pow(parameters::cutoff, 5.0);
  if (MPI::rank == 0) std::cerr << "PFORCE = " <<
                                   parameters::pforce << "\n";
#endif
}

// write particle positions to file
void write_coordinates(kofstream &file) {
  file << "i\n";
  for (unsigned int i = 0; i < run_var.my_particles; ++i) {
    particle_type *ptype =
                 &parameters::ptype_map[run_var.particle_array[i].type];
    if (ptype->write) run_var.particle_array[i].write(file);
  }
}

// copy the src file to dst
bool copyFile(std::string src, std::string dst) {
  std::ifstream in(src.c_str(), std::ios_base::binary);
  std::ofstream out(dst.c_str(), std::ios_base::binary);
  
  if (!in.is_open() || !out.is_open()) return false;

  out << in.rdbuf();
  
  return true;
}

// backup full system information
void backup(unsigned long long t) {
  // backup the previous backup, if it exists
  std::ifstream check(run_var.backup_file_name.c_str());
  if (check.is_open()) {
    check.close();
    if (!copyFile(run_var.backup_file_name,
                  run_var.backup_file_name + "~")) {
      std::cerr << "Failure while backing up file \"" <<
                   run_var.backup_file_name << "\"! (Aborting)\n";
      crash(2);
    }
  }
  
  std::ofstream backupfile;
  backupfile.open(run_var.backup_file_name.c_str(),
                  std::ios_base::trunc | std::ios_base::out);
  backupfile.write((char*)&VERSION, sizeof(int));
  t += run_var.elapsed_steps;
  backupfile.write((char*)&t, sizeof(unsigned long long));
  backupfile.write((char*)&run_var.output_location,
                           sizeof(unsigned long long));
  backupfile.write((char*)&run_var.my_particles, sizeof(int));
  backupfile.write((char*)&run_var.visible_particles, sizeof(int));
  for (unsigned int i = 0; i < run_var.visible_particles; ++i) {
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
    if (!copyFile(run_var.backup_viscosity_name,
                  run_var.backup_viscosity_name + "~")) {
      std::cerr << "Failure while backing up file \"" <<
                   run_var.backup_viscosity_name << "\"! (Aborting)\n";
      crash(2);
    }
  }
  
  backupfile.open(run_var.backup_viscosity_name.c_str(),
                  std::ios_base::trunc | std::ios_base::out);
  backupfile.write((char*)&VERSION, sizeof(int));
  backupfile.write((char*)run_var.v_count,
                          sizeof(unsigned long long) * run_var.v_bins);
  backupfile.write((char*)run_var.v_sum,
                          sizeof(kfloat) * run_var.v_bins);
  
  if (backupfile.bad()) {
    std::cerr << "Bad file detected during backup! (Aborting)\n";
    crash(2);
  }
  
  backupfile.close();
#endif
}
