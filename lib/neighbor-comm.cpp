// send and receive force data
void communicate_forces() {
  if (run_var.neighbor_count == 0) return;
  MPI::empty(req_out);
  if (run_var.buffer) delete [] run_var.buffer;
  
  int total_send_size = 0;
  int total_receive_size = 0;
  int total_receive_count = 0;
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i) {
    total_send_size += run_var.neighbors[i].force_send_size;
    total_receive_size += run_var.neighbors[i].force_receive_size;
    total_receive_count +=
                        run_var.neighbors[i].force_receive_list.size();
  }
  
  char *incoming_buffer = new char[total_receive_size];
  run_var.buffer = new char[total_send_size];
  char *in_buffer = incoming_buffer;
  char *out_buffer = run_var.buffer;
  MPI_Request *req = new MPI_Request[run_var.neighbor_count];
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i)
    run_var.neighbors[i].force_receive(in_buffer, &req[i]);
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i)
    run_var.neighbors[i].force_send(out_buffer);
  
  int index;
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i) {
    MPI::wait_any(run_var.neighbor_count, req, &index);
    run_var.neighbors[index].force_process();
  }
  
  delete [] req;
  delete [] incoming_buffer;
}

// send and receive position data
void communicate_positions() {
  if (run_var.neighbor_count == 0) return;
  MPI::empty(req_out);
  if (run_var.buffer) delete [] run_var.buffer;
  
  int total_send_size = 0;
  int total_receive_size = 0;
  int total_receive_count = 0;
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i) {
    total_send_size += run_var.neighbors[i].position_send_size;
    total_receive_size += run_var.neighbors[i].position_receive_size;
    //force send is position receive
    total_receive_count += run_var.neighbors[i].force_send_list.size();
  }
  
  char *incoming_buffer = new char[total_receive_size];
  run_var.buffer = new char[total_send_size];
  char *in_buffer = incoming_buffer;
  char *out_buffer = run_var.buffer;
  MPI_Request *req = new MPI_Request[run_var.neighbor_count];
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i)
    run_var.neighbors[i].position_receive(in_buffer, &req[i]);
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i)
    run_var.neighbors[i].position_send(out_buffer);
  
  int index;
  for (unsigned int i = 0; i < run_var.neighbor_count; ++i) {
    MPI::wait_any(run_var.neighbor_count, req, &index);
    run_var.neighbors[index].position_process();
  }
  
  delete [] req;
  delete [] incoming_buffer;
}
