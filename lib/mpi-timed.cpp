// for timing purposes
#if defined(__i386__)

static __inline__ unsigned long long rdtsc(void) {
	unsigned long long int x;
	__asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
	return x;
}
#elif defined(__x86_64__)


static __inline__ unsigned long long rdtsc(void) {
	unsigned hi, lo;
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#elif defined(__powerpc__)
static __inline__ unsigned long long rdtsc(void) {
	unsigned long long int result=0;
	unsigned long int upper, lower,tmp;
	__asm__ volatile(
		"0:                  \n"
		"\tmftbu   %0           \n"
		"\tmftb    %1           \n"
		"\tmftbu   %2           \n"
		"\tcmpw    %2,%0        \n"
		"\tbne     0b         \n"
		: "=r"(upper),"=r"(lower),"=r"(tmp)
	);
	result = upper;
	result = result<<32;
	result = result|lower;

	return(result);
}
#endif

// simple timer using rdtsc()
class timer {
public:
  unsigned long long sum;
  unsigned long long temp;
  
  void start() {
    temp = rdtsc();
  }
  
  void stop() {
    sum += (rdtsc() - temp);
  }
  
  void wipe() {
    sum = 0;
  }
  
  timer() {
    sum = 0;
  }
};

// request types
enum dummy_request_type {
  req_in,
  req_out,
  req_MAX
};

// request tags
enum tag_type {
  tag_search_particles,
  tag_transfer_particles,
  tag_force,
  tag_position
};

// border contact between neighbors
enum connect_type {
  c0,
  cx,
  cy,
  cxy,
  cz,
  cxz,
  cyz,
  cxyz
};

// abstraction layer for automatic timing
namespace MPI {
  timer twait, tinteract, tstep, tsearch, tforce, tposition;
  int rank, size;
  std::queue<MPI_Request *> req_stack[req_MAX];
  
  void init(int *argc, char ***argv) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
  }
  
  void abort(int i) {
    MPI_Abort(MPI_COMM_WORLD, i);
  }
  
  void end() {
    MPI_Finalize();
  }
  
  MPI_Request *request(int i) {
    MPI_Request *q = new MPI_Request;
    req_stack[i].push(q);
    return q;
  }
  
  void wait(MPI_Request *req, MPI_Status *stat=MPI_STATUS_IGNORE) {
    twait.start();
    MPI_Wait(req, stat);
    twait.stop();
  }
  
  void wait_any(int count, MPI_Request *req, int *index,
                MPI_Status *stat=MPI_STATUS_IGNORE) {
    twait.start();
    MPI_Waitany(count, req, index, stat);
    twait.stop();
  }
  
  // wait on and purge the stack of pending requests
  void empty(int i) {
    while (!req_stack[i].empty()) {
      MPI::wait(req_stack[i].front());
      delete [] req_stack[i].front();
      req_stack[i].pop();
    }
  }
  
  // empty all the stacks
  void empty_all() {
    for (int i = 0; i < (int)req_MAX; ++i) MPI::empty(i);
  }
  
  template <class T>
  void isend(T *buff, int dest, int buffsize, int tag,
                                              MPI_Request *req) {
    MPI_Isend((char*)buff, buffsize, MPI_CHAR, dest, tag,
                                               MPI_COMM_WORLD, req);
  }
  
  template <class T>
  void irecv(T *buff, int source, int buffsize, int tag,
                                                MPI_Request *req) {
    MPI_Irecv((char*)buff, buffsize, MPI_CHAR, source, tag,
                                               MPI_COMM_WORLD, req);
  }
  
  template <class T>
  void bcast(T *buff, int buffsize, int root) {
    MPI_Bcast((char *)buff, buffsize, MPI_CHAR, root, MPI_COMM_WORLD);
  }
  
  void allred(void *sbuff, void *rbuff, int count, MPI_Datatype d,
                                                   MPI_Op op) {
    MPI_Allreduce(sbuff, rbuff, count, d, op, MPI_COMM_WORLD);
  }
  
  void reduce(void *sbuff, void *rbuff, int count, int root,
                                        MPI_Datatype dt, MPI_Op op) {
    MPI_Reduce(sbuff, rbuff, count, dt, op, root, MPI_COMM_WORLD);
  }
  
  // collect timing data from processes and print stats
  void clock_assembly() {
    unsigned long long rwait, rstep, rinteract, rsearch, rforce,
                                                         rposition;
    rwait = rstep = rinteract = rsearch = rforce = rposition = 0;
    reduce(&twait.sum, &rwait, 1, 0, MPI_UNSIGNED_LONG_LONG, MPI_SUM);
    reduce(&tstep.sum, &rstep, 1, 0, MPI_UNSIGNED_LONG_LONG, MPI_SUM);
    reduce(&tinteract.sum, &rinteract, 1, 0, MPI_UNSIGNED_LONG_LONG,
                                             MPI_SUM);
    reduce(&tsearch.sum, &rsearch, 1, 0, MPI_UNSIGNED_LONG_LONG,
                                         MPI_SUM);
    reduce(&tforce.sum, &rforce, 1, 0, MPI_UNSIGNED_LONG_LONG, MPI_SUM);
    reduce(&tposition.sum, &rposition, 1, 0, MPI_UNSIGNED_LONG_LONG,
                                             MPI_SUM);
    // only the head process prints
    if (rank == 0) {
      std::cerr << "Wait:       " << rwait / size << '\n';
      std::cerr << "Step:       " << rstep / size << '\n';
      std::cerr << "Interact:   " << rinteract / size << '\n';
      std::cerr << "Search:     " << rsearch / size << '\n';
      std::cerr << "Force:      " << rforce / size << '\n';
      std::cerr << "Position:   " << rposition / size << '\n';
    }
  }
}
