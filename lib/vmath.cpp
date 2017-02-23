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
