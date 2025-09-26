
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

template <typename F> long long benchmark(F f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void gemm_naive_ikj(float *a, float *b, float *c, int n, int k, int m) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      for (int kk = 0; kk < k; kk++)
        c[i * m + j] += (a[i * k + kk] * b[kk * m + j]);
}

void gemm_blocked(float *a, float *b, float *c, int n, int k, int m, int bs) {
  for (int ib = 0; ib < n; ib += bs) {
    int mi = std::min(ib + bs, n);
    for (int jb = 0; jb < m; jb += bs) {
      int mj = std::min(jb + bs, m);
      for (int kb = 0; kb < k; kb += bs) {
        int mk = std::min(kb + bs, k);
        for (int i = ib; i < mi; i++)
          for (int kk = kb; kk < mk; kk++)
            for (int j = jb; j < mj; j++)
              c[i * m + j] += (a[i * k + kk] * b[kk * m + j]);
      }
    }
  }
}

void gemm_packed(float *a, float *b, float *c, int n, int k, int m, int bs) {
  float *a_pack = (float *)malloc(bs * bs * sizeof(float));
  float *b_pack = (float *)malloc(bs * bs * sizeof(float));
  if (!a_pack || !b_pack) {
    free(a_pack);
    free(b_pack);
    return;
  }
  for (int ib = 0; ib < n; ib += bs) {
    int n_sub = std::min(bs, n - ib);
    for (int kb = 0; kb < k; kb += bs) {
      int k_sub = std::min(bs, k - kb);
      for (int i = 0; i < n_sub; i++)
        for (int kk = 0; kk < k_sub; kk++)
          a_pack[i * k_sub + kk] = a[(ib + i) * k + (kb + kk)];
      for (int jb = 0; jb < m; jb += bs) {
        int m_sub = std::min(bs, m - jb);
        for (int kk = 0; kk < k_sub; kk++)
          for (int j = 0; j < m_sub; j++)
            b_pack[kk * m_sub + j] = b[(kb + kk) * m + (jb + j)];
        for (int i = 0; i < n_sub; i++)
          for (int kk = 0; kk < k_sub; kk++) {
            float a_val = a_pack[i * k_sub + kk];
            for (int j = 0; j < m_sub; j++)
              c[(ib + i) * m + (jb + j)] += a_val * b_pack[kk * m_sub + j];
          }
      }
    }
  }
  free(a_pack);
  free(b_pack);
}

bool test_correctness(float *a, float *b, int n, int k, int m, int bs) {
  std::vector<float> c_ref(n * m, 0), c_test(n * m, 0);
  gemm_naive_ikj(a, b, c_ref.data(), n, k, m);
  gemm_blocked(a, b, c_test.data(), n, k, m, bs);
  for (int i = 0; i < n * m; i++)
    if (std::fabs(c_ref[i] - c_test[i]) > 1e-3)
      return false;
  std::fill(c_test.begin(), c_test.end(), 0);
  gemm_packed(a, b, c_test.data(), n, k, m, bs);
  for (int i = 0; i < n * m; i++)
    if (std::fabs(c_ref[i] - c_test[i]) > 1e-3)
      return false;
  return true;
}

int main() {
  std::ofstream out("results.csv");
  out << "size,naive,blocked,packed\n";
  for (int s = 1; s <= 4096; s *= 2) {
    std::cout << "Current size : " << s << std::endl;
    int n = s, k = s, m = s, bs = 64;
    std::vector<float> a(n * k), b(k * m), c(n * m);
    for (int i = 0; i < n * k; i++)
      a[i] = (i % 100) * 0.01f;
    for (int i = 0; i < k * m; i++)
      b[i] = (i % 100) * 0.02f;
    if (!test_correctness(a.data(), b.data(), n, k, m, bs)) {
      std::cerr << "Test failed at size " << s << "\n";
      return -1;
    }
    std::fill(c.begin(), c.end(), 0);
    auto naive = benchmark(
        [&]() { gemm_naive_ikj(a.data(), b.data(), c.data(), n, k, m); });
    std::fill(c.begin(), c.end(), 0);
    auto blocked = benchmark(
        [&]() { gemm_blocked(a.data(), b.data(), c.data(), n, k, m, bs); });
    std::fill(c.begin(), c.end(), 0);
    auto packed = benchmark(
        [&]() { gemm_packed(a.data(), b.data(), c.data(), n, k, m, bs); });
    out << s << "," << naive << "," << blocked << "," << packed << "\n";
  }
  return 0;
}
