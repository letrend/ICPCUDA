#include "containers/safe_call.hpp"
#include "internal.h"

template <int D>
__inline__ __device__ void
warpReduceSum(Eigen::Matrix<float, D, 1, Eigen::DontAlign> &val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < D; i++) {
      val[i] += __shfl_down_sync(0xFFFFFFFF, val[i], offset);
    }
  }
}

template <int D>
__inline__ __device__ void
blockReduceSum(Eigen::Matrix<float, D, 1, Eigen::DontAlign> &val) {
  // Allocate shared memory in two steps otherwise NVCC complains about Eigen's
  // non-empty constructor
  static __shared__ unsigned char
      sharedMem[32 * sizeof(Eigen::Matrix<float, D, 1, Eigen::DontAlign>)];

  Eigen::Matrix<float, D, 1, Eigen::DontAlign>(&shared)[32] =
      reinterpret_cast<Eigen::Matrix<float, D, 1, Eigen::DontAlign>(&)[32]>(
          sharedMem);

  int lane = threadIdx.x % warpSize;

  int wid = threadIdx.x / warpSize;

  warpReduceSum(val);

  // write reduced value to shared memory
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // ensure we only grab a value from shared memory if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize)
            ? shared[lane]
            : Eigen::Matrix<float, D, 1, Eigen::DontAlign>::Zero();

  if (wid == 0) {
    warpReduceSum(val);
  }
}

template <int D>
__global__ void reduceSum(Eigen::Matrix<float, D, 1, Eigen::DontAlign> *in,
                          Eigen::Matrix<float, D, 1, Eigen::DontAlign> *out,
                          int N) {
  Eigen::Matrix<float, D, 1, Eigen::DontAlign> sum =
      Eigen::Matrix<float, D, 1, Eigen::DontAlign>::Zero();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }

  blockReduceSum(sum);

  if (threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}

struct Reduction {
  Eigen::Matrix<float, 3, 3, Eigen::DontAlign> R_prev_curr;
  Eigen::Matrix<float, 3, 1, Eigen::DontAlign> t_prev_curr;

  Intr intr;

  PtrStep<float> vmap_curr;
  PtrStep<float> nmap_curr;

  PtrStep<float> vmap_prev;
  PtrStep<float> nmap_prev;

  float dist_thresh;
  float angle_thresh;

  int cols;
  int rows;
  int N;

  Eigen::Matrix<float, 29, 1, Eigen::DontAlign> *out;

  // And now for some template metaprogramming magic
  template <int outer, int inner, int end> struct SquareUpperTriangularProduct {
    __device__ __forceinline__ static void
    apply(Eigen::Matrix<float, 29, 1, Eigen::DontAlign> &values,
          const float (&rows)[end + 1]) {
      values[((end + 1) * outer) + inner - (outer * (outer + 1) / 2)] =
          rows[outer] * rows[inner];

      SquareUpperTriangularProduct<outer, inner + 1, end>::apply(values, rows);
    }
  };

  // Inner loop base
  template <int outer, int end>
  struct SquareUpperTriangularProduct<outer, end, end> {
    __device__ __forceinline__ static void
    apply(Eigen::Matrix<float, 29, 1, Eigen::DontAlign> &values,
          const float (&rows)[end + 1]) {
      values[((end + 1) * outer) + end - (outer * (outer + 1) / 2)] =
          rows[outer] * rows[end];

      SquareUpperTriangularProduct<outer + 1, outer + 1, end>::apply(values,
                                                                     rows);
    }
  };

  // Outer loop base
  template <int end> struct SquareUpperTriangularProduct<end, end, end> {
    __device__ __forceinline__ static void
    apply(Eigen::Matrix<float, 29, 1, Eigen::DontAlign> &values,
          const float (&rows)[end + 1]) {
      values[((end + 1) * end) + end - (end * (end + 1) / 2)] =
          rows[end] * rows[end];
    }
  };

  __device__ __forceinline__ void operator()() const {
    Eigen::Matrix<float, 29, 1, Eigen::DontAlign> sum =
        Eigen::Matrix<float, 29, 1, Eigen::DontAlign>::Zero();

    SquareUpperTriangularProduct<0, 0, 6> sutp;

    Eigen::Matrix<float, 29, 1, Eigen::DontAlign> values;

    // Cache R and t into raw scalars to avoid Eigen Matrix-Matrix product on
    // device (Eigen 3.3.x has __host__-only Assignment that NVCC silently
    // calls from __device__ code, yielding undefined behavior).
    const float R00 = R_prev_curr(0,0), R01 = R_prev_curr(0,1), R02 = R_prev_curr(0,2);
    const float R10 = R_prev_curr(1,0), R11 = R_prev_curr(1,1), R12 = R_prev_curr(1,2);
    const float R20 = R_prev_curr(2,0), R21 = R_prev_curr(2,1), R22 = R_prev_curr(2,2);
    const float tx = t_prev_curr(0), ty = t_prev_curr(1), tz = t_prev_curr(2);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
      const int y = i / cols;
      const int x = i - (y * cols);

      const float vcx = vmap_curr.ptr(y)[x];
      const float vcy = vmap_curr.ptr(y + rows)[x];
      const float vcz = vmap_curr.ptr(y + 2 * rows)[x];

      // v_curr_in_prev = R * v_curr + t  (raw scalars)
      const float vpx = R00 * vcx + R01 * vcy + R02 * vcz + tx;
      const float vpy = R10 * vcx + R11 * vcy + R12 * vcz + ty;
      const float vpz = R20 * vcx + R21 * vcy + R22 * vcz + tz;

      const int px = __float2int_rn(vpx * intr.fx / vpz + intr.cx);
      const int py = __float2int_rn(vpy * intr.fy / vpz + intr.cy);

      float row[7] = {0, 0, 0, 0, 0, 0, 0};

      values[28] = 0;

      if (px >= 0 && py >= 0 && px < cols && py < rows &&
          vcz > 0 && vpz > 0) {
        const float vprev_x = vmap_prev.ptr(py)[px];
        const float vprev_y = vmap_prev.ptr(py + rows)[px];
        const float vprev_z = vmap_prev.ptr(py + 2 * rows)[px];

        const float ncx = nmap_curr.ptr(y)[x];
        const float ncy = nmap_curr.ptr(y + rows)[x];
        const float ncz = nmap_curr.ptr(y + 2 * rows)[x];

        // n_curr_in_prev = R * n_curr
        const float nipx = R00 * ncx + R01 * ncy + R02 * ncz;
        const float nipy = R10 * ncx + R11 * ncy + R12 * ncz;
        const float nipz = R20 * ncx + R21 * ncy + R22 * ncz;

        const float npx = nmap_prev.ptr(py)[px];
        const float npy = nmap_prev.ptr(py + rows)[px];
        const float npz = nmap_prev.ptr(py + 2 * rows)[px];

        // cross(n_curr_in_prev, n_prev)
        const float cx_ = nipy * npz - nipz * npy;
        const float cy_ = nipz * npx - nipx * npz;
        const float cz_ = nipx * npy - nipy * npx;
        const float ncross = sqrtf(cx_*cx_ + cy_*cy_ + cz_*cz_);

        // v_prev - v_curr_in_prev
        const float dx = vprev_x - vpx;
        const float dy = vprev_y - vpy;
        const float dz = vprev_z - vpz;
        const float dnorm = sqrtf(dx*dx + dy*dy + dz*dz);

        if (ncross < angle_thresh && dnorm < dist_thresh &&
            !isnan(ncx) && !isnan(npx)) {
          // row[0..2] = n_prev
          row[0] = npx; row[1] = npy; row[2] = npz;
          // row[3..5] = cross(v_curr_in_prev, n_prev)
          row[3] = vpy * npz - vpz * npy;
          row[4] = vpz * npx - vpx * npz;
          row[5] = vpx * npy - vpy * npx;
          // row[6] = n_prev.dot(v_prev - v_curr_in_prev)
          row[6] = npx * dx + npy * dy + npz * dz;

          values[28] = 1;

          sutp.apply(values, row);

          sum += values;
        }
      }
    }

    blockReduceSum(sum);

    if (threadIdx.x == 0) {
      out[blockIdx.x] = sum;
    }
  }
};

__global__ void estimateKernel(const Reduction reduction) { reduction(); }

void estimateStep(
    const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> &R_prev_curr,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> &t_prev_curr,
    const DeviceArray2D<float> &vmap_curr,
    const DeviceArray2D<float> &nmap_curr, const Intr &intr,
    const DeviceArray2D<float> &vmap_prev,
    const DeviceArray2D<float> &nmap_prev, float dist_thresh,
    float angle_thresh,
    DeviceArray<Eigen::Matrix<float, 29, 1, Eigen::DontAlign>> &sum,
    DeviceArray<Eigen::Matrix<float, 29, 1, Eigen::DontAlign>> &out,
    float *matrixA_host, float *vectorB_host, float *residual_inliers,
    int threads, int blocks) {
  int cols = vmap_curr.cols();
  int rows = vmap_curr.rows() / 3;

  Reduction reduction;

  reduction.R_prev_curr = R_prev_curr;
  reduction.t_prev_curr = t_prev_curr;

  reduction.vmap_curr = vmap_curr;
  reduction.nmap_curr = nmap_curr;

  reduction.intr = intr;

  reduction.vmap_prev = vmap_prev;
  reduction.nmap_prev = nmap_prev;

  reduction.dist_thresh = dist_thresh;
  reduction.angle_thresh = angle_thresh;

  reduction.cols = cols;
  reduction.rows = rows;

  reduction.N = cols * rows;
  reduction.out = sum;

  estimateKernel<<<blocks, threads>>>(reduction);

  reduceSum<29><<<1, MAX_THREADS>>>(sum, out, blocks);

  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());

  float host_data[29];
  out.download((Eigen::Matrix<float, 29, 1, Eigen::DontAlign> *)&host_data[0]);

  int shift = 0;
  for (int i = 0; i < 6; ++i) // rows
  {
    for (int j = i; j < 7; ++j) // cols + b
    {
      float value = host_data[shift++];
      if (j == 6) // vector b
        vectorB_host[i] = value;
      else
        matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
    }
  }

  residual_inliers[0] = host_data[27];
  residual_inliers[1] = host_data[28];
}
