#include <cuda_runtime.h>
#include <cuComplex.h>
#include <inttypes.h>
#include <stdio.h>


__global__ void byte_to_float_kernel(const char* input, float* output, unsigned size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned ii;
  for (ii=idx; ii<size; ii+=gridDim.x*blockDim.x)
  {
//    output[ii] = ((float) input[ii]) / 127.0;  // use this if scaling required
    output[ii] = (float)input[ii];
  }
  return;
}

extern "C"
int mwax_byte_to_float(char* input, float* output, unsigned size, cudaStream_t stream)
{
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props,0);
  int nthreads = props.maxThreadsPerBlock;
  int max_blocks = props.maxGridSize[0];
  int nblocks;
  if (size < nthreads)
    nthreads = size;
  nblocks = (size + nthreads - 1) / nthreads;
  if (nblocks > max_blocks)
    nblocks = max_blocks;

  byte_to_float_kernel<<<nblocks,nthreads,0,stream>>>(input,output,size);

  return 0;
}



extern "C"
char * mwax_cuda_get_device_name (int index)
{
  cudaDeviceProp device_prop;
  cudaError_t error_id = cudaGetDeviceProperties(&device_prop, index);
  if (error_id != cudaSuccess)
  {
    fprintf (stderr, "mwax_cuda_get_device_name: cudaGetDeviceProperties failed: %s\n",
                     cudaGetErrorString(error_id) );
    return 0;
  }
  return strdup(device_prop.name);
}



__global__ void scalar_weight_complex_kernel(const float weight, float* output, unsigned size)
// multiplies all elements of complex float vector in output by the same scalar weight value and
// places the results in output. weight is a float multiplier, size is the number of complex samples.
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned ii, jj;
  for (ii=idx; ii<size; ii+=(gridDim.x*blockDim.x))
  {
    jj = 2*ii;
    output[jj] = output[jj]*weight;
    output[jj+1] = output[jj+1]*weight;
  }
  return;
}

extern "C"
int mwax_scalar_weight_complex(float weight, float* output, unsigned size, cudaStream_t stream)
{
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props,0);
  int nthreads = props.maxThreadsPerBlock;
  int max_blocks = props.maxGridSize[0];
  int nblocks;
  if (size < nthreads)
    nthreads = size;
  nblocks = (size + nthreads - 1) / nthreads;
  if (nblocks > max_blocks)
    nblocks = max_blocks;

  scalar_weight_complex_kernel<<<nblocks,nthreads,0,stream>>>(weight,output,size);

  return 0;
}



__global__ void vector_weight_complex_kernel(const float* weights, float* output, unsigned size)
// multiplies the elements of the complex float vector in output by the values in vector weights
// and places the results in output. size is the number of complex samples.  weight is a vector
// of float multipliers, of length size.
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned ii, jj;
  for (ii=idx; ii<size; ii+=(gridDim.x*blockDim.x))
  {
    jj = 2*ii;
    output[jj] = output[jj]*weights[ii];
    output[jj+1] = output[jj+1]*weights[ii];
  }
  return;
}

extern "C"
int mwax_vector_weight_complex(float* weights, float* output, unsigned size, cudaStream_t stream)
{
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props,0);
  int nthreads = props.maxThreadsPerBlock;
  int max_blocks = props.maxGridSize[0];
  int nblocks;
  if (size < nthreads)
    nthreads = size;
  nblocks = (size + nthreads - 1) / nthreads;
  if (nblocks > max_blocks)
    nblocks = max_blocks;

  vector_weight_complex_kernel<<<nblocks,nthreads,0,stream>>>(weights,output,size);

  return 0;
}



__global__ void array_weight_complex_kernel(const float* weights, float* output, unsigned rows, unsigned columns)
// multiplies the elements of the complex float 2D array in output by the values in vector weights
// and places the results in output. The same weight is applied to all elements of a row.
// weight is a vector of float multipliers, of length rows.
{
  // blockIdx.x is column (input freq and time)
  // threadIdx.x is row (input antenna)
  float weight = weights[threadIdx.x];
  int idx = 2*(threadIdx.x*columns + blockIdx.x);
  output[idx] = output[idx]*weight;
  output[idx+1] = output[idx+1]*weight;

  return;
}

extern "C"
int mwax_array_weight_complex(float* weights, float* output, unsigned rows, unsigned columns, cudaStream_t stream)
{
  int nblocks = (int)columns;
  int nthreads = (int)rows;
  array_weight_complex_kernel<<<nblocks,nthreads,0,stream>>>(weights,output,rows,columns);

  return 0;
}



__global__ void complex_multiply_kernel(const float* input, float* output, unsigned size)
// multiplies output by input and places in output, size is number of complex samples
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned ii,jj;
  float temp;
  for (ii=idx; ii<size; ii+=(gridDim.x*blockDim.x))
  {
    jj = 2*ii;
    temp = output[jj];
    output[jj] = temp*input[jj] - output[jj+1]*input[jj+1];
    output[jj+1] = output[jj+1]*input[jj] + temp*input[jj+1];
  }
  return;
}

extern "C"
int mwax_complex_multiply(float* input, float* output, unsigned size, cudaStream_t stream)
{
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props,0);
  int nthreads = props.maxThreadsPerBlock;
  int max_blocks = props.maxGridSize[0];
  int nblocks;
  if (size < nthreads)
    nthreads = size;
  nblocks = (size + nthreads - 1) / nthreads;
  if (nblocks > max_blocks)
    nblocks = max_blocks;

  complex_multiply_kernel<<<nblocks,nthreads,0,stream>>>(input,output,size);

  return 0;
}



__global__ void fast_complex_multiply_kernel(const float* input, float* output, unsigned size)
// multiplies output by input and places in output, size is number of complex samples
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned ii,jj;
  float temp1, temp2, temp3;
  for (ii=idx; ii<size; ii+=(gridDim.x*blockDim.x))
  {
    jj = 2*ii;
    temp1 = output[jj]*(input[jj] + input[jj+1]);
    temp2 = input[jj+1]*(output[jj] + output[jj+1]);
    temp3 = input[jj]*(output[jj+1] - output[jj]);
    output[jj] = temp1 - temp2;
    output[jj+1] = temp1 + temp3;
  }
  return;
}

extern "C"
int mwax_fast_complex_multiply(float* input, float* output, unsigned size, cudaStream_t stream)
{
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props,0);
  int nthreads = props.maxThreadsPerBlock;
  int max_blocks = props.maxGridSize[0];
  int nblocks;
  if (size < nthreads)
    nthreads = size;
  nblocks = (size + nthreads - 1) / nthreads;
  if (nblocks > max_blocks)
    nblocks = max_blocks;

  fast_complex_multiply_kernel<<<nblocks,nthreads,0,stream>>>(input,output,size);

  return 0;
}



__global__ void mwax_lookup_delay_gains_kernel(const int32_t* delays, const cuFloatComplex* delay_lut, cuFloatComplex* delay_gains, unsigned paths, unsigned fft_length, unsigned num_ffts)
// assembles the complex float 2D array of delay_gains taken from the delay_lut, indexed with delays
{
  // blockIdx.x is fft_length (input freqs)
  // threadIdx.x is row (input antenna)
  #define NUM_DELAYS 2001  // zero delay case and +- 1000 millisamples
  #define MAX_DELAY ((NUM_DELAYS-1)/2)
  int delay_idx = (delays[threadIdx.x] + MAX_DELAY)*fft_length + blockIdx.x;
  cuFloatComplex delay_gain = delay_lut[delay_idx];
  int gains_idx = (threadIdx.x*fft_length*num_ffts) + blockIdx.x;
  int i;
  for (i=0; i<num_ffts; i++)  // each FFT will use the same delay gain gradient
  {
    delay_gains[gains_idx + i*fft_length] = delay_gain;
  }
  return;
}

extern "C"
int mwax_lookup_delay_gains(int32_t* delays, cuFloatComplex* delay_lut, cuFloatComplex* delay_gains, unsigned paths, unsigned fft_length, unsigned num_ffts, cudaStream_t stream)
{
  int nblocks = (int)fft_length;
  int nthreads = (int)paths;
  mwax_lookup_delay_gains_kernel<<<nblocks,nthreads,0,stream>>>(delays,delay_lut,delay_gains,paths,fft_length,num_ffts);

  return 0;
}



__global__ void mwax_lookup_and_apply_delay_gains_kernel(const int32_t* delays, const float* delay_lut, float* input, unsigned paths, unsigned fft_length, unsigned num_ffts)
// applies delay gains to the complex float 2D array of input, using values taken from the delay_lut, indexed with delays
// floats are used to allow fast complex multiply algorithm
{
  // blockIdx.x is fft_length (input freqs)
  // threadIdx.x is row (input antenna)
  #define NUM_DELAYS 2001  // zero delay case and +- 1000 millisamples
  #define MAX_DELAY ((NUM_DELAYS-1)/2)
  int delay_idx = 2*((delays[threadIdx.x] + MAX_DELAY)*fft_length + blockIdx.x);
  float delay_gain_real = delay_lut[delay_idx];
  float delay_gain_imag = delay_lut[delay_idx+1];
  int input_idx = 2*((threadIdx.x*fft_length*num_ffts) + blockIdx.x);
  float input_real = input[input_idx];
  float input_imag = input[input_idx+1];
  float temp1 = input_real*(delay_gain_real + delay_gain_imag);  // form intermediate products of fast complex multiply
  float temp2 = delay_gain_imag*(input_real + input_imag);
  float temp3 = delay_gain_real*(input_imag - input_real);

  int i;
  for (i=0; i<(2*num_ffts); i+=2)  // each FFT will use the same delay gain gradient
  {
    input[input_idx + i*fft_length] = temp1 - temp2;
    input[input_idx + i*fft_length + 1] = temp1 + temp3;
  }
  return;
}

extern "C"
int mwax_lookup_and_apply_delay_gains(int32_t* delays, float* delay_lut, float* input, unsigned paths, unsigned fft_length, unsigned num_ffts, cudaStream_t stream)
{
  int nblocks = (int)fft_length;
  int nthreads = (int)paths;
  mwax_lookup_and_apply_delay_gains_kernel<<<nblocks,nthreads,0,stream>>>(delays,delay_lut,input,paths,fft_length,num_ffts);

  return 0;
}



__global__ void transpose_to_xGPU_kernel(const cuFloatComplex* input, cuFloatComplex* output, unsigned rows, unsigned columns)
{
  // blockIdx.x is column (input freq and time)
  // threadIdx.x is row (input antenna)

  output[(blockIdx.x * rows) + threadIdx.x] = input[blockIdx.x + (threadIdx.x * columns)];

  return;
}

extern "C"
int mwax_transpose_to_xGPU(cuFloatComplex* input, cuFloatComplex* output, unsigned rows, unsigned columns, cudaStream_t stream)
{
  int nblocks = (int)columns;
  int nthreads = (int)rows;

  transpose_to_xGPU_kernel<<<nblocks,nthreads,0,stream>>>(input,output,rows,columns);

  return 0;
}



__global__ void transpose_to_xGPU_and_weight_kernel(const float* weights, const float* input, float* output, unsigned rows, unsigned columns)
{
  // blockIdx.x is column (input freq and time)
  // threadIdx.x is row (input antenna)
  float weight = weights[threadIdx.x];
  int idx_read = 2*(threadIdx.x*columns + blockIdx.x);
  int idx_write = 2*(blockIdx.x*rows + threadIdx.x);

  output[idx_write] = input[idx_read]*weight;
  output[idx_write+1] = input[idx_read+1]*weight;

  return;
}

extern "C"
int mwax_transpose_to_xGPU_and_weight(float* weights, float* input, float* output, unsigned rows, unsigned columns, cudaStream_t stream)
{
  int nblocks = (int)columns;  // frequencies and time
  int nthreads = (int)rows;    // signal paths
  transpose_to_xGPU_and_weight_kernel<<<nblocks,nthreads,0,stream>>>(weights,input,output,rows,columns);

  return 0;
}



__global__ void detranspose_from_xGPU_kernel(const cuFloatComplex* input, cuFloatComplex* output, unsigned rows, unsigned columns)
{
  // blockIdx.x is row (input freq and time)
  // threadIdx.x is column (input antenna)

  output[blockIdx.x + (threadIdx.x * rows)] = input[(blockIdx.x * columns) + threadIdx.x];

  return;
}

extern "C"
int mwax_detranspose_from_xGPU(cuFloatComplex* input, cuFloatComplex* output, unsigned rows, unsigned columns, cudaStream_t stream)
{
  int nblocks = (int)rows;     // frequencies and time
  int nthreads = (int)columns; // signal paths

  detranspose_from_xGPU_kernel<<<nblocks,nthreads,0,stream>>>(input,output,rows,columns);

  return 0;
}



__global__ void detranspose_from_xGPU_and_weight_kernel(const float* weights, const float* input, float* output, unsigned rows, unsigned columns)
{
  // blockIdx.x is row (input freq and time)
  // threadIdx.x is column (input antenna)
  float weight = weights[threadIdx.x];
  int idx_read = 2*(blockIdx.x*columns + threadIdx.x);
  int idx_write = 2*(threadIdx.x*rows + blockIdx.x);

  output[idx_write] = input[idx_read]*weight;
  output[idx_write+1] = input[idx_read+1]*weight;

  return;
}

extern "C"
int mwax_detranspose_from_xGPU_and_weight(float* weights, float* input, float* output, unsigned rows, unsigned columns, cudaStream_t stream)
{
  int nblocks = (int)rows;     // frequencies and time
  int nthreads = (int)columns; // signal paths
  detranspose_from_xGPU_and_weight_kernel<<<nblocks,nthreads,0,stream>>>(weights,input,output,rows,columns);

  return 0;
}



#if 0
// MORE THREADS VERSION - no faster than nthreads=128
__global__ void xGPU_channel_average_kernel(const float* input, float* output, unsigned num_visibility_samps_per_chan, unsigned num_output_channels, unsigned fscrunch_factor)
{
  // blockDim threads per block: each thread sums all values (real or imag floats) for one output channel
  // blockIdx.x is a chunk of visibility columns (total columns = num_visibility_samps_per_chan)
  // threadIdx.x is index within a block
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // column index
  int read_idx,write_idx;
  float tempSum;
  int ii,j,k;

  for (ii=idx; ii<num_visibility_samps_per_chan; ii+=gridDim.x*blockDim.x)
  {
    read_idx = ii;
    write_idx = ii;
    for (j=0; j<num_output_channels; j++)
    {
      tempSum = input[read_idx];  // first row of this channel
      read_idx += num_visibility_samps_per_chan;  // advance to next row
      for (k=1; k<fscrunch_factor; k++)  // sum remaining rows
      {
        tempSum += input[read_idx];  // sum in this row
        read_idx += num_visibility_samps_per_chan;  // advance to next row
      }
      output[write_idx] = tempSum;  // channel summed - write to this row
      write_idx += num_visibility_samps_per_chan;  // advance to next output row
    }
  }
  return;
}

extern "C"
int xGPU_channel_average(float* input, float* output, unsigned num_visibility_samps_per_chan, unsigned num_input_channels, unsigned fscrunch_factor, cudaStream_t stream)
{
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props,0);
  int nthreads = props.maxThreadsPerBlock;
  int max_blocks = props.maxGridSize[0];
  int nblocks;
  if (num_visibility_samps_per_chan/nthreads > max_blocks)
    nblocks = max_blocks;
  else
    nblocks = (int)num_visibility_samps_per_chan/nthreads;

  if (num_input_channels % fscrunch_factor)
  {
    printf("xGPU_channel_average: ERROR: number of input channels not an integer multiple of averaging factor\n");
    return -1;
  }
  unsigned num_output_channels = num_input_channels/fscrunch_factor;

  xGPU_channel_average_kernel<<<nblocks,nthreads,0,stream>>>(input,output,num_visibility_samps_per_chan,num_output_channels,fscrunch_factor);

  return 0;
}
#endif


#if 1
// FAST VERSION - nthreads=128
__global__ void xGPU_channel_average_kernel(const float* input, float* output, unsigned num_visibility_samps_per_chan, unsigned num_output_channels, unsigned fscrunch_factor)
{
  // blockDim threads per block: each thread sums all values (real or imag floats) for one output channel
  // blockIdx.x is a chunk of visibility columns (total columns = num_visibility_samps_per_chan)
  // threadIdx.x is index within a block
  int read_idx = blockIdx.x * blockDim.x + threadIdx.x;  // column index
  int write_idx = read_idx;
  float tempSum;
  int j,k;
  for (j=0; j<num_output_channels; j++)
  {
    tempSum = input[read_idx];  // first row of this channel
    read_idx += num_visibility_samps_per_chan;  // advance to next row
    for (k=1; k<fscrunch_factor; k++)  // sum remaining rows
    {
      tempSum += input[read_idx];  // sum in this row
      read_idx += num_visibility_samps_per_chan;  // advance to next row
    }
    output[write_idx] = tempSum;  // channel summed - write to this row
    write_idx += num_visibility_samps_per_chan;  // advance to next output row
  }
  return;
}

extern "C"
int xGPU_channel_average(float* input, float* output, unsigned num_visibility_samps_per_chan, unsigned num_input_channels, unsigned fscrunch_factor, cudaStream_t stream)
{
  if (num_visibility_samps_per_chan % 64)
  {
    printf("xGPU_channel_average: ERROR: number of visibility samples (real or imag) per channel should always be divisible by 64\n");
    return -1;
  }

  if (num_input_channels % fscrunch_factor)
  {
    printf("xGPU_channel_average: ERROR: number of input channels not an integer multiple of averaging factor\n");
    return -1;
  }

  int nthreads = 64;  // all blocks will be 64 threads in size
  int nblocks = (int)num_visibility_samps_per_chan/nthreads;
  unsigned num_output_channels = num_input_channels/fscrunch_factor;

  xGPU_channel_average_kernel<<<nblocks,nthreads,0,stream>>>(input,output,num_visibility_samps_per_chan,num_output_channels,fscrunch_factor);

  return 0;
}
#endif


#if 0
// SLOW VERSION - that seems to work

__global__ void xGPU_channel_average_kernel(const float* input, float* output, unsigned num_visibility_samps_per_chan, unsigned fscrunch_factor)
{
  // blockIdx.x is column (baselines - length num_visibility_samps_per_chan)
  // threadIdx.x is row (output channel)
  int read_idx = blockIdx.x + (threadIdx.x * fscrunch_factor * num_visibility_samps_per_chan);
  float tempSum;
  int k;
  for (k=0; k<fscrunch_factor; k++)
  {
    tempSum += input[read_idx];
    read_idx += num_visibility_samps_per_chan;
  }
  output[blockIdx.x + (threadIdx.x * num_visibility_samps_per_chan)] = tempSum;
  return;
}
#endif

#if 0
// REALLY SLOW VERSION - that seems to work

__global__ void xGPU_channel_average_kernel(const float* input, float* output, unsigned num_visibility_samps_per_chan, unsigned fscrunch_factor)
{
  // blockIdx.x is column (baselines - length num_visibility_samps_per_chan)
  // threadIdx.x is row (output channel)
  // TO DO: speed up by changing multiplications in index calculations to sums around the FOR loop
  int k;
  for (k=0; k<fscrunch_factor; k++)
  {
    output[blockIdx.x + (threadIdx.x * num_visibility_samps_per_chan)] += input[blockIdx.x + ((fscrunch_factor*threadIdx.x + k) * num_visibility_samps_per_chan)];
  }
  return;
}
#endif

#if 0
extern "C"
int xGPU_channel_average(float* input, float* output, unsigned num_visibility_samps_per_chan, unsigned num_input_channels, unsigned fscrunch_factor, cudaStream_t stream)
{
  if (num_input_channels % fscrunch_factor)
  {
    printf("xGPU_channel_average: ERROR: number of input channels not an integer multiple of averaging factor\n");
    return -1;
  }

  int nblocks = (int)num_visibility_samps_per_chan;
  int nthreads = (int)(num_input_channels/fscrunch_factor);  // number of output channels

  // clear all outputs
  cudaMemset( output, 0.0, (num_visibility_samps_per_chan*num_input_channels*4) );

  xGPU_channel_average_kernel<<<nblocks,nthreads,0,stream>>>(input,output,num_visibility_samps_per_chan,fscrunch_factor);

  return 0;
}
#endif



// Performs channel summation on xGPU output visibilities as well as shifting the DC channel to the centre output channel
// and scaling by a specified multiplicative factor that combines time/frequency normalisation and weighting (due to missing data).
// Also excludes the DC ultrafine channel from the summation of the centre channel.
__global__ void xGPU_channel_average_shift_and_scale_kernel(const float* input, float* output, unsigned num_visibility_samps_per_chan, unsigned num_output_channels, unsigned fscrunch_factor, float chan0_scale_factor, float remaining_scale_factor)
{
  // blockDim threads per block: each thread sums all values (real or imag floats) for one output channel
  // blockIdx.x is a chunk of visibility columns (total columns = num_visibility_samps_per_chan)
  // threadIdx.x is index within a block
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;  // which visibility sample across the row
  int write_idx = column_index + (num_visibility_samps_per_chan*num_output_channels/2);  // start halfway into output array to centre the DC channel
  int read_idx = column_index;
  float tempSum;
  int j,k;
  // first output channel is different - we exclude the first (DC) ultrafine channel
  tempSum = input[read_idx];  // first row of this channel
  read_idx += num_visibility_samps_per_chan;  // advance to next row
  for (k=2; k<fscrunch_factor; k++)  // sum remaining rows
  {
    tempSum += input[read_idx];  // sum in this row
    read_idx += num_visibility_samps_per_chan;  // advance to next row
  }
  output[write_idx] = tempSum * chan0_scale_factor;  // channel summed - scale and write to this row
  if (num_output_channels == 1)   // all done so return
    return;

  // process the remaining output channels
  for (j=1; j<num_output_channels; j++)
  {
    tempSum = input[read_idx];  // first row of this channel
    read_idx += num_visibility_samps_per_chan;  // advance to next row
    for (k=1; k<fscrunch_factor; k++)  // sum remaining rows
    {
      tempSum += input[read_idx];  // sum in this row
      read_idx += num_visibility_samps_per_chan;  // advance to next row
    }

    if (j == (num_output_channels/2))
      write_idx = column_index;  // wrap back to first output row (most negative frequency channel)
    else
      write_idx += num_visibility_samps_per_chan;  // advance to next output row

    output[write_idx] = tempSum * remaining_scale_factor;  // channel summed - scale and write to this row
  }
  return;
}

extern "C"
int xGPU_channel_average_shift_and_scale(float* input, float* output, unsigned num_visibility_samps_per_chan, unsigned num_input_channels, unsigned fscrunch_factor, float scale_factor, cudaStream_t stream)
{
  if (num_visibility_samps_per_chan % 64)
  {
    printf("xGPU_channel_average: ERROR: number of visibility samples (real or imag) per channel should always be divisible by 64\n");
    return -1;
  }

  if (num_input_channels % fscrunch_factor)
  {
    printf("xGPU_channel_average: ERROR: number of input channels not an integer multiple of averaging factor\n");
    return -1;
  }

  int nthreads = 64;  // all blocks will be 64 threads in size
  int nblocks = (int)num_visibility_samps_per_chan/nthreads;
  unsigned num_output_channels = num_input_channels/fscrunch_factor;
  float channel0_scale_factor;
  if (fscrunch_factor == 1)
    channel0_scale_factor = 0.0;  // clear channel 0 when no channel averaging
  else
    channel0_scale_factor = scale_factor*(float)fscrunch_factor/((float)fscrunch_factor - 1.0);  // channel 0 is averaging over one fewer ultrafine channels

  // call kernel with input pointer advanced to second row to exclude the first (DC) ultrafine channel
  xGPU_channel_average_shift_and_scale_kernel<<<nblocks,nthreads,0,stream>>>((input+num_visibility_samps_per_chan),output,num_visibility_samps_per_chan,num_output_channels,fscrunch_factor,channel0_scale_factor,scale_factor);

  return 0;
}




__global__ void beamform_summation_kernel(const float* input, float* output, unsigned num_antennas, unsigned num_samples_per_antenna)
{
  // each thread sums the values (real or imag floats) for all antennas, for one sample index (column)
  // blockIdx.x is a chunk of sample columns (total columns = num_samples_per_antenna)
  // threadIdx.x is index within a block
  int read_idx = blockIdx.x * blockDim.x + threadIdx.x;  // column index.  In 2D, there are columns=num_samples_per_antenna, rows=(2*num_antennas)
  int write_idx = read_idx;
  int two_row_stride = 2*num_samples_per_antenna;
  float tempSumPol1;
  float tempSumPol2;
  int j;
  tempSumPol1 = input[read_idx];  // first antenna this column (1st pol)
  tempSumPol2 = input[read_idx + num_samples_per_antenna];  // first antenna this column (2nd pol)
  read_idx += two_row_stride;     // advance two rows (next row of same pol)
  for (j=1; j<num_antennas; j++)
  {
    tempSumPol1 += input[read_idx];  // sum in element this antenna (1st pol)
    tempSumPol2 += input[read_idx + num_samples_per_antenna];  // sum in element this antenna (2nd pol)
    read_idx += two_row_stride;      // advance two rows (next row of same pol)
  }
  output[write_idx] = tempSumPol1;  // samples summed - write to 1st pol row
  output[write_idx + num_samples_per_antenna] = tempSumPol2;  // samples summed - write to 2nd pol row

  return;
}

extern "C"
int mwax_beamform_summation(float* input, float* output, unsigned num_antennas, unsigned num_samples_per_antenna, cudaStream_t stream)
// NOTE: this version assumes float samples, so for complex samples must be called with num_samples_per_antenna = 2*NUM_SAMPS_PER_BLOCK_PER_ANT
{
  if (num_samples_per_antenna % 128)
  {
    printf("mwax_beamform_summation: ERROR: number of samples (real + imag) per antenna should always be divisible by 128\n");
    return -1;
  }

  int nthreads = 128;  // all blocks will be 128 threads in size
  int nblocks = (int)num_samples_per_antenna/nthreads;  // 800 when 2*51,200 samples

  beamform_summation_kernel<<<nblocks,nthreads,0,stream>>>(input,output,num_antennas,num_samples_per_antenna);

  return 0;
}



__global__ void sum_powers_dual_pol_kernel(const float* input, float* output, unsigned num_antennas, unsigned num_samples_per_antenna)
{
  // each thread sums the sample power for all antennas, for one sample index (column), for 1 pol only
  // blockIdx.x is a chunk of complex sample columns (total columns = num_samples_per_antenna)
  // threadIdx.x is index within a block
  // work in floats, so read_idx doubled
  int read_idx = 2*(blockIdx.x*blockDim.x + threadIdx.x);  // column index.  In 2D, there are columns=num_samples_per_antenna, rows=num_antennas
  int write_idx = blockIdx.x*blockDim.x + threadIdx.x;     //
  int two_row_stride = 4*num_samples_per_antenna;  // 2 for working in floats, 2 for skipping a row
  float real, imag;
  float tempSum;
  int j;
  real = input[read_idx];
  imag = input[read_idx+1];
  tempSum = real*real + imag*imag;  // power of first antenna this column
  read_idx += two_row_stride;       // advance two rows (next row of same pol)
  for (j=1; j<num_antennas; j++)
  {
    real = input[read_idx];
    imag = input[read_idx+1];
    tempSum += real*real + imag*imag;  // sum in power this antenna
    read_idx += two_row_stride;        // advance two rows (next row of same pol)
  }
  output[write_idx] = tempSum;  // samples summed - write row

  return;
}

extern "C"
int mwax_sum_powers_dual_pol (float* input, float* output, unsigned num_antennas, unsigned num_samples_per_antenna, cudaStream_t stream)
// NOTE: this version assumes float complex samples, so must be called with num_samples_per_antenna = NUM_SAMPS_PER_BLOCK_PER_ANT
//       and with num_antennas = NUM_SIG_PATHS/2 (dual pol is assumed)
{
  if (num_samples_per_antenna % 128)
  {
    printf("mwax_sum_powers_dual_pol: ERROR: number of complex samples per antenna should always be divisible by 128\n");
    return -1;
  }

  int nthreads = 128;  // all blocks will be 128 threads in size
  int nblocks = (int)num_samples_per_antenna/nthreads;  // 400 when 51,200 samples

  sum_powers_dual_pol_kernel<<<nblocks,nthreads,0,stream>>>(input,output,num_antennas,num_samples_per_antenna);

  return 0;
}



// Takes a block of (rows x columns) where columns are 8-bit time samples and rows is signal paths
// and re-orders it such that the time axis is extended by the factor "num_to_aggregate"
// and this current block's data are placed according to "aggregate_count".
// - samples are also promoted from 8-bit integers to floats
// - samples are also weighted by path according to "weights"
__global__ void aggregate_promote_and_weight_kernel(const float* weights, const char* input, float* output, unsigned rows, unsigned columns, int extended_row_length, int first_write_idx)
{
  // blockIdx.x is column (input time)
  // threadIdx.x is row (input path)
  float weight = weights[threadIdx.x];
  int idx_read = 2*(threadIdx.x*columns + blockIdx.x);  // x2 for complex input samples
  //int extended_row_length = columns*num_to_aggregate;   // length of aggregated rows
  //int first_idx_write = columns*aggregate_count;        // location that first input sample to be placed
  // write sequentially with blockIdx (time samples) and stride by the extended row length with threadIdx
  //int idx_write = 2*(first_write_idx + blockIdx.x + extended_row_length*threadIdx.x);
  float * out_address = output;
  //float * out_address = output + 2*(first_write_idx + blockIdx.x + extended_row_length*threadIdx.x);
  //output[idx_write] = (float)input[idx_read]*weight;     // real sample
  //output[idx_write+1] = (float)input[idx_read+1]*weight; // imag sample
  *out_address = (float)input[idx_read]*weight;     // real sample
  out_address++;
  *out_address = (float)input[idx_read+1]*weight; // imag sample

  return;
}

extern "C"
int mwax_aggregate_promote_and_weight(float* weights, char* input, float* output, unsigned rows, unsigned columns, unsigned num_to_aggregate, unsigned aggregate_count, cudaStream_t stream)
{
  int nblocks = (int)columns;  // input time samples
  int nthreads = (int)rows;    // input signal paths
  aggregate_promote_and_weight_kernel<<<nblocks,nthreads,0,stream>>>(weights,input,output,rows,columns,(columns*num_to_aggregate),(columns*aggregate_count));

  return 0;
}
