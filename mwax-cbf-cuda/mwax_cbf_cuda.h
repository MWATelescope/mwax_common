#include <cuda_runtime.h>
#include <inttypes.h>

int mwax_byte_to_float(char* input, float* output, unsigned size, cudaStream_t stream);

char * mwax_cuda_get_device_name (int index);

int mwax_scalar_weight_complex(float weight, float* output, unsigned size, cudaStream_t stream);

int mwax_vector_weight_complex(float* weights, float* output, unsigned size, cudaStream_t stream);

int mwax_array_weight_complex(float* weights, float* output, unsigned rows, unsigned columns, cudaStream_t stream);

int mwax_complex_multiply(float* input, float* output, unsigned size, cudaStream_t stream);

int mwax_fast_complex_multiply(float* input, float* output, unsigned size, cudaStream_t stream);

int mwax_lookup_delay_gains(int32_t* delays, float complex* delay_lut, float complex* delay_gains, unsigned paths, unsigned fft_length, unsigned num_ffts, cudaStream_t stream);

int mwax_lookup_and_apply_delay_gains(int32_t* delays, float* delay_lut, float* input, unsigned paths, unsigned fft_length, unsigned num_ffts, cudaStream_t stream);

int mwax_transpose_to_xGPU(float complex* input, float complex* output, unsigned rows, unsigned columns, cudaStream_t stream);

int mwax_transpose_to_xGPU_and_weight(float* weights, float* input, float* output, unsigned rows, unsigned columns, cudaStream_t stream);

int mwax_detranspose_from_xGPU(float complex* input, float complex* output, unsigned rows, unsigned columns, cudaStream_t stream);

int mwax_detranspose_from_xGPU_and_weight(float* weights, float* input, float* output, unsigned rows, unsigned columns, cudaStream_t stream);

int xGPU_channel_average(float* input, float* output, unsigned num_visibility_samps_per_chan, unsigned num_input_channels, unsigned fscrunch_factor, cudaStream_t stream);

int xGPU_channel_average_shift_and_scale(float* input, float* output, unsigned num_visibility_samps_per_chan, unsigned num_input_channels, unsigned fscrunch_factor, float scale_factor, cudaStream_t stream);

int mwax_beamform_summation(float* input, float* output, unsigned num_antennas,  unsigned num_samples_per_antenna, cudaStream_t stream);

int mwax_sum_powers_dual_pol (float* input, float* output, unsigned num_antennas, unsigned num_samples_per_antenna, cudaStream_t stream);

int mwax_aggregate_promote_and_weight(float* weights, char* input, float* output, unsigned rows, unsigned columns, unsigned num_to_aggregate, unsigned aggregate_count, cudaStream_t stream);
