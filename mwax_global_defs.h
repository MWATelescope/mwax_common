/**
 * @file mwax_global_defs.h
 * @author Ian Morrison, Greg Sleap
 * @date 30 Jul 2018
 * @brief This is common defines for all mwax C code
 *
 */

//#ifndef MWAX_GLOBAL_DEFS_H_INCLUDED // This prevents linker errors due to multiple translation units using this
//#define MWAX_GLOBAL_DEFS_H_INCLUDED

int is_mwax_mode_valid(const char *mode);
int is_mwax_mode_correlator(const char *mode);
int is_mwax_mode_vcs(const char *mode);
int is_mwax_mode_no_capture(const char *mode);
int is_mwax_mode_quit(const char *mode);


//
// Defined parameters (not currently selectable by the user)
//
#define CORRELATOR_NFFTS 10                                         // Each subfile block of NTIMESAMPLES samples is processed with NFFTS FFTs
#define BEAMFORMER_NFFTS 10

//
// Correlator keywords within the PSRDADA Header
//
#define HEADER_POPULATED "POPULATED"                                // 0=header being assembled; 1=header is ready for reading
#define HEADER_OBS_ID "OBS_ID"                                      // Observation ID (GPS start time of obs)
#define HEADER_SUBOBS_ID "SUBOBS_ID"                                // GPS start time of this 8 second sub observation
#define HEADER_MODE "MODE"                                          // Telescope mode: HW_LFILES, VOLTAGE_START, NO_CAPTURE, QUIT
#define HEADER_UTC_START "UTC_START"                                // UTC start time of observation (string)
#define HEADER_OBS_OFFSET "OBS_OFFSET"                              // Seconds since start of observation; e.g. 0,8,16,etc
#define HEADER_NBIT "NBIT"                                          // Bits per value (nominally 32 for float)
#define HEADER_NPOL "NPOL"                                          // Number of polarisations (nominally 2)
#define HEADER_NTIMESAMPLES "NTIMESAMPLES"                          // How many high time resolution (VCS) samples do we get (per sec?)
#define HEADER_NINPUTS "NINPUTS"                                    // Number of inputs (tiles*pols) which were received by the vcs machines
#define HEADER_NINPUTS_XGPU "NINPUTS_XGPU"                          // Number of inputs (tiles*pols) rounded up to the nearest 16 sent to xGPU
#define HEADER_APPLY_PATH_WEIGHTS "APPLY_PATH_WEIGHTS"              // Apply path weights prior to xGPU correlation?
#define HEADER_APPLY_PATH_DELAYS "APPLY_PATH_DELAYS"                // Apply path delays prior to xGPU correlation?
#define HEADER_APPLY_PATH_PHASE_OFFSETS "APPLY_PATH_PHASE_OFFSETS"  // Apply path phase offsets prior to xGPU correlation?
#define HEADER_APPLY_COARSE_DERIPPLE "APPLY_COARSE_DERIPPLE"        // Apply coarse channelisaer de-ripple prior to xGPU correlation?
#define HEADER_INT_TIME_MSEC "INT_TIME_MSEC"                        // Correlator mode: integrations every x milliseconds
#define HEADER_FSCRUNCH_FACTOR "FSCRUNCH_FACTOR"                    // How many 125 Hz ultra fine channels do we average together
#define HEADER_RAW_SCALE_FACTOR "RAW_SCALE_FACTOR"                  // Absolute scaling factor for output visibilities
#define HEADER_APPLY_VIS_WEIGHTS "APPLY_VIS_WEIGHTS"                // Apply calculated data occupancy weights to visibilities before outputting them?
#define HEADER_TRANSFER_SIZE "TRANSFER_SIZE"                        /* Number of bytes of data to expect in this subobservation including weights: \
                                                                    // == baselines * (finechannels+1) * (pols^2) * (real_bytes + imaginary_bytes) \
                                                                    // ==((NINPUTS_XGPU *(NINPUTS_XGPU+2))/8)*(NFINE_CHAN+1)*(NPOL^2)*(NBIT*2/8)*/
#define HEADER_PROJ_ID "PROJ_ID"                                    // Project code for this observation
#define HEADER_EXPOSURE_SECS "EXPOSURE_SECS"                        // Duration of the observation in seconds (always a factor of 8)
#define HEADER_COARSE_CHANNEL "COARSE_CHANNEL"                      // Coarse channel number (0..255)
#define HEADER_CORR_COARSE_CHANNEL "CORR_COARSE_CHANNEL"            // Correlator Coarse channel number (0..23)
#define HEADER_SECS_PER_SUBOBS "SECS_PER_SUBOBS"                    // How many seconds are in a sub observation
#define HEADER_UNIXTIME "UNIXTIME"                                  // Unix time in seconds
#define HEADER_UNIXTIME_MSEC "UNIXTIME_MSEC"                        // Milliseconds portion of Unix time (0-999)
#define HEADER_FINE_CHAN_WIDTH_HZ "FINE_CHAN_WIDTH_HZ"              // Width of fine channels post correlator (kHz)
#define HEADER_NFINE_CHAN "NFINE_CHAN"                              // How many fine channels per coarse channel
#define HEADER_BANDWIDTH_HZ "BANDWIDTH_HZ"                          // Bandwidth of a coarse channel
#define HEADER_SAMPLE_RATE "SAMPLE_RATE"                            // Sampling rate of a coarse channel (different to BW if oversampled)
#define HEADER_MC_IP "MC_IP"                                        // Multicast IP that the data was addressed to
#define HEADER_MC_PORT "MC_PORT"                                    // Multicast port that the data was addressed to
#define HEADER_MWAX_U2S_VERSION "MWAX_U2S_VER"                      // mwax_u2s version
#define HEADER_MWAX_DB2CORRELATE2DB_VERSION "MWAX_DB2CORR2DB_VER"   // mwax_db2correlate2db version
#define HEADER_MWAX_DB2MULTIBEAM2DB_VERSION "MWAX_DB2BEAM2DB_VER"   // mwax_db2multibeam2db version

//
// Beamformer keywords within the PSRDADA Header
//
#define INCOHERENT_BEAMS_MAX 10                             // Max number of incoherent beams we support
#define HEADER_NUM_INCOHERENT_BEAMS "NUM_INCOHERENT_BEAMS"  // Number of incoherent beams to form

#define COHERENT_BEAMS_MAX 10                               // Max number of coherent beams we support
#define HEADER_NUM_COHERENT_BEAMS "NUM_COHERENT_BEAMS"      // Number of coherent beams to form

#define HEADER_INCOHERENT_BEAM_01_CHANNELS "INCOHERENT_BEAM_01_CHANNELS"  // Specifies the channelisation FFT length for a requested beam
#define HEADER_INCOHERENT_BEAM_02_CHANNELS "INCOHERENT_BEAM_02_CHANNELS"  // (defaults to channels=1 if keyword not present for a requested beam)
#define HEADER_INCOHERENT_BEAM_03_CHANNELS "INCOHERENT_BEAM_03_CHANNELS"
#define HEADER_INCOHERENT_BEAM_04_CHANNELS "INCOHERENT_BEAM_04_CHANNELS"
#define HEADER_INCOHERENT_BEAM_05_CHANNELS "INCOHERENT_BEAM_05_CHANNELS"
#define HEADER_INCOHERENT_BEAM_06_CHANNELS "INCOHERENT_BEAM_06_CHANNELS"
#define HEADER_INCOHERENT_BEAM_07_CHANNELS "INCOHERENT_BEAM_07_CHANNELS"
#define HEADER_INCOHERENT_BEAM_08_CHANNELS "INCOHERENT_BEAM_08_CHANNELS"
#define HEADER_INCOHERENT_BEAM_09_CHANNELS "INCOHERENT_BEAM_09_CHANNELS"
#define HEADER_INCOHERENT_BEAM_10_CHANNELS "INCOHERENT_BEAM_10_CHANNELS"

#define HEADER_INCOHERENT_BEAM_01_TIME_INTEG "INCOHERENT_BEAM_01_TIME_INTEG"  // Specifies the desired output time integration for a requested beam
#define HEADER_INCOHERENT_BEAM_02_TIME_INTEG "INCOHERENT_BEAM_02_TIME_INTEG"  // i.e. time-scrunch factor, e.g. 10 means sum 10 powers samples per output
#define HEADER_INCOHERENT_BEAM_03_TIME_INTEG "INCOHERENT_BEAM_03_TIME_INTEG"  // Note: can only integrate up to the number of time samples in one aggregated input block
#define HEADER_INCOHERENT_BEAM_04_TIME_INTEG "INCOHERENT_BEAM_04_TIME_INTEG"  // (defaults to no integration if keyword not present for a requested beam)
#define HEADER_INCOHERENT_BEAM_05_TIME_INTEG "INCOHERENT_BEAM_05_TIME_INTEG"
#define HEADER_INCOHERENT_BEAM_06_TIME_INTEG "INCOHERENT_BEAM_06_TIME_INTEG"
#define HEADER_INCOHERENT_BEAM_07_TIME_INTEG "INCOHERENT_BEAM_07_TIME_INTEG"
#define HEADER_INCOHERENT_BEAM_08_TIME_INTEG "INCOHERENT_BEAM_08_TIME_INTEG"
#define HEADER_INCOHERENT_BEAM_09_TIME_INTEG "INCOHERENT_BEAM_09_TIME_INTEG"
#define HEADER_INCOHERENT_BEAM_10_TIME_INTEG "INCOHERENT_BEAM_10_TIME_INTEG"

extern char *incoherent_beam_time_integ_string[INCOHERENT_BEAMS_MAX];
extern char *incoherent_beam_fine_chan_string[INCOHERENT_BEAMS_MAX];
