/**
 * @file mwax_global_defs.h
 * @author Ian Morrison, Greg Sleap
 * @date 30 Jul 2018
 * @brief This is common defines for all mwax C code
 *
 */

#pragma once

//
// Correlator COMMAND keywords
//
#define MWAX_COMMAND_CAPTURE "CAPTURE"                      // Normal operation
#define MWAX_COMMAND_IDLE "IDLE"                            // Do not create/archive FITS files
#define MWAX_COMMAND_QUIT "QUIT"                            // Finish current task, then exit

//
// Keywords within the PSRDADA Header
//
#define HEADER_POPULATED "POPULATED"                        // 0=header being assembled; 1=header is ready for reading
#define HEADER_OBS_ID "OBS_ID"                              // Observation ID (GPS start time of obs)
#define HEADER_SUBOBS_ID "SUBOBS_ID"                        // GPS start time of this 8 second sub observation
#define HEADER_COMMAND "COMMAND"                            // Command to run: CAPTURE, IDLE, QUIT
#define HEADER_UTC_START "UTC_START"                        // UTC start time of observation (string)
#define HEADER_OBS_OFFSET "OBS_OFFSET"                      // Seconds since start of observation; e.g. 0,8,16,etc
#define HEADER_NBIT "NBIT"                                  // Bits per value (nominally 32 for float)
#define HEADER_NPOL "NPOL"                                  // Number of polarisations (nominally 2)
#define HEADER_NTIMESAMPLES "NTIMESAMPLES"                  // How many high time resolution (VCS) samples do we get (per sec?)
#define HEADER_NINPUTS "NINPUTS"                            // Number of inputs (tiles*pols) which were received by the vcs machines
#define HEADER_NINPUTS_XGPU "NINPUTS_XGPU"                  // Number of inputs (tiles*pols) rounded up to the nearest 16 sent to xGPU
#define HEADER_METADATA_BEAMS "METADATA_BEAMS"              // How many beams to form?
#define HEADER_APPLY_WEIGHTS "APPLY_WEIGHTS"                // Does precorrelator apply weights?
#define HEADER_APPLY_DELAYS "APPLY_DELAYS"                  // Does precorrelator apply delays?
#define HEADER_INT_TIME_MSEC "INT_TIME_MSEC"                // Correlator mode: integrations every x milliseconds
#define HEADER_FSCRUNCH_FACTOR "FSCRUNCH_FACTOR"            // How many 125 Hz ultra fine channels do we average together
#define HEADER_TRANSFER_SIZE "TRANSFER_SIZE"                // Number of bytes of data to expect in this subobservation including weights:
                                                            // == baselines * (finechannels+1) * (pols^2) * (real_bytes + imaginary_bytes)
                                                            // ==((NINPUTS_XGPU *(NINPUTS_XGPU+2))/8)*(NFINE_CHAN+1)*(NPOL^2)*(NBIT*2/8)
#define HEADER_PROJ_ID "PROJ_ID"                            // Project code for this observation
#define HEADER_EXPOSURE_SEC "EXPOSURE_SEC"                  // Duration of the observation in seconds (always a factor of 8)
#define HEADER_COARSE_CHANNEL "COARSE_CHANNEL"              // Coarse channel number 
#define HEADER_SECS_PER_SUBOBS "SECS_PER_SUBOBS"            // How many seconds are in a sub observation
#define HEADER_UNIXTIME "UNIXTIME"                          // Unix time in seconds
#define HEADER_UNIXTIME_MSEC "UNIXTIME_MSEC"                // Milliseconds portion of Unix time (0-999)
#define HEADER_FINE_CHAN_WIDTH_HZ "FINE_CHAN_WIDTH_HZ"      // Width of fine channels post correlator (kHz)
#define HEADER_NFINE_CHAN "NFINE_CHAN"                      // How many fine channels per coarse channel
#define HEADER_BANDWIDTH_HZ "BANDWIDTH_HZ"                  // Bandwidth of a coarse channel
