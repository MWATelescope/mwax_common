/**
 * @file mwax_global_defs.c
 * @author Greg Sleap
 * @date 4 Jul 2018
 * @brief This is the code that provides helper functions to mwax
 *
 */
#include <string.h>
#include "mwax_global_defs.h"

//
// define array of beam keyword strings
// Setup arrays of strings
//
char* incoherent_beam_time_integ_string[INCOHERENT_BEAMS_MAX] = {
    HEADER_INCOHERENT_BEAM_01_TIME_INTEG, HEADER_INCOHERENT_BEAM_02_TIME_INTEG,
    HEADER_INCOHERENT_BEAM_03_TIME_INTEG, HEADER_INCOHERENT_BEAM_04_TIME_INTEG,
    HEADER_INCOHERENT_BEAM_05_TIME_INTEG, HEADER_INCOHERENT_BEAM_06_TIME_INTEG,
    HEADER_INCOHERENT_BEAM_07_TIME_INTEG, HEADER_INCOHERENT_BEAM_08_TIME_INTEG,
    HEADER_INCOHERENT_BEAM_09_TIME_INTEG, HEADER_INCOHERENT_BEAM_10_TIME_INTEG
};
char* incoherent_beam_fine_chan_string[INCOHERENT_BEAMS_MAX] = { 
    HEADER_INCOHERENT_BEAM_01_CHANNELS, HEADER_INCOHERENT_BEAM_02_CHANNELS,
    HEADER_INCOHERENT_BEAM_03_CHANNELS, HEADER_INCOHERENT_BEAM_04_CHANNELS,
    HEADER_INCOHERENT_BEAM_05_CHANNELS, HEADER_INCOHERENT_BEAM_06_CHANNELS,
    HEADER_INCOHERENT_BEAM_07_CHANNELS, HEADER_INCOHERENT_BEAM_08_CHANNELS,
    HEADER_INCOHERENT_BEAM_09_CHANNELS, HEADER_INCOHERENT_BEAM_10_CHANNELS
};

//
// Telescope MODE keywords
//
const char* MWAX_MODE_V1_CORRELATOR  = "HW_LFILES";         // v1 Correlator observation
const char* MWAX_MODE_V1_VOLTAGE_CAPTURE = "VOLTAGE_START"; // v1 Voltage Capture observation
const char* MWAX_MODE_V1_VOLTAGE_BUFFER = "VOLTAGE_BUFFER"; // v1 Voltage Capture observation
const char* MWAX_MODE_V2_CORRELATOR = "MWAX_CORRELATOR";    // v2 Correlator observation
const char* MWAX_MODE_V2_VOLTAGE_CAPTURE = "MWAX_VCS";      // v2 Voltage Capture observation
const char* MWAX_MODE_NO_CAPTURE = "NO_CAPTURE";            // Subfiles will still be produced but no correlation or voltage capture
const char* MWAX_MODE_QUIT = "QUIT";                        // Finish current task, then exit 

//
// Telescope MODE helper functions
//
int is_mwax_mode_valid(const char* mode)
{
    return is_mwax_mode_correlator(mode) ||
           is_mwax_mode_vcs(mode)        ||
           is_mwax_mode_no_capture(mode) ||
           is_mwax_mode_quit(mode);
}

int is_mwax_mode_correlator(const char* mode)
{
    return strcmp(mode, MWAX_MODE_V1_CORRELATOR) == 0 ||            
           strcmp(mode, MWAX_MODE_V2_CORRELATOR) == 0;
}

int is_mwax_mode_vcs(const char* mode)
{
    return strcmp(mode, MWAX_MODE_V1_VOLTAGE_CAPTURE) == 0 || 
           strcmp(mode, MWAX_MODE_V1_VOLTAGE_BUFFER)  == 0 ||
           strcmp(mode, MWAX_MODE_V2_VOLTAGE_CAPTURE) == 0;
}

int is_mwax_mode_no_capture(const char* mode)
{
    return strcmp(mode, MWAX_MODE_NO_CAPTURE) == 0;
}

int is_mwax_mode_quit(const char* mode)
{
    return strcmp(mode, MWAX_MODE_QUIT) == 0;
}