/* Dispatcher for pre-generated RNNoise tables. */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "denoise.h"

#if RNNOISE_SAMPLE_RATE == 48000
#include "rnnoise_tables_48k.c"
#elif RNNOISE_SAMPLE_RATE == 8000
#include "rnnoise_tables_8k.c"
#else
#error "Unsupported RNNOISE_SAMPLE_RATE for rnnoise_tables"
#endif

