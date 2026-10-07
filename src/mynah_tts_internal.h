#ifndef MYNAH_TTS_INTERNAL_H
#define MYNAH_TTS_INTERNAL_H

#include "mynah_tts.h"
#include "backend.h"
#include "weights.h"
#include "qmat.h"

struct mynah_tts_model {
    char *model_dir;
    mynah_tts_model_info info;
    mynah_weights *tts;
    mynah_weights *codec;
    mynah_backend *backend;
    mynah_qmat_cache *qcache;
    void *codec_cache;
    void *local_projection_cache;
};

/* mynah_tts_startup_mark: how many marks so far (0 before the first) and
 * whether the last one said serving. Lock-free; any thread. */
unsigned mynah_tts_startup_generation(int *serving);

#endif
