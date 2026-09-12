/* Engine lookup by the `engine` field of model.json.
 *
 * Its own translation unit so that adding an engine touches one list, and so
 * that neither engine has to know the other exists.
 */
#include <string.h>

#include "tts_engine.h"
#include "engine_magpie.h"
#include "engine_pocket.h"

const mynah_tts_engine *mynah_engine_lookup(const char *name) {
    if (name == NULL) return NULL;
    const mynah_tts_engine *const engines[] = {
        mynah_engine_magpie(),
        mynah_engine_pocket(),
    };
    for (size_t i = 0; i < sizeof(engines) / sizeof(engines[0]); ++i) {
        if (engines[i] != NULL && engines[i]->name != NULL &&
            strcmp(engines[i]->name, name) == 0) {
            return engines[i];
        }
    }
    return NULL;
}
