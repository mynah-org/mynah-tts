#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* ------------------------------------------------------------------ */
/*  Vocabulary: token-string → id                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    char  (*tokens)[64];   /* token strings */
    int   *ids;            /* parallel ids  */
    size_t count;
} vocab;

static void vocab_free(vocab *v) {
    free(v->tokens);
    free(v->ids);
    memset(v, 0, sizeof(*v));
}

/* Load "id\ttoken" TSV (first line may be a # comment). */
static int vocab_load_tsv(vocab *v, const char *path, char *err, size_t ec) {
    FILE *f = fopen(path, "r");
    if (!f) { snprintf(err, ec, "cannot open %s", path); return -1; }
    size_t cap = 128, n = 0;
    char (*toks)[64] = malloc(cap * sizeof(*toks));
    int  *ids        = malloc(cap * sizeof(int));
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = '\0';
        int id = atoi(line);
        char *tok = tab + 1;
        tok[strcspn(tok, "\r\n")] = '\0';
        if (n >= cap) {
            cap *= 2;
            toks = realloc(toks, cap * sizeof(*toks));
            ids  = realloc(ids,  cap * sizeof(int));
        }
        snprintf(toks[n], 64, "%s", tok);
        ids[n] = id;
        n++;
    }
    fclose(f);
    v->tokens = toks;
    v->ids    = ids;
    v->count  = n;
    return 0;
}

/* Load JSON array of strings: ["tok0","tok1",...] */
static int vocab_load_json(vocab *v, const char *path, char *err, size_t ec) {
    FILE *f = fopen(path, "r");
    if (!f) { snprintf(err, ec, "cannot open %s", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); snprintf(err, ec, "read error %s", path); return -1; }
    buf[sz] = '\0';
    fclose(f);

    size_t cap = 256, n = 0;
    char (*toks)[64] = malloc(cap * sizeof(*toks));
    int  *ids        = malloc(cap * sizeof(int));
    char *p = buf;
    while ((p = strchr(p, '"')) != NULL) {
        p++;
        char *end = p;
        while (*end && *end != '"') {
            if (*end == '\\') end++;   /* skip escaped char */
            end++;
        }
        size_t len = (size_t)(end - p);
        if (len > 0 && len < 63) {
            if (n >= cap) { cap *= 2; toks = realloc(toks, cap * sizeof(*toks)); ids = realloc(ids, cap * sizeof(int)); }
            memcpy(toks[n], p, len);
            toks[n][len] = '\0';
            ids[n] = (int)n;
            n++;
        }
        p = (*end == '"') ? end + 1 : end;
    }
    free(buf);
    v->tokens = toks;
    v->ids    = ids;
    v->count  = n;
    return 0;
}

static int vocab_lookup(const vocab *v, const char *tok) {
    int result = -1;
    for (size_t i = 0; i < v->count; i++)
        if (strcmp(v->tokens[i], tok) == 0) result = v->ids[i];
    return result;  /* last match wins (matches Python dict semantics) */
}

/* ------------------------------------------------------------------ */
/*  G2P dictionary: word → IPA phoneme string                         */
/* ------------------------------------------------------------------ */

typedef struct {
    char  (*words)[128];
    char  (*phonemes)[256];
    size_t count;
} g2p_dict;

static void g2p_free(g2p_dict *d) {
    free(d->words);
    free(d->phonemes);
    memset(d, 0, sizeof(*d));
}

/* Load "WORD  phoneme string" dictionary (skip ;;; comments). */
static int g2p_load(g2p_dict *d, const char *path, char *err, size_t ec) {
    FILE *f = fopen(path, "r");
    if (!f) { snprintf(err, ec, "cannot open %s", path); return -1; }
    size_t cap = 4096, n = 0;
    char (*ws)[128] = malloc(cap * sizeof(*ws));
    char (*ps)[256] = malloc(cap * sizeof(*ps));
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == ';' || line[0] == '#' || line[0] == '\n') continue;
        /* WORD  phonemes  (two spaces or tab separator) */
        char *sep = strstr(line, "  ");
        if (!sep) sep = strchr(line, '\t');
        if (!sep) continue;
        *sep = '\0';
        char *phon = sep + 1;
        while (*phon == ' ' || *phon == '\t') phon++;
        phon[strcspn(phon, "\r\n")] = '\0';
        /* Skip alternate pronunciations like WORD(1) */
        if (strchr(line, '(')) continue;
        if (n >= cap) { cap *= 2; ws = realloc(ws, cap * sizeof(*ws)); ps = realloc(ps, cap * sizeof(*ps)); }
        snprintf(ws[n], 128, "%s", line);
        snprintf(ps[n], 256, "%s", phon);
        n++;
    }
    fclose(f);
    d->words = ws;
    d->phonemes = ps;
    d->count = n;
    return 0;
}

static const char *g2p_lookup(const g2p_dict *d, const char *word) {
    /* Case-insensitive lookup: dictionary words are uppercase */
    for (size_t i = 0; i < d->count; i++) {
        if (strcasecmp(d->words[i], word) == 0) return d->phonemes[i];
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Per-language tokenizer state                                      */
/* ------------------------------------------------------------------ */

typedef enum {
    TOK_BYT5,          /* fr, it, vi, ko */
    TOK_IPA_G2P,       /* en, de, es, pt, hi */
    TOK_ARABIC_CHARS,  /* ar */
    TOK_JA_PHONEME,    /* ja */
    TOK_ZH_PHONEME,    /* zh */
} tok_type;

typedef struct {
    char      lang[8];
    tok_type  type;
    vocab     voc;
    g2p_dict  dict;
    int       has_dict;
    char      grapheme_prefix[8];  /* e.g. "#" for de/pt */
    int       grapheme_upper;      /* uppercase graphemes? */
    int       pad_with_space;      /* wrap with space tokens? */
} lang_tok;

struct mynah_tokenizer {
    lang_tok  langs[16];
    size_t    lang_count;
    char     *base_dir;   /* model_dir/tokenizer/ */
};

/* ------------------------------------------------------------------ */
/*  IPA phoneme string → token IDs (longest-match-first)              */
/* ------------------------------------------------------------------ */

static int ipa_parse(const vocab *voc, const char *ipa, int *out, size_t max_out) {
    size_t n = 0;
    const char *p = ipa;
    while (*p && n < max_out) {
        /* Try longest match first (max phoneme length ~4 chars) */
        int matched = 0;
        for (size_t len = 4; len >= 1 && !matched; len--) {
            char buf[8];
            if (strlen(p) < len) continue;
            memcpy(buf, p, len);
            buf[len] = '\0';
            int id = vocab_lookup(voc, buf);
            if (id >= 0) {
                out[n++] = id;
                p += len;
                matched = 1;
            }
        }
        if (!matched) {
            /* Skip unknown character (try as single UTF-8 codepoint) */
            unsigned char c = (unsigned char)*p;
            size_t skip = 1;
            if (c >= 0xC0) skip = (c >= 0xE0) ? (c >= 0xF0 ? 4 : 3) : 2;
            /* Try the multi-byte character as a token */
            char mbuf[8];
            memcpy(mbuf, p, skip);
            mbuf[skip] = '\0';
            int id = vocab_lookup(voc, mbuf);
            if (id >= 0) {
                out[n++] = id;
            }
            /* else: silently skip */
            p += skip;
        }
    }
    return (int)n;
}

/* ------------------------------------------------------------------ */
/*  Encoding functions                                                */
/* ------------------------------------------------------------------ */

static int encode_byt5(const char *text, int **out, size_t *count) {
    size_t len = strlen(text);
    int *ids = malloc((len + 1) * sizeof(int));
    if (!ids) return -1;
    size_t n = 0;
    for (size_t i = 0; i < len; i++)
        ids[n++] = (unsigned char)text[i] + 3;
    ids[n++] = 1;  /* EOS </s> */
    *out = ids;
    *count = n;
    return 0;
}

static int encode_ipa(const lang_tok *lt, const char *text, int **out, size_t *count) {
    /* Tokenize: split by spaces, look up each word in G2P dict,
     * fall back to per-character graphemes. */
    size_t cap = 256, n = 0;
    int *ids = malloc(cap * sizeof(int));
    if (!ids) return -1;

    if (lt->pad_with_space) {
        int sp = vocab_lookup(&lt->voc, " ");
        if (sp >= 0 && n < cap) ids[n++] = sp;
    }

    const char *p = text;
    int first_word = 1;
    while (*p) {
        /* Skip leading spaces */
        while (*p == ' ') p++;
        if (!*p) break;

        /* Extract word */
        char word[256];
        size_t wl = 0;
        while (*p && *p != ' ' && wl < 255) {
            /* Handle multi-byte UTF-8 */
            unsigned char c = (unsigned char)*p;
            size_t skip = 1;
            if (c >= 0xC0) skip = (c >= 0xE0) ? (c >= 0xF0 ? 4 : 3) : 2;
            for (size_t i = 0; i < skip && *p; i++) word[wl++] = *p++;
        }
        word[wl] = '\0';

        /* Add space between words */
        if (!first_word) {
            int sp = vocab_lookup(&lt->voc, " ");
            if (sp >= 0) { if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); } ids[n++] = sp; }
        }
        first_word = 0;

        /* Try G2P dictionary */
        const char *ipa = lt->has_dict ? g2p_lookup(&lt->dict, word) : NULL;
        if (ipa) {
            if (n + 64 >= cap) { cap = (n + 64) * 2; ids = realloc(ids, cap * sizeof(int)); }
            n += (size_t)ipa_parse(&lt->voc, ipa, ids + n, cap - n);
        } else {
            /* Fallback: character-by-character graphemes */
            const char *wp = word;
            while (*wp) {
                unsigned char c = (unsigned char)*wp;
                size_t skip = 1;
                if (c >= 0xC0) skip = (c >= 0xE0) ? (c >= 0xF0 ? 4 : 3) : 2;
                char gbuf[16];
                size_t gl = 0;
                if (lt->grapheme_prefix[0]) {
                    strcpy(gbuf, lt->grapheme_prefix);
                    gl = strlen(gbuf);
                }
                for (size_t i = 0; i < skip && *wp; i++) gbuf[gl++] = *wp++;
                gbuf[gl] = '\0';
                /* Try uppercase */
                if (lt->grapheme_upper && gl == 1 && gbuf[0] >= 'a' && gbuf[0] <= 'z')
                    gbuf[0] -= 32;
                int id = vocab_lookup(&lt->voc, gbuf);
                if (id < 0 && lt->grapheme_prefix[0]) {
                    /* Try without prefix */
                    id = vocab_lookup(&lt->voc, gbuf + strlen(lt->grapheme_prefix));
                }
                if (id >= 0) { if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); } ids[n++] = id; }
            }
        }
    }

    if (lt->pad_with_space) {
        int sp = vocab_lookup(&lt->voc, " ");
        if (sp >= 0) { if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); } ids[n++] = sp; }
    }

    *out = ids;
    *count = n;
    return 0;
}

static int encode_arabic(const lang_tok *lt, const char *text, int **out, size_t *count) {
    size_t cap = 256, n = 0;
    int *ids = malloc(cap * sizeof(int));
    if (!ids) return -1;
    /* Wrap with spaces */
    int sp = vocab_lookup(&lt->voc, " ");
    if (sp >= 0) ids[n++] = sp;
    const char *p = text;
    while (*p) {
        unsigned char c = (unsigned char)*p;
        size_t skip = 1;
        if (c >= 0xC0) skip = (c >= 0xE0) ? (c >= 0xF0 ? 4 : 3) : 2;
        char buf[8];
        memcpy(buf, p, skip);
        buf[skip] = '\0';
        int id = vocab_lookup(&lt->voc, buf);
        if (id >= 0) { if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); } ids[n++] = id; }
        p += skip;
    }
    if (sp >= 0) { if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); } ids[n++] = sp; }
    *out = ids;
    *count = n;
    return 0;
}

/* Japanese: text → katakana tokens + accent markers.
 * Full G2P requires open_jtalk; this basic version handles kana directly
 * and falls back to character-by-character for kanji. */
static int encode_japanese(const lang_tok *lt, const char *text, int **out, size_t *count) {
    size_t cap = 256, n = 0;
    int *ids = malloc(cap * sizeof(int));
    if (!ids) return -1;
    int sp = vocab_lookup(&lt->voc, " ");
    int zero = vocab_lookup(&lt->voc, "0");
    if (sp >= 0) ids[n++] = sp;
    /* Simple: emit each character as a token with default accent 0.
     * Hiragana → katakana (U+3041-U+3096 → +0x60). */
    const char *p = text;
    while (*p) {
        unsigned char c = (unsigned char)*p;
        size_t skip = 1;
        if (c >= 0xC0) skip = (c >= 0xE0) ? (c >= 0xF0 ? 4 : 3) : 2;
        /* Decode UTF-8 codepoint */
        unsigned int cp = 0;
        if (skip == 1) cp = c;
        else if (skip == 2) cp = ((c & 0x1F) << 6) | ((unsigned char)p[1] & 0x3F);
        else if (skip == 3) cp = ((c & 0x0F) << 12) | (((unsigned char)p[1] & 0x3F) << 6) | ((unsigned char)p[2] & 0x3F);
        else cp = ((c & 0x07) << 18) | (((unsigned char)p[1] & 0x3F) << 12) | (((unsigned char)p[2] & 0x3F) << 6) | ((unsigned char)p[3] & 0x3F);

        /* Hiragana → katakana */
        if (cp >= 0x3041 && cp <= 0x3096) cp += 0x60;

        /* Encode codepoint back to UTF-8 for vocab lookup */
        char buf[8];
        size_t bl = 0;
        if (cp < 0x80) { buf[bl++] = (char)cp; }
        else if (cp < 0x800) { buf[bl++] = (char)(0xC0 | (cp >> 6)); buf[bl++] = (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { buf[bl++] = (char)(0xE0 | (cp >> 12)); buf[bl++] = (char)(0x80 | ((cp >> 6) & 0x3F)); buf[bl++] = (char)(0x80 | (cp & 0x3F)); }
        else { buf[bl++] = (char)(0xF0 | (cp >> 18)); buf[bl++] = (char)(0x80 | ((cp >> 12) & 0x3F)); buf[bl++] = (char)(0x80 | ((cp >> 6) & 0x3F)); buf[bl++] = (char)(0x80 | (cp & 0x3F)); }
        buf[bl] = '\0';

        if (n + 2 >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
        /* Accent marker before each character */
        if (zero >= 0) ids[n++] = zero;
        int id = vocab_lookup(&lt->voc, buf);
        if (id >= 0) ids[n++] = id;
        p += skip;
    }
    if (sp >= 0) { if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); } ids[n++] = sp; }
    *out = ids;
    *count = n;
    return 0;
}

/* Chinese: text → pinyin phonemes + tone markers.
 * Full G2P requires jieba + pinyin dict; this basic version handles
 * common characters via a bundled mapping. */
static int encode_chinese(const lang_tok *lt, const char *text, int **out, size_t *count) {
    size_t cap = 256, n = 0;
    int *ids = malloc(cap * sizeof(int));
    if (!ids) return -1;
    int sp = vocab_lookup(&lt->voc, " ");
    if (sp >= 0) ids[n++] = sp;
    /* Without a pinyin dictionary, emit characters as-is if in vocab.
     * Full support requires exporting the pinyin table from NeMo. */
    const char *p = text;
    while (*p) {
        unsigned char c = (unsigned char)*p;
        size_t skip = 1;
        if (c >= 0xC0) skip = (c >= 0xE0) ? (c >= 0xF0 ? 4 : 3) : 2;
        char buf[8];
        memcpy(buf, p, skip);
        buf[skip] = '\0';
        if (n + 4 >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
        int id = vocab_lookup(&lt->voc, buf);
        if (id >= 0) ids[n++] = id;
        p += skip;
    }
    if (sp >= 0) { if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); } ids[n++] = sp; }
    *out = ids;
    *count = n;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Language registration                                             */
/* ------------------------------------------------------------------ */

static lang_tok *add_lang(mynah_tokenizer *t, const char *lang, tok_type type) {
    lang_tok *lt = &t->langs[t->lang_count++];
    memset(lt, 0, sizeof(*lt));
    snprintf(lt->lang, sizeof(lt->lang), "%s", lang);
    lt->type = type;
    return lt;
}

static int try_load_vocab(vocab *v, const char *dir, const char *name, char *err, size_t ec) {
    char path[512];
    /* Try JSON first, then TSV */
    snprintf(path, sizeof(path), "%s/%s_vocab.json", dir, name);
    if (vocab_load_json(v, path, err, ec) == 0) return 0;
    snprintf(path, sizeof(path), "%s/%s.tsv", dir, name);
    return vocab_load_tsv(v, path, err, ec);
}

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

mynah_tokenizer *mynah_tokenizer_open(const char *model_dir,
                                      char *error, size_t error_capacity) {
    mynah_tokenizer *t = calloc(1, sizeof(*t));
    if (!t) { snprintf(error, error_capacity, "out of memory"); return NULL; }
    size_t dlen = strlen(model_dir) + 16;
    t->base_dir = malloc(dlen);
    snprintf(t->base_dir, dlen, "%s/tokenizer", model_dir);

    char err[256];

    /* ByT5 languages: fr, it, vi, ko — no vocab file needed */
    add_lang(t, "fr", TOK_BYT5);
    add_lang(t, "it", TOK_BYT5);
    add_lang(t, "vi", TOK_BYT5);
    add_lang(t, "ko", TOK_BYT5);

    /* IPA G2P languages */
    struct { const char *lang; const char *voc_name; const char *dict_file;
             const char *prefix; int upper; int pad; } ipa_langs[] = {
        { "en", "english_phoneme",
          "dc7d60d6b15a4651b21c9ca2932b62c6_ipa_cmudict-0.7b_nv23.01.txt", "", 0, 0 },
        { "de", "german_phoneme",
          "cf01ab5c48c84f3282ef7888263361e5_de_nv230119.dict", "#", 0, 1 },
        { "es", "spanish_phoneme",
          "7dbc31751f224f2486090d59dc95b9f7_es_ES_nv230301.dict", "", 0, 1 },
        { "pt", "portuguese_Brazilian_phoneme",
          "05adc40366e149b69319acd4b28a4919_pt_br_prondict-v1.0.dict", "#", 1, 1 },
        { "hi", "hindi_phoneme",
          "339da71c54b046f98cbcf38ef6d4ff67_hindi_phoneme_merged_phoneme_dict.dict", "", 1, 1 },
    };
    for (size_t i = 0; i < sizeof(ipa_langs) / sizeof(ipa_langs[0]); i++) {
        lang_tok *lt = add_lang(t, ipa_langs[i].lang, TOK_IPA_G2P);
        snprintf(lt->grapheme_prefix, sizeof(lt->grapheme_prefix), "%s", ipa_langs[i].prefix);
        lt->grapheme_upper = ipa_langs[i].upper;
        lt->pad_with_space = ipa_langs[i].pad;
        if (try_load_vocab(&lt->voc, t->base_dir, ipa_langs[i].voc_name, err, sizeof(err)) != 0) {
            snprintf(error, error_capacity, "%s: %s", ipa_langs[i].lang, err);
            mynah_tokenizer_close(t);
            return NULL;
        }
        char dpath[512];
        snprintf(dpath, sizeof(dpath), "%s/%s", t->base_dir, ipa_langs[i].dict_file);
        if (g2p_load(&lt->dict, dpath, err, sizeof(err)) == 0) {
            lt->has_dict = 1;
        }
        /* Dictionary is optional — fall back to graphemes if missing */
    }

    /* Arabic */
    {
        lang_tok *lt = add_lang(t, "ar", TOK_ARABIC_CHARS);
        lt->pad_with_space = 1;
        if (try_load_vocab(&lt->voc, t->base_dir, "arabic_MSA_chartokenizer", err, sizeof(err)) != 0) {
            snprintf(error, error_capacity, "ar: %s", err);
            mynah_tokenizer_close(t);
            return NULL;
        }
    }

    /* Japanese */
    {
        lang_tok *lt = add_lang(t, "ja", TOK_JA_PHONEME);
        lt->pad_with_space = 1;
        if (try_load_vocab(&lt->voc, t->base_dir, "japanese_phoneme", err, sizeof(err)) != 0) {
            snprintf(error, error_capacity, "ja: %s", err);
            mynah_tokenizer_close(t);
            return NULL;
        }
    }

    /* Chinese */
    {
        lang_tok *lt = add_lang(t, "zh", TOK_ZH_PHONEME);
        lt->pad_with_space = 1;
        if (try_load_vocab(&lt->voc, t->base_dir, "mandarin_phoneme", err, sizeof(err)) != 0) {
            snprintf(error, error_capacity, "zh: %s", err);
            mynah_tokenizer_close(t);
            return NULL;
        }
    }

    return t;
}

void mynah_tokenizer_close(mynah_tokenizer *t) {
    if (!t) return;
    for (size_t i = 0; i < t->lang_count; i++) {
        vocab_free(&t->langs[i].voc);
        g2p_free(&t->langs[i].dict);
    }
    free(t->base_dir);
    free(t);
}

int mynah_tokenizer_encode(const mynah_tokenizer *t, const char *language,
                           const char *text, int **out_ids, size_t *out_count,
                           char *error, size_t error_capacity) {
    if (!t || !language || !text || !out_ids || !out_count) {
        snprintf(error, error_capacity, "invalid tokenizer arguments");
        return -1;
    }
    const lang_tok *lt = NULL;
    for (size_t i = 0; i < t->lang_count; i++) {
        if (strcmp(t->langs[i].lang, language) == 0) { lt = &t->langs[i]; break; }
    }
    if (!lt) {
        snprintf(error, error_capacity, "unsupported language: %s", language);
        return -1;
    }
    switch (lt->type) {
    case TOK_BYT5:         return encode_byt5(text, out_ids, out_count);
    case TOK_IPA_G2P:      return encode_ipa(lt, text, out_ids, out_count);
    case TOK_ARABIC_CHARS: return encode_arabic(lt, text, out_ids, out_count);
    case TOK_JA_PHONEME:   return encode_japanese(lt, text, out_ids, out_count);
    case TOK_ZH_PHONEME:   return encode_chinese(lt, text, out_ids, out_count);
    }
    snprintf(error, error_capacity, "unknown tokenizer type");
    return -1;
}

static const char *lang_codes[] = {
    "en", "fr", "it", "es", "de", "pt", "vi", "ko", "ja", "zh", "hi", "ar", NULL
};

const char *const *mynah_tokenizer_languages(const mynah_tokenizer *t) {
    (void)t;
    return lang_codes;
}
