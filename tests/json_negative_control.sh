#!/bin/sh
# Negative control for tests/test_json.c.
#
# A suite that has only ever passed is not evidence. It can be a suite that
# exercises nothing, or one whose assertions are all vacuously true, and the
# only way to tell from the outside is to break the thing it watches and check
# that it notices.
#
# So: four deliberate defects in src/json.c, one per property the parser
# promises, each built and run in a scratch directory that the repository never
# sees. Every one of them MUST make the suite fail. A break that slips through
# is reported by name, and this script exits non-zero -- which means the suite,
# not the parser, is what needs fixing.
#
#   1  UTF-8 validation   accept raw CESU-8 surrogates (ED A0 80)
#   2  surrogate pairing   accept a second high surrogate as the low half
#   3  document framing    stop refusing trailing content after the root value
#   4  duplicate keys      let the LAST duplicate win instead of the first
#
# Usage: sh tests/json_negative_control.sh        (or: make json-negative-control)
set -u

CC="${CC:-cc}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

CFLAGS_TEST="-std=c11 -Wall -Wextra -O1 -I$WORK"

copy_sources() {
    cp "$ROOT/src/json.c" "$ROOT/src/json.h" "$WORK/"
    cp "$ROOT/server/http_util.c" "$ROOT/server/http_util.h" "$WORK/"
    cp "$ROOT/tests/test_json.c" "$WORK/"
}

build() {   # -> $WORK/suite
    rm -f "$WORK/suite"
    # shellcheck disable=SC2086
    $CC $CFLAGS_TEST "$WORK/test_json.c" "$WORK/json.c" "$WORK/http_util.c" \
        -lm -lpthread -o "$WORK/suite" 2>"$WORK/cc.log"
}

# Apply one named break to $WORK/json.c. Refuses to "succeed" if the text it
# expected is not there -- a patch that silently matched nothing would look
# exactly like a defect the suite caught, which is the one way this script
# could lie.
patch_break() {
    python3 - "$WORK/json.c" "$1" <<'PYEOF'
import sys
path, which = sys.argv[1], sys.argv[2]
s = open(path).read()

BREAKS = {
    # 1. The UTF-8 validator stops excluding the surrogate block, so a raw
    #    ED A0 80 (CESU-8 U+D800) is accepted as if it were a character.
    "utf8": (
        "else if (b0 == 0xEDu)                { need = 3u; lo = 0x80u; hi = 0x9Fu; }",
        "else if (b0 == 0xEDu)                { need = 3u; lo = 0x80u; hi = 0xBFu; }",
    ),
    # 2. The low half of a surrogate pair is checked against the whole
    #    surrogate block instead of the low range, so \\uD83D\\uD83D passes and
    #    assembles into a nonsense codepoint.
    "surrogate": (
        "if (low < 0xDC00u || low > 0xDFFFu) {",
        "if (low < 0xD800u || low > 0xDFFFu) {",
    ),
    # 3. A document stops being one value: anything after the root is ignored,
    #    which is how a second request smuggled into one body goes unnoticed.
    "trailing": (
        "    skip_ws(&s);\n    if (s.pos < s.len) {\n        char seen[16];",
        "    skip_ws(&s);\n    if (0) {\n        char seen[16];",
    ),
    # 4. Duplicate keys resolve to the LAST occurrence rather than the first,
    #    so a key appended to a body overrides the key already in it.
    "duplicate": (
        """    size_t cursor = 0;
    mynah_json_value key, value;
    while (mynah_json_object_next(object, &cursor, &key, &value) == 0) {
        /* FIRST MATCH WINS -- see src/json.h on duplicate keys. */
        if (name_equals(object, key.start, key.end, name)) {
            *out = value;
            return 0;
        }
    }
    return -1;""",
        """    size_t cursor = 0;
    int found = 0;
    mynah_json_value key, value;
    while (mynah_json_object_next(object, &cursor, &key, &value) == 0) {
        if (name_equals(object, key.start, key.end, name)) {
            *out = value;
            found = 1;
        }
    }
    return found ? 0 : -1;""",
    ),
}

old, new = BREAKS[which]
if s.count(old) != 1:
    sys.stderr.write(
        "negative control '%s' matched %d times, expected 1 -- the parser moved "
        "under the patch and this script must be updated before it can be "
        "believed\n" % (which, s.count(old)))
    sys.exit(2)
open(path, "w").write(s.replace(old, new))
PYEOF
}

describe() {
    case "$1" in
        utf8)      echo "UTF-8 validation accepts raw CESU-8 surrogates" ;;
        surrogate) echo "a high surrogate is accepted as the low half of a pair" ;;
        trailing)  echo "trailing content after the root value is ignored" ;;
        duplicate) echo "the LAST duplicate key wins instead of the first" ;;
    esac
}

echo "== baseline: the suite must PASS on an unmodified parser"
copy_sources
if ! build; then
    echo "FAILED: the unmodified suite does not compile" >&2
    cat "$WORK/cc.log" >&2
    exit 2
fi
if ! "$WORK/suite" >"$WORK/out.log" 2>&1; then
    echo "FAILED: the unmodified suite does not pass" >&2
    tail -20 "$WORK/out.log" >&2
    exit 2
fi
echo "   ok: $(tail -1 "$WORK/out.log")"

missed=0
for name in utf8 surrogate trailing duplicate; do
    echo
    echo "== break '$name': $(describe "$name")"
    copy_sources
    if ! patch_break "$name"; then
        echo "FAILED: could not apply the break" >&2
        missed=$((missed + 1))
        continue
    fi
    if ! build; then
        echo "FAILED: the broken parser does not compile -- the break is not a" >&2
        echo "        behaviour change, so it proves nothing" >&2
        cat "$WORK/cc.log" >&2
        missed=$((missed + 1))
        continue
    fi
    if "$WORK/suite" >"$WORK/out.log" 2>&1; then
        echo "NOT CAUGHT: the suite still passes with this defect in place" >&2
        missed=$((missed + 1))
    else
        echo "   caught. The suite reported:"
        grep '^FAIL' "$WORK/out.log" | head -4 | sed 's/^/     /'
        echo "     $(tail -1 "$WORK/out.log")"
    fi
done

echo
if [ "$missed" -ne 0 ]; then
    echo "negative control: $missed of 4 defects were NOT caught" >&2
    exit 1
fi
echo "negative control: 4 of 4 deliberate defects caught"
