#!/usr/bin/env python3
"""Deterministic validation corpus for the SentencePiece Unigram tokenizer.

This is offline tooling only: it never runs as part of the C runtime. It exists
so `tools/oracle_pocket_tokenizer.py` and the C side agree on *which* inputs are
being compared, without either of them carrying a 9,000-entry literal.

Every case is `bytes`, never `str`. The interesting half of this corpus is not
valid UTF-8 at all -- truncated multibyte sequences, overlongs, encoded
surrogates, lone continuation bytes -- and those are exactly the inputs that
exercise the normalizer rule the note calls trap 4: an invalid byte becomes one
U+FFFD, consuming *one* byte, so `ED A0 80` is three replacement characters and
not one. A corpus of `str` cannot express any of that, which is why the oracle
writes the input back out as hex.

The classes, per `.work/tokenizer-sentencepiece.md` ("Corpus"):

    A  curated sentences, per language (~40 each)
    B  Unicode stress: ligatures, fullwidth, combining vs precomposed, CJK,
       Cyrillic, Arabic, Devanagari, emoji, ZWJ, variation selectors, flags
    C  whitespace: doubles, leading/trailing, tab/newline/CRLF/formfeed, NBSP,
       a literal U+2581, empty, spaces only
    D  adversarial vocabulary: `<s>`, `</s>`, `<unk>`, `<pad>`, `<0x41>` as text
    E  byte-level fuzz: truncated prefixes of valid sequences, the invalid
       catalogue, 9,000 random byte strings
    F  one 100,000-character string

Determinism is the whole contract: same seed -> same corpus, byte for byte.
Class A-D are literals, class E/F draw from a single `random.Random(seed)` that
is consumed in a fixed order and is never touched by the language, so the fuzz
half is identical across the five languages and only the curated sentences
differ.

Usage (normally imported, but runnable for a quick look):
    uv run python tools/corpus_pocket.py --language italian --seed 1234
"""

from __future__ import annotations

import argparse
import random
import unicodedata
from typing import Iterator, NamedTuple

CORPUS_VERSION = 1

LANGUAGES = ("english", "german", "italian", "portuguese", "spanish")


class Case(NamedTuple):
    """One corpus entry: the class it came from and the raw input bytes."""

    category: str
    data: bytes


# --------------------------------------------------------------------------
# A. Curated sentences, per language.
#
# Each list mixes the things a TTS front end actually receives and that the
# unigram lattice segments differently per language: digits and decimals with a
# comma, dates, currency, URLs, e-mail, apostrophes, mixed case, hyphens, low
# quotes/guillemets, em dashes, ellipses.
# --------------------------------------------------------------------------

CURATED: dict[str, tuple[str, ...]] = {
    "english": (
        "Hello world.",
        "The meeting is at 9:30 on Tuesday, March 3rd, 2026.",
        "That will be $1,299.99 plus tax.",
        "She scored 98.6 out of 100 -- twice.",
        "Call +1 (415) 555-0132 before 5 p.m.",
        "Visit https://example.com/docs?page=2&lang=en for details.",
        "Write to jane.doe+tts@example.co.uk, please.",
        "It's Dr. O'Neill's third-quarter report.",
        "We can't, won't, and shouldn't.",
        "The CEO of NASA met the IMF's CFO.",
        "iPhone, eBay, LaTeX, macOS, JavaScript.",
        "A well-known, state-of-the-art, off-the-shelf solution.",
        "He said “stop” -- and then, well... nothing.",
        "Wait… what? Really…",
        "Chapter 7 — The Long Road — begins on page 143.",
        "Temperatures hit -12.5 °C overnight.",
        "3 x 4 = 12, and 10 / 4 = 2.5.",
        "Version 1.3.1 fixes issue #482.",
        "Room 4B, Building 12, 221B Baker Street.",
        "The 1920s, the '90s, and the 2000s.",
        "50% off, 2/3 full, 1⁄2 done.",
        "Q1 revenue rose 4.2% year-over-year.",
        "RSVP by 2026-09-30T18:00Z.",
        "MR. SMITH SHOUTED IN ALL CAPS.",
        "mixedCASE and MiXeD cAsE and lowercase.",
        "Read „The Trial“ in the original.",
        "He whispered «goodbye» and left.",
        "naive, naïve, resumé, résumé, café.",
        "coöperate, re-enter, re‐enter.",
        "1st, 2nd, 3rd, 4th, 21st, 102nd.",
        "The file is 3.5 GB; the disk has 1.2 TB free.",
        "Press Ctrl+C, then Cmd⌘+V.",
        "Latitude 37.7749, longitude -122.4194.",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z.",
        "0 1 2 3 4 5 6 7 8 9 10 100 1000 10000 100000.",
        "Hyphen-minus, en–dash, em—dash, minus−sign.",
        "Tokyo, Tōkyō, 東京.",
        "The password is hunter2; don't share it.",
        "Once upon a time, in a land far, far away, there lived a very small king.",
        "Okay.",
    ),
    "german": (
        "Hallo Welt.",
        "Die Sitzung beginnt am Dienstag, den 3. März 2026, um 9:30 Uhr.",
        "Das macht 1.299,99 € inklusive Mehrwertsteuer.",
        "Sie erreichte 98,6 von 100 Punkten — zweimal.",
        "Rufen Sie +49 (0)30 555 0132 vor 17 Uhr an.",
        "Besuchen Sie https://beispiel.de/hilfe?seite=2&lang=de für Details.",
        "Schreiben Sie an max.mustermann+tts@beispiel.co.de.",
        "Herrn Dr. O'Brien-Müllers Quartalsbericht liegt vor.",
        "Straße, Fußball, größer, heiß, Maß.",
        "Der Geschäftsführer der BaFin traf den CFO des IWF.",
        "Donaudampfschifffahrtsgesellschaftskapitän.",
        "Rechtsschutzversicherungsgesellschaften.",
        "Er sagte „stopp“ — und dann, nun ja … nichts.",
        "Moment mal … wie bitte? Wirklich …",
        "Kapitel 7 — Der lange Weg — beginnt auf Seite 143.",
        "Über Nacht sank die Temperatur auf -12,5 °C.",
        "3 × 4 = 12, und 10 : 4 = 2,5.",
        "Version 1.3.1 behebt Fehler Nr. 482.",
        "Zimmer 4B, Gebäude 12, Hauptstraße 221b.",
        "Die 1920er, die 90er und die 2000er.",
        "50 % Rabatt, 2/3 voll, 1⁄2 erledigt.",
        "Der Umsatz stieg im 1. Quartal um 4,2 %.",
        "Antwort bitte bis zum 30.09.2026 um 18:00 Uhr.",
        "HERR SCHMIDT RIEF IN GROSSBUCHSTABEN.",
        "gemischteSCHREIBWEISE und GeMiScHt und klein.",
        "Lesen Sie „Der Prozeß“ im Original.",
        "Er flüsterte «auf Wiedersehen» und ging.",
        "Äpfel, Öl, Übung, äußerst, öfter, übrig.",
        "E-Mail-Adresse, Kfz-Kennzeichen, S-Bahn-Station.",
        "1., 2., 3., 4., 21., 102.",
        "Die Datei ist 3,5 GB groß; die Platte hat 1,2 TB frei.",
        "Drücken Sie Strg+C, dann Cmd⌘+V.",
        "Breitengrad 52,5200, Längengrad 13,4050.",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z Ä Ö Ü.",
        "0 1 2 3 4 5 6 7 8 9 10 100 1000 10000 100000.",
        "Bindestrich, Gedanken–strich, Geviert—strich, Minus−zeichen.",
        "München, Muenchen, MUENCHEN.",
        "Das Passwort lautet hunter2; gib es nicht weiter.",
        "Es war einmal, in einem weit, weit entfernten Land, ein sehr kleiner König.",
        "Gut.",
    ),
    "italian": (
        "Ciao mondo.",
        "La riunione è martedì 3 marzo 2026 alle 9:30.",
        "Fanno 1.299,99 €, IVA inclusa.",
        "Ha totalizzato 98,6 su 100 — due volte.",
        "Chiami il +39 02 5550132 entro le 17.",
        "Visita https://esempio.it/guida?pagina=2&lang=it per i dettagli.",
        "Scrivi a mario.rossi+tts@esempio.co.it, per favore.",
        "È la relazione trimestrale del dott. Dell'Orto.",
        "Non possiamo, non vogliamo e non dovremmo.",
        "L'amministratore delegato dell'ENI ha incontrato il CFO del FMI.",
        "Po', qual è, perché, così, più, già, città, virtù.",
        "Un'idea all'avanguardia, pronta all'uso.",
        "Ha detto «basta» — e poi, beh… niente.",
        "Aspetta… come? Davvero…",
        "Capitolo 7 — La lunga strada — inizia a pagina 143.",
        "Stanotte si è scesi a -12,5 °C.",
        "3 × 4 = 12, e 10 : 4 = 2,5.",
        "La versione 1.3.1 risolve il problema n. 482.",
        "Stanza 4B, edificio 12, via Roma 221/b.",
        "Gli anni '20, gli anni '90 e i primi anni 2000.",
        "Sconto del 50%, 2/3 pieno, 1⁄2 fatto.",
        "Nel primo trimestre i ricavi sono saliti del 4,2%.",
        "Conferma entro il 30/09/2026 alle 18:00.",
        "IL SIGNOR ROSSI HA URLATO TUTTO IN MAIUSCOLO.",
        "maiuscoleMISTE e MaIuScOlE e minuscole.",
        "Leggi „Il processo“ in lingua originale.",
        "Sussurrò «addio» e se ne andò.",
        "perché, perchè, cioè, cioè.",
        "ex-moglie, anti-virus, video‐gioco.",
        "1º, 2º, 3º, 4ª, 21º, 102º.",
        "Il file pesa 3,5 GB; il disco ha 1,2 TB liberi.",
        "Premi Ctrl+C, poi Cmd⌘+V.",
        "Latitudine 41,9028, longitudine 12,4964.",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z È É Ò.",
        "0 1 2 3 4 5 6 7 8 9 10 100 1000 10000 100000.",
        "Trattino-breve, lineetta–media, lineetta—lunga, segno−meno.",
        "Torino, Tòrin, TORINO.",
        "La password è hunter2; non condividerla.",
        "C'era una volta, in un paese molto, molto lontano, un re piccolissimo.",
        "Va bene.",
    ),
    "portuguese": (
        "Olá, mundo.",
        "A reunião é na terça-feira, 3 de março de 2026, às 9:30.",
        "São 1.299,99 € com impostos incluídos.",
        "Ela obteve 98,6 em 100 — duas vezes.",
        "Ligue para +351 21 555 0132 antes das 17h.",
        "Visite https://exemplo.pt/ajuda?pagina=2&lang=pt para mais detalhes.",
        "Escreva para joao.silva+tts@exemplo.com.br, por favor.",
        "É o relatório trimestral do Dr. D'Almeida.",
        "Não podemos, não queremos e não deveríamos.",
        "O presidente da Petrobras encontrou o CFO do FMI.",
        "coração, informações, pão, avô, avó, único, acção.",
        "Uma solução de ponta, pronta a usar.",
        "Ele disse «pare» — e depois, bem… nada.",
        "Espera… o quê? A sério…",
        "Capítulo 7 — A longa estrada — começa na página 143.",
        "A temperatura desceu a -12,5 °C durante a noite.",
        "3 × 4 = 12, e 10 : 4 = 2,5.",
        "A versão 1.3.1 corrige o problema n.º 482.",
        "Sala 4B, edifício 12, Rua Augusta 221-B.",
        "Os anos 20, os anos 90 e o início dos anos 2000.",
        "50% de desconto, 2/3 cheio, 1⁄2 feito.",
        "No 1.º trimestre a receita subiu 4,2%.",
        "Confirme até 30/09/2026 às 18:00.",
        "O SENHOR SILVA GRITOU TUDO EM MAIÚSCULAS.",
        "maiúsculasMISTAS e MiStUrAdO e minúsculas.",
        "Leia „O Processo“ no original.",
        "Sussurrou “adeus” e foi-se embora.",
        "Pôr, por, pôde, pode, são, sao.",
        "guarda-chuva, anti-vírus, vídeo‐jogo.",
        "1.º, 2.º, 3.º, 4.ª, 21.º, 102.º.",
        "O ficheiro tem 3,5 GB; o disco tem 1,2 TB livres.",
        "Carregue em Ctrl+C e depois Cmd⌘+V.",
        "Latitude 38,7223, longitude -9,1393.",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z Á Ã Ç.",
        "0 1 2 3 4 5 6 7 8 9 10 100 1000 10000 100000.",
        "Traço-curto, traço–médio, travessão—longo, sinal−menos.",
        "São Paulo, Sao Paulo, SAO PAULO.",
        "A palavra-passe é hunter2; não a partilhes.",
        "Era uma vez, num país muito, muito distante, um rei pequenino.",
        "Está bem.",
    ),
    "spanish": (
        "Hola, mundo.",
        "La reunión es el martes 3 de marzo de 2026 a las 9:30.",
        "Son 1.299,99 € con impuestos incluidos.",
        "Obtuvo 98,6 sobre 100 — dos veces.",
        "Llame al +34 91 555 0132 antes de las 17:00.",
        "Visite https://ejemplo.es/ayuda?pagina=2&lang=es para más detalles.",
        "Escriba a juan.perez+tts@ejemplo.com.mx, por favor.",
        "Es el informe trimestral del Dr. O'Donnell.",
        "¡No podemos! ¿No quieres? ¡Claro que sí!",
        "El presidente del BBVA se reunió con el CFO del FMI.",
        "niño, año, señor, mañana, cigüeña, pingüino.",
        "Una solución de vanguardia, lista para usar.",
        "Dijo «basta» — y luego, bueno… nada.",
        "Espera… ¿cómo? ¿En serio…?",
        "Capítulo 7 — El largo camino — empieza en la página 143.",
        "Por la noche bajó a -12,5 °C.",
        "3 × 4 = 12, y 10 : 4 = 2,5.",
        "La versión 1.3.1 corrige el problema n.º 482.",
        "Sala 4B, edificio 12, Gran Vía 221 bis.",
        "Los años 20, los 90 y los primeros 2000.",
        "50 % de descuento, 2/3 lleno, 1⁄2 hecho.",
        "En el 1.er trimestre los ingresos subieron un 4,2 %.",
        "Confirme antes del 30/09/2026 a las 18:00.",
        "EL SEÑOR GARCÍA GRITÓ TODO EN MAYÚSCULAS.",
        "mayúsculasMEZCLADAS y MeZcLaDo y minúsculas.",
        "Lea „El proceso“ en su idioma original.",
        "Susurró “adiós” y se marchó.",
        "él, el, tú, tu, sí, si, más, mas.",
        "para-aguas, anti-virus, vídeo‐juego.",
        "1.º, 2.º, 3.º, 4.ª, 21.º, 102.º.",
        "El archivo ocupa 3,5 GB; el disco tiene 1,2 TB libres.",
        "Pulse Ctrl+C y luego Cmd⌘+V.",
        "Latitud 40,4168, longitud -3,7038.",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z Á Ñ Ü.",
        "0 1 2 3 4 5 6 7 8 9 10 100 1000 10000 100000.",
        "Guión-corto, raya–media, raya—larga, signo−menos.",
        "México, Mejico, MÉXICO.",
        "La contraseña es hunter2; no la compartas.",
        "Érase una vez, en un país muy, muy lejano, un rey muy pequeño.",
        "De acuerdo.",
    ),
}


# --------------------------------------------------------------------------
# B. Unicode stress.
# --------------------------------------------------------------------------

UNICODE_STRESS: tuple[str, ...] = (
    # ligatures and compatibility forms -- NFKC would fold these; the pinned
    # models normalize with `identity`, so they must fall through to bytes.
    "ﬁne",                     # fi ligature
    "ﬂour",                    # fl ligature
    "ﬀ, ﬃ, ﬄ",       # ff, ffi, ffl
    "Ǆ ǅ ǆ",         # DZ caron title/upper/lower
    "①②③④",     # circled digits
    "ⒶⒷⒸ",           # circled latin capitals
    "ⅰⅱⅲ ⅠⅡⅢ",  # roman numerals
    "⁵₂½⅓",     # superscript/subscript/vulgar fractions
    "Ｈｅｌｌｏ",          # fullwidth Hello
    "０１２３４",          # fullwidth digits
    "ｱｲｳｴｵ",          # halfwidth katakana
    "アイウエオ",          # fullwidth katakana
    "ㄱㄲㄴ ﾡﾢﾤ",   # hangul jamo vs halfwidth
    # combining vs precomposed -- byte-identical output is not expected, the
    # point is that neither form is silently converted into the other.
    "é vs é",
    "àáâã vs àáâã",
    "ñ ñ ö ö ǖ ǖ",
    "q̣̇ q̣̇",  # combining class reordering
    "กำ้",           # Thai with tone + sara am
    "ź́́́́́́́",  # stacked marks
    # scripts
    "日本語のテキストです。",
    "中文测试，简体与繁體。",
    "한국어 문장입니다.",
    "Привет, мир! Щука.",
    "Γεια σου κόσμε.",
    "مرحبا بالعالم",
    "שלום עולם",
    "नमस्ते दुनिया",
    "สวัสดีชาวโลก",
    "გამარჯობა",
    "աշխարհ",
    # bidi controls and invisible characters
    "before​after",            # zero width space
    "soft­hyphen",
    "‮reversed‬ text",
    "⁦isolate⁩",
    "﻿bom at the start",
    "word⁠joiner",
    # emoji
    "\U0001f600 \U0001f601 \U0001f602",
    "\U0001f44d\U0001f3fb \U0001f44d\U0001f3ff",           # skin tone modifiers
    "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466",  # ZWJ family
    "\U0001f3f3️‍\U0001f308",                    # ZWJ + VS16 flag
    "❤️ vs ❤︎",                        # VS16 vs VS15
    "\U0001f1ee\U0001f1f9 \U0001f1e9\U0001f1ea \U0001f1f5\U0001f1f9",  # flags
    "\U0001f1ee\U0001f1f9\U0001f1ee\U0001f1f9",            # adjacent flags
    "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",  # tag sequence
    "keycap 1️⃣ 2️⃣",
    # plane boundaries and rare codepoints
    "\U0001d11e \U0001d400\U0001d401",   # musical symbol, math bold
    "\U000104b0\U000104b1",              # Osage
    "\U0002a6b2\U0002b81d",              # CJK ext B/D
    "\U0010fffe\U0010ffff",              # last plane 16 noncharacters
    "� literal replacement char",
    "￿￾ noncharacters",
    "\U0001fbf0\U0001fbf9",              # segmented digits
    "mixed 日本語 and English and Русский in one line",
    "é\U0001f600中مa1 — ▂▁▃",
)


# --------------------------------------------------------------------------
# C. Whitespace.
#
# `remove_extra_whitespaces = 0` and only 0x20 is escaped to U+2581, so every
# other whitespace character is expected to reach byte fallback untouched.
# --------------------------------------------------------------------------

WHITESPACE: tuple[str, ...] = (
    "",
    " ",
    "  ",
    "   ",
    "          ",
    " leading",
    "trailing ",
    "  both  ",
    "ciao  mondo",
    "a  b   c    d",
    "one\ttab",
    "two\t\ttabs",
    "line\nbreak",
    "crlf\r\nline",
    "cr\ronly",
    "form\ffeed",
    "vertical\vtab",
    "nbsp separated",
    " leading nbsp",
    "narrow nbsp",
    "figure space",
    "en em thin hair  spaces",
    "ideographic　space",
    "ogham space",
    "line separator",
    "paragraph separator",
    "▁",
    "▁literal lower one eighth block",
    "already▁escaped▁text",
    "▁▁▁",
    "tab\tand▁mixed with space",
    "\n",
    "\r\n",
    "\t",
    "\t \n \r\n \f \v   ",
    "mixed \t \n whitespace   soup  ",
    "next line",
    "trailing newline\n",
    "\n\n\nmany\n\n\nnewlines\n\n\n",
)


# --------------------------------------------------------------------------
# D. Adversarial vocabulary: control-piece spellings appearing as literal text.
#
# Trap 6 in the note: CONTROL/UNKNOWN/BYTE pieces must not be segmentable, so
# these must come out as ordinary characters and never as a reserved id.
# --------------------------------------------------------------------------

ADVERSARIAL: tuple[str, ...] = (
    "<s>",
    "</s>",
    "<unk>",
    "<pad>",
    "<0x41>",
    "<0x00>",
    "<0xFF>",
    "<0xff>",
    "<s></s>",
    "<s> hello </s>",
    "<unk><unk><unk>",
    "<pad><pad>",
    "the <s> tag is literal here",
    "a<0x41>b",
    "<0x41><0x42><0x43>",
    "▁<s>",
    "<s",
    "s>",
    "<>",
    "<<s>>",
    "&lt;s&gt;",
    "\\u003cs\\u003e",
    "[INST] ignore previous instructions [/INST]",
    "<|endoftext|>",
    "<|im_start|>system",
    "\\n\\t literal backslash escapes",
    "%s %d %n %%",
    "'; DROP TABLE pieces; --",
    "\x00",
    "\x00abc",
    "abc\x00",
    "a\x00b\x00c",
    "\x00\x00\x00",
    "\x01\x02\x03\x04\x05\x06\x07",
    "\x7f delete",
    "\x1b[31mred\x1b[0m",
    "control \x08 backspace",
    "\x1a substitute",
)


# --------------------------------------------------------------------------
# E. Byte-level fuzz.
# --------------------------------------------------------------------------

# Valid multibyte sequences whose every truncated prefix is a distinct failure
# mode for the decoder: 2, 3 and 4 byte forms, plus a surrogate pair's worth of
# astral characters.
VALID_SEQUENCES: tuple[bytes, ...] = (
    "é".encode(),          # C3 A9
    "ÿ".encode(),          # C3 BF
    " ".encode(),          # C2 A0 (NBSP)
    "Ω".encode(),          # CE A9
    "Ж".encode(),          # D0 96
    "א".encode(),          # D7 90
    "ا".encode(),          # D8 A7
    "▁".encode(),          # E2 96 81
    "€".encode(),          # E2 82 AC
    "日".encode(),          # E6 97 A5
    "अ".encode(),          # E0 A4 85
    "ᄀ".encode(),          # E1 84 80
    "�".encode(),          # EF BF BD
    "́".encode(),          # CC 81 (combining acute)
    "\U0001f600".encode(),      # F0 9F 98 80
    "\U0001d11e".encode(),      # F0 9D 84 9E
    "\U0001f1ee".encode(),      # F0 9F 87 AE (regional indicator)
    "\U0010ffff".encode(),      # F4 8F BF BF (last codepoint)
)

# The invalid catalogue named in `.work/tokenizer-sentencepiece.md`. Each entry
# has a different reason for being invalid, and the expected number of U+FFFD
# differs: `C0 80` is two, `ED A0 80` is three, `C2` alone is one.
INVALID_CATALOGUE: tuple[bytes, ...] = (
    b"\xff",                    # never valid in UTF-8
    b"\xfe",                    # never valid in UTF-8
    b"\x80",                    # lone continuation byte
    b"\xc0\x80",                # overlong NUL
    b"\xc1\xbf",                # overlong 0x7F
    b"\xe0\x80\x80",            # overlong 3-byte
    b"\xed\xa0\x80",            # encoded high surrogate D800
    b"\xed\xbf\xbf",            # encoded low surrogate DFFF
    b"\xf4\x90\x80\x80",        # above U+10FFFF
    b"\xf5\x80\x80\x80",        # 4-byte lead beyond F4
    b"\xf8\x88\x80\x80\x80",    # 5-byte form
    b"\xc2",                    # truncated 2-byte
    b"\xe2\x96",                # truncated 3-byte
    b"\xf0\x9f\x99",            # truncated 4-byte
)

# Pool for the mixed random class: bytes that are individually interesting --
# structural ASCII, lead bytes, continuation bytes, and the ones that only ever
# appear inside U+2581.
MIXED_POOL: bytes = bytes(
    (
        0x00, 0x09, 0x0A, 0x0D, 0x20, 0x20, 0x20, 0x21, 0x2C, 0x2E,
        0x30, 0x39, 0x3C, 0x3E, 0x41, 0x5A, 0x61, 0x7A, 0x7F,
        0x80, 0x81, 0x8F, 0x9F, 0xA0, 0xA9, 0xBF,
        0xC0, 0xC1, 0xC2, 0xC3, 0xCC, 0xCE,
        0xE0, 0xE1, 0xE2, 0xE6, 0xED, 0xEF,
        0xF0, 0xF4, 0xF5, 0xF8, 0xFE, 0xFF,
        0x81, 0x96, 0x9F, 0x88,
    )
)

FUZZ_UNIFORM = 3000
FUZZ_ASCII = 3000
FUZZ_MIXED = 3000
FUZZ_MAX_LEN = 40

# F. one long string
LONG_STRING_CHARS = 100_000

# Character pool for the long string: mostly Latin text so the lattice has real
# work to do, with enough multibyte and whitespace to keep the byte path warm.
LONG_POOL: tuple[str, ...] = tuple(
    "abcdefghijklmnopqrstuvwxyz"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "          "
    ".,;:!?'\"-"
) + (
    "é", "è", "ü", "ñ", "ç", "ß", "à",
    "’", "—", "…", "«", "»", "€",
    "日", "П", "\U0001f600", "▁", "\t", "\n",
)


def _truncated_prefixes() -> Iterator[bytes]:
    """Every prefix of every valid sequence, bare and with ASCII context.

    The bare form checks end-of-buffer truncation; the `a...b` form checks that
    a truncated sequence in the middle of the buffer consumes exactly one byte
    and resynchronizes, instead of eating the byte that follows.
    """
    for seq in VALID_SEQUENCES:
        for n in range(1, len(seq) + 1):
            prefix = seq[:n]
            yield prefix
            yield b"a" + prefix + b"b"
            yield b" " + prefix + b" "


def _fuzz(rng: random.Random) -> Iterator[bytes]:
    """9,000 random byte strings: uniform, ASCII-biased and pool-biased."""
    for _ in range(FUZZ_UNIFORM):
        n = rng.randrange(0, FUZZ_MAX_LEN + 1)
        yield bytes(rng.randrange(0, 256) for _ in range(n))
    for _ in range(FUZZ_ASCII):
        n = rng.randrange(0, FUZZ_MAX_LEN + 1)
        out = bytearray()
        for _ in range(n):
            # 7 in 8 printable ASCII, the rest anywhere -- enough high bytes to
            # break a sequence, few enough that the lattice still sees words.
            if rng.randrange(8):
                out.append(rng.randrange(0x20, 0x7F))
            else:
                out.append(rng.randrange(0, 256))
        yield bytes(out)
    for _ in range(FUZZ_MIXED):
        n = rng.randrange(0, FUZZ_MAX_LEN + 1)
        yield bytes(rng.choice(MIXED_POOL) for _ in range(n))


def _long_string(rng: random.Random) -> str:
    """One string of exactly LONG_STRING_CHARS characters (not bytes)."""
    chunk = [rng.choice(LONG_POOL) for _ in range(LONG_STRING_CHARS)]
    text = "".join(chunk)
    assert len(text) == LONG_STRING_CHARS
    return text


def build_corpus(language: str, seed: int) -> list[Case]:
    """Return the full corpus for one language, deterministically.

    Duplicates are dropped, keeping first occurrence, so the case count is
    stable but never counts the same input twice. The RNG is created here and
    consumed in a fixed order, and the language only selects class A, so the
    fuzz half is identical across languages for a given seed.
    """
    if language not in CURATED:
        raise KeyError(f"no curated sentences for language {language!r}")

    cases: list[Case] = []
    seen: set[bytes] = set()

    def add(category: str, data: bytes) -> None:
        if data in seen:
            return
        seen.add(data)
        cases.append(Case(category, data))

    for text in CURATED[language]:
        add("A-curated", text.encode("utf-8"))
        # NFD alongside the curated (mostly NFC) form: same visible sentence,
        # different lattice, and it costs one line.
        nfd = unicodedata.normalize("NFD", text)
        if nfd != text:
            add("A-curated-nfd", nfd.encode("utf-8"))

    for text in UNICODE_STRESS:
        add("B-unicode", text.encode("utf-8"))
    for text in WHITESPACE:
        add("C-whitespace", text.encode("utf-8"))
    for text in ADVERSARIAL:
        add("D-adversarial", text.encode("utf-8"))

    for data in _truncated_prefixes():
        add("E-truncated", data)
    for data in INVALID_CATALOGUE:
        add("E-invalid", data)
        add("E-invalid", b"ok " + data + b" ok")

    rng = random.Random(seed)
    for data in _fuzz(rng):
        add("E-fuzz", data)

    add("F-long", _long_string(rng).encode("utf-8"))
    return cases


def category_counts(cases: list[Case]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language", default="italian", choices=LANGUAGES)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--show", type=int, default=0, help="print the first N cases")
    args = ap.parse_args()

    cases = build_corpus(args.language, args.seed)
    total_bytes = sum(len(c.data) for c in cases)
    print(f"corpus v{CORPUS_VERSION} language={args.language} seed={args.seed}")
    print(f"  {len(cases)} cases, {total_bytes} bytes")
    for category, count in sorted(category_counts(cases).items()):
        print(f"  {category:18s} {count:6d}")
    for case in cases[: args.show]:
        print(f"  {case.category:18s} {case.data.hex()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
