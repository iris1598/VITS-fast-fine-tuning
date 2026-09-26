"""Text cleaners.

Every language backend (pyopenjtalk, indic_transliteration, num_thai, ...) is
imported *inside* the cleaner that needs it rather than at module import time.

This matters because ``text/__init__.py`` imports this module, and the training
config only ever uses one cleaner.  Previously a single missing optional
dependency — e.g. ``indic_transliteration`` or ``num_thai`` — made
``import text`` fail and therefore broke training entirely, even for a
Chinese-only model.  Now the failure is deferred to the moment you actually ask
for that language.

Each backend raises a :class:`MissingBackend` with an install hint.
"""

import re

__all__ = [
    "MissingBackend",
    "japanese_cleaners",
    "japanese_cleaners2",
    "korean_cleaners",
    "chinese_cleaners",
    "zh_ja_mixture_cleaners",
    "sanskrit_cleaners",
    "cjks_cleaners",
    "cjke_cleaners",
    "cjke_cleaners2",
    "thai_cleaners",
]

_INSTALL_HINTS = {
    "japanese": "pip install pyopenjtalk-plus",
    "korean": "pip install jamo ko_pron",
    "mandarin": "pip install pypinyin jieba cn2an",
    "sanskrit": "pip install indic_transliteration",
    "english": "pip install inflect unidecode eng_to_ipa",
    "thai": "pip install num_thai",
}


class MissingBackend(ImportError):
    """Raised when a text cleaner's language backend is not installed."""


def _require(module_name, language):
    """Import an optional backend or raise a helpful error."""
    try:
        return __import__(module_name, fromlist=["_"])
    except ImportError as exc:
        hint = _INSTALL_HINTS.get(language, "")
        raise MissingBackend(
            f"The '{language}' text frontend is not available: could not import "
            f"'{module_name}'. Install it with:  {hint}"
        ) from exc


# --- Japanese ---------------------------------------------------------------
def japanese_cleaners(text):
    japanese = _require("text.japanese", "japanese")
    text = japanese.japanese_to_romaji_with_accent(text)
    text = re.sub(r'([A-Za-z])$', r'\1.', text)
    return text


def japanese_cleaners2(text):
    return japanese_cleaners(text).replace('ts', 'ʦ').replace('...', '…')


# --- Korean -----------------------------------------------------------------
def korean_cleaners(text):
    korean = _require("text.korean", "korean")
    text = korean.latin_to_hangul(text)
    text = korean.number_to_hangul(text)
    text = korean.divide_hangul(text)
    text = re.sub(r'([\u3131-\u3163])$', r'\1.', text)
    return text


# --- Mandarin ---------------------------------------------------------------
def chinese_cleaners(text):
    '''Pipeline for Chinese text'''
    mandarin = _require("text.mandarin", "mandarin")
    text = text.replace("[ZH]", "")
    text = mandarin.number_to_chinese(text)
    text = mandarin.chinese_to_bopomofo(text)
    text = mandarin.latin_to_bopomofo(text)
    text = re.sub(r'([ˉˊˇˋ˙])$', r'\1。', text)
    return text


def zh_ja_mixture_cleaners(text):
    mandarin = _require("text.mandarin", "mandarin")
    japanese = _require("text.japanese", "japanese")
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: mandarin.chinese_to_romaji(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese.japanese_to_romaji_with_accent(
        x.group(1)).replace('ts', 'ʦ').replace('u', 'ɯ').replace('...', '…') + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


# --- Sanskrit ---------------------------------------------------------------
def sanskrit_cleaners(text):
    _require("text.sanskrit", "sanskrit")
    text = text.replace('॥', '।').replace('ॐ', 'ओम्')
    text = re.sub(r'([^।])$', r'\1।', text)
    return text


# --- Multilingual (CJKS / CJKE) --------------------------------------------
def cjks_cleaners(text):
    mandarin = _require("text.mandarin", "mandarin")
    japanese = _require("text.japanese", "japanese")
    korean = _require("text.korean", "korean")
    sanskrit = _require("text.sanskrit", "sanskrit")
    english = _require("text.english", "english")
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: mandarin.chinese_to_lazy_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: japanese.japanese_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: korean.korean_to_lazy_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[SA\](.*?)\[SA\]',
                  lambda x: sanskrit.devanagari_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english.english_to_lazy_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def cjke_cleaners(text):
    mandarin = _require("text.mandarin", "mandarin")
    japanese = _require("text.japanese", "japanese")
    korean = _require("text.korean", "korean")
    english = _require("text.english", "english")
    text = re.sub(r'\[ZH\](.*?)\[ZH\]', lambda x: mandarin.chinese_to_lazy_ipa(x.group(1)).replace(
        'ʧ', 'tʃ').replace('ʦ', 'ts').replace('ɥan', 'ɥæn') + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese.japanese_to_ipa(x.group(1)).replace('ʧ', 'tʃ').replace(
        'ʦ', 'ts').replace('ɥan', 'ɥæn').replace('ʥ', 'dz') + ' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: korean.korean_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]', lambda x: english.english_to_ipa2(x.group(1)).replace('ɑ', 'a').replace(
        'ɔ', 'o').replace('ɛ', 'e').replace('ɪ', 'i').replace('ʊ', 'u') + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def cjke_cleaners2(text):
    mandarin = _require("text.mandarin", "mandarin")
    japanese = _require("text.japanese", "japanese")
    korean = _require("text.korean", "korean")
    english = _require("text.english", "english")
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: mandarin.chinese_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: japanese.japanese_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: korean.korean_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english.english_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


# --- Thai -------------------------------------------------------------------
def thai_cleaners(text):
    thai = _require("text.thai", "thai")
    text = thai.num_to_thai(text)
    text = thai.latin_to_thai(text)
    return text
