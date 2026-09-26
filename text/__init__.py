""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.cleaners import MissingBackend
from text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, symbols, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      symbols: the symbol table to use (usually hps.symbols)
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  symbol_to_id = {s: i for i, s in enumerate(symbols)}
  clean_text = _clean_text(text, cleaner_names)
  for symbol in clean_text:
    if symbol not in symbol_to_id.keys():
      continue
    symbol_id = symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence


def cleaned_text_to_sequence(cleaned_text, symbols):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  symbol_to_id = {s: i for i, s in enumerate(symbols)}
  sequence = [symbol_to_id[symbol] for symbol in cleaned_text if symbol in symbol_to_id.keys()]
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def check_cleaners(cleaner_names):
  """Eagerly run the cleaners on a tiny probe string so that a missing language
  backend surfaces as a clear error before training starts."""
  missing = []
  for name in cleaner_names:
    if not hasattr(cleaners, name):
      missing.append(f"{name} (unknown cleaner — check data.text_cleaners)")
      continue
    try:
      getattr(cleaners, name)("[ZH]测试[JA]テスト[EN]test")
    except MissingBackend as exc:
      missing.append(f"{name}: {exc}")
    except Exception:  # noqa: BLE001 - cleaner-specific quirks are fine here
      pass
  return missing
