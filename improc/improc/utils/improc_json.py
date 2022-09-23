from json import JSONEncoder, JSONDecoder

from functools import singledispatch

import dataclasses
import enum

__ORIG = JSONEncoder.default

def monkeypatch_global_encoder():
    """ Overrides the default json encoder with some extended jazz """
    def _default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, enum.Enum):
            return o.value
        elif (e := getattr(o.__class__, "to_json", None)) is not None:
            e(o)
        else:
            _default.default(self, o)

    _default.default = __ORIG      # Save unmodified default. Yes this is allowed for some reason.
    JSONEncoder.default = _default # Replace it.

def restore_original_encoder():
    """ Puts it back as long as you haven't messed with __ORIG """
    JSONEncoder.default = __ORIG
