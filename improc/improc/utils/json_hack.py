from json import JSONEncoder

from datetime import datetime, timedelta
import dataclasses
import enum

# You... you wouldn't dare modify this... would you?
__ORIG = JSONEncoder.default

def patch_global_encoder():
    """ Overrides the default json encoder with some extended freeform jazz """
    def _default(self, o):
        if dataclasses.is_dataclass(o):
            return {"@type": type(o).__name__, **dataclasses.asdict(o)}
        elif isinstance(o, enum.Enum):
            return o.value
        elif isinstance(o, datetime):
            return o.timestamp()
        elif isinstance(o, timedelta):
            return o.total_seconds()
        elif (e := getattr(o.__class__, "to_json", None)) is not None:
            e(o)
        else:
            __ORIG(self, o)

    JSONEncoder.default = _default

def restore_original_encoder():
    """ Puts it back as long as you haven't messed with __ORIG """
    JSONEncoder.default = __ORIG
