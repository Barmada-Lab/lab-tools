from . import core

import sys

try:
    args = core.parse_args(sys.argv)
except Exception as e:
    print(e)
    sys.exit(1)

core.hdf5ify(args.src, args.dest)
