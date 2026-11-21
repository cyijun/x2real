"""policy-space CLI entry with the Wall-X vision-MLP padding workaround applied first (see wallx_patch.py).

Import order matters: policy_space.runtime.server must be imported BEFORE anything from wall_x, because it
snapshots the un-patched msgpack Packer/Unpacker at import time and wall_x calls msgpack_numpy.patch().
"""
import os, sys
import policy_space.runtime.server  # noqa: F401  (snapshot original msgpack classes first)
from policy_space.cli import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wallx_patch
wallx_patch.apply()
sys.exit(main())
