#!/usr/bin/env python3
"""Console entry point for mooncake_nvlink_proxy.

Runs the node-local GPU copy daemon used by the nvlink_proxy transport; all
arguments are passed through (see ``mooncake_nvlink_proxy --help``).
"""

import os
import stat
import sys


def main():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(package_dir, "mooncake_nvlink_proxy")
    if not os.path.exists(bin_path):
        sys.stderr.write(
            "mooncake_nvlink_proxy is not part of this build "
            "(build with -DUSE_NVLINK_PROXY=ON)\n"
        )
        return 1
    if not os.access(bin_path, os.X_OK):
        st = os.stat(bin_path)
        os.chmod(bin_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # exec so that signals (SIGTERM from the container runtime) reach the
    # daemon directly.
    os.execv(bin_path, [bin_path] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
