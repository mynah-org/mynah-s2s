"""Drop the page cache of OUR large files (posix_fadvise DONTNEED, no root) before a big GPU load on GB10, where CUDA
allocations share the 121 GB with the page cache. Prints MemAvailable before/after. Only files we own are touched."""
import os
import sys
from pathlib import Path
from labpaths import HF_CACHE, LAB_HOME, MYNAH_HOME  # noqa: E402  (MYNAH_HOME / MYNAH_LAB_HOME, no hard-coded home layout)

ROOTS = [HF_CACHE, LAB_HOME, MYNAH_HOME / "data"]


def avail():
    return next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))


def main():
    before, n, uid = avail(), 0, os.getuid()
    for root in ROOTS:
        for p in root.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink() and p.stat().st_uid == uid and p.stat().st_size > 50 << 20:
                    fd = os.open(p, os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    os.close(fd)
                    n += 1
            except OSError:
                pass
    print(f"drop_cache: {n} files, MemAvailable {before} -> {avail()} MiB", file=sys.stderr)


if __name__ == "__main__":
    main()
