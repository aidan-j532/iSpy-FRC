import time
import os
import logging
from typing import Literal

logger = logging.getLogger(__name__)

def pick_best_compressions_alg(cond: Literal["speed", "comp", "auto"] = "speed",):
    results = []
    
    DATA_SIZE = 512 * 1024  # 512 KB
    data = os.urandom(DATA_SIZE)

    # Test LZ4
    try:
        import lz4.frame

        start = time.perf_counter()
        compressed = lz4.frame.compress(data)
        elapsed = time.perf_counter() - start
        results.append(("LZ4", compressed, elapsed))
    except ImportError:
        logger.warning("LZ4 not installed")

    # Test ZStandard
    try:
        import zstandard as zstd

        for level in (1, 3, 6):
            compressor = zstd.ZstdCompressor(level=level)

            start = time.perf_counter()
            compressed = compressor.compress(data)
            elapsed = time.perf_counter() - start
            results.append((f"Zstd level {level}", compressed, elapsed))
    except ImportError:
        logger.warning("Zstandard not installed")

    if not results:
        raise RuntimeError("No compression algorithms available (did you not pip install iSpy-FRC, or pip install -e . (for a cloned repo)?)")

    if cond == "speed":
        results.sort(key=lambda x: x[2]) # speed
    elif cond == "comp":
        results.sort(key=lambda x: len(x[1])) # compression ratio
    elif cond == "auto":
        # Pick the fastest Zstd level that is reasonably close to LZ4.
        fastest = min(results, key=lambda x: x[2])
        best_compression = min(results, key=lambda x: len(x[1]))

        if best_compression[2] <= fastest[2] * 2:
            results = [best_compression]
        else:
            results = [fastest]

    best_algorithm, best_compressed, best_time = results[0]
    logger.info(
        f"Best compression algorithm: {best_algorithm} "
        f"(Size: {len(best_compressed) / 1024:.1f} KB, "
        f"Ratio: {len(data) / len(best_compressed):.2f}x, "
        f"Time: {best_time * 1000:.1f} ms)"
    )
    return best_algorithm, best_compressed

if __name__ == "__main__": # this would mess up my whole code, but because of the __name__ thuing it lets it work for only this file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
