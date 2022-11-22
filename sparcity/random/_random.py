"""
random seed generator func
"""
import numpy as np

from sparcity.dev import typechecker, valchecker

def seedfixer(
        seed: int,
        length: int,
        index: int = 0,
        vmin: int = 0,
        vmax: int = 10000
) -> None:
    """
    function to fix random seed in a sequencial process with pseudo-random variables

    Parameters
    ----------
    seed: int
        random seed to fix. This parameter determines all subsequent random seeds.

    length: int
        length of random seed sequence. Expected to be a positive int.

    index: int, default: 0
        index of random seed to choose from the generated sequence to fix random state

    vim: int, default: 0
        minimum value in random seed sequence. vmin shold be smaller than vmax.

    vmax: int, default: 10000
        maximum value in radom seed sequence. vmax should be larger than vmin.

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity.random import seedfixer
    >>> seedfixer(seed=123, length=2)
    >>> np.random.randint(100)
    15
    >>> seedfixer(seed=123, length=2, index=0)
    >>> np.random.randint(100)
    15
    >>> seedfixer(seed=123, length=2, index=1)
    >>> np.random.randint(100)
    39
    """
    typechecker(seed, int, "seed")
    typechecker(length, int, "length")
    typechecker(index, int, "index")
    typechecker(vmin, int, "vmin")
    typechecker(vmax, int, "vmax")
    valchecker(0 < length)
    valchecker(0 <= index < length)
    msg = "In numpy, seed must be between 0 and 2 ** 32 -1"
    valchecker(0 <= seed <= 2 ** 32 - 1, msg)
    valchecker(0 <= vmin <= 2 ** 32 - 1, msg)
    valchecker(0 <= vmax <= 2 ** 32 - 1, msg)
    valchecker(vmin < vmax)

    np.random.seed(seed)
    seeds = np.random.randint(vmin, vmax, length)
    np.random.seed(seeds[index])
