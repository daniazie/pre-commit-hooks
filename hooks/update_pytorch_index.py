import re
from argparse import ArgumentParser
from collections.abc import Sequence

import numpy as np


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='*', help="Filenames to fix")
    args = parser.parse_args(argv)

    retval = 0
    
    old_pattern = r"((index = \"pytorch-)(?:[cx]pu|cu\d+[^(130)])\")"
    replace = "\\2cu130\""

    for pyproj_file in args.filenames:
        with open(pyproj_file, "r") as file:
            data = file.read()

        new_pyproj = re.sub(old_pattern, replace, data)
            
        with open(pyproj_file, "w") as file:
            file.write(new_pyproj)

        retval |= np.int8(new_pyproj != data).item()
        
    return retval


if __name__ == "__main__":
    raise SystemExit(main())
