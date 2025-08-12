''' code to try open segy headers and look at them'''

import segyio
from pathlib import Path
import sys


def get_segy_readable_text(filepath: Path = None, verbose: bool=False):
    if filepath is None:
        return None
    
    with segyio.open(filepath, "r", ignore_geometry=True) as segyfile:
        text = segyfile.text
        opt_rows = segyfile.ext_headers

        textual_header = segyio.tools.wrap(text[0])
        optional_header = '\n'.join([segyio.tools.wrap(text[row]) for row in range(1, opt_rows+1)])
        
        all_text = textual_header + optional_header
        if verbose:
            print(all_text)
            print('number of optional textual header rows:', opt_rows) if verbose else None

        return '\n'.join(all_text)

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print("Usage: python script.py <number_in_csv>")
        sys.exit(1)

    number_in_csv = int(args[1])
    
    from utils import import_segys, DEFAULT_SEGYS
    segy_path = Path(import_segys(DEFAULT_SEGYS)[number_in_csv])
    print(segy_path)

    get_segy_readable_text(segy_path, verbose=True)




