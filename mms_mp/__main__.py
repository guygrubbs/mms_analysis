#!/usr/bin/env python3
"""
Entry point for running mms_mp as a module:
    python -m mms_mp.cli
    python -m mms_mp
"""

import sys
from .cli import main

if __name__ == "__main__":
    # Check if 'cli' is specified as a subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Remove 'cli' from arguments and run CLI
        sys.argv.pop(1)
        main()
    else:
        # Default behavior - run CLI
        main()
