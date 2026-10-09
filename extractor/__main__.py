"""Allow ``python -m extractor ...`` as an alias for the ``wwz-extract`` command."""

from extractor.cli import main

raise SystemExit(main())
