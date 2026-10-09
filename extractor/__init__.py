"""Read-only extraction pipeline for World War Z (Swarm Engine) map data.

Layers, lowest first:

* ``pak_reader`` / ``catalog``: open the game's ``.pak`` archives (plain ZIP files)
  and find which archives belong to which map.
* ``ser``: the ``1SER`` binary container that wraps almost every resource, plus
  a reverse-engineering probe for payloads we have not decoded yet.

Nothing in this package writes to the game folder. Output only ever goes to the
configured output directory.
"""

__version__ = "0.1.0"
