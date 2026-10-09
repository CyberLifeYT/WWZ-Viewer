"""Payload decoders, one module per 1SER resource type.

Each decoder takes a parsed :class:`~extractor.ser.SerResource` and returns
plain data (dataclasses and numpy arrays). Decoders never read paks or write
files, and nothing here knows about rendering.
"""
