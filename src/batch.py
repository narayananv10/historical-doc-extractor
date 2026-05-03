"""Batch runner: process a folder of images, write catalogue.csv.

Sequential processing (the per-document Claude vision call dominates;
multiprocessing risks PyTorch fork hazards for marginal gain at 30-50 docs).

Usage:
    python -m src.batch data/raw/loc/ -o catalogue.csv
"""

# TODO: implement CLI + main loop
