# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import time
import logging


def count_global(global_counter, s, num_batches):
    global_counter.value += 1
    if global_counter.value % 10 == 0:
        e = time.perf_counter()
        print("\n", "*" * 20)
        print(
            f"|cost {e - s} seconds! | cleaned {global_counter.value} / {num_batches} batches......")
        print("*" * 20, "\n")


def count_positive(symbol_counter, line, cleaned_line):
    symbol_counter.value += 1
    if symbol_counter.value % 5000 == 0:
        print("ORIGINAL|", line, "|")
        print(">CLEANED|", cleaned_line, "|")
        print("Length: ", f"{len(line)} (original)",
              f"{len(cleaned_line)} (cleaned)")


def err_callback(e):
    logging.warning(f"MP ERROR: {e}")
