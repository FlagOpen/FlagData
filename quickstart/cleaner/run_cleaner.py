# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagdata.cleaner.text_cleaner import DataCleaner
# safe importing of main module in multi-processing
if __name__ == "__main__":
    cleaner = DataCleaner("flagdata/cleaner/config.yaml")
    cleaner.clean()
