# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagdata.condensation.data_distillation import DataDistillationTrainer
trainer = DataDistillationTrainer("/data1/scripts/FlagData/config/distillation_config.yaml")
trainer.load_data()  # setup dataset class here
trainer.fit()
