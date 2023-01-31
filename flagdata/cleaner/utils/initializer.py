# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from .filter import BasicCleaner
from .extractor import BasicExtractor


class ProcessorInitializer:
    def __init__(self, config, processor_type="filters"):
        self.config = config
        self.processor_type = processor_type
        processor_config = self.config.get(processor_type)
        if processor_config:
            self.processors = self._get_processor_class(
                processor_config.keys())
        else:
            self.processors = []
        self.num_processors = len(self.processors)

    def _get_processor_class(self, processor_names):
        if self.processor_type == "filters":
            processor_class_dict = BasicCleaner.get_subclasses()
        elif self.processor_type == "extractors":
            processor_class_dict = BasicExtractor.get_subclasses()
        processors = [(name, processor_class_dict[name])
                      for name in processor_names]
        return processors

    def __iter__(self):
        self.processors = iter(self.processors)
        self.current_processor_id = 0
        return self

    def __next__(self):
        if self.current_processor_id < self.num_processors:
            self.current_processor_id += 1
            name, processor = next(self.processors)
            processor_config = self.config[self.processor_type][name]
            processor = processor(**processor_config)
            return processor
        else:
            raise StopIteration
