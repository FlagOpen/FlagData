# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import re


class CoreNLPAnaResult():
    """
    convert response to readable result
    """
    def __init__(self, ann_result):
        self.ann_result = ann_result
        self.tokens = []
        self.parse_tree = []
        self.bi_parse_tree = []
        self.basic_dep = []
        self.enhanced_dep = []
        self.enhanced_pp_dep = []
        self.entities = []
        self.openie = []
        self._extract_ann()

    def _extract_ann(self):
        """
        parse result
        """
        ann_dict = dict()
        if "sentences" in self.ann_result:
            for ann_sent in self.ann_result["sentences"]:
                self.tokens.append(ann_sent["tokens"])
                self.parse_tree.append(
                    re.sub(r"\s+", " ", ann_sent.get("parse", "")))
                self.bi_parse_tree.append(
                    re.sub(r"\s+", " ", ann_sent.get("binaryParse", "")))
                self.basic_dep.append(ann_sent.get("basicDependencies", ""))
                self.enhanced_dep.append(
                    ann_sent.get("enhancedDependencies", ""))
                self.enhanced_pp_dep.append(ann_sent.get(
                    "enhancedPlusPlusDependencies", ""))
                self.entities.append(ann_sent.get("entitymentions", ""))
                self.openie.append(ann_sent.get("openie", ""))
        else:
            self.tokens = self.ann_result["tokens"]
        return ann_dict
