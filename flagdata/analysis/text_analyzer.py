# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import requests
import os
from nltk import Tree
from subprocess import Popen
import time
import shlex
import multiprocessing
from urllib import parse
from .utils.ana_result import CoreNLPAnaResult
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class CoreNLPAnalyzer:
    """
    An analyzer client which re-format CoreNLP server's response
    to make it readable
    """
    def __init__(self, url=None, lang="en", annotators="",
                 corenlp_dir=None, local_port=9000, max_mem=4,
                 threads=multiprocessing.cpu_count(), timeout=150000):
        if url:
            self.url = url.rstrip("/")
        self.annotators_list = \
            "tokenize,ssplit,pos,lemma,ner,parse,depparse,openie".split(",")
        self.lang = lang
        self.corenlp_subprocess = None
        self.timeout = timeout
        if annotators and self._check_annotators_format(annotators):
            self.annotators = self._reorganize_annotators(annotators)
        else:
            self.annotators = ",".join(self.annotators_list)
        print(f"annotators: {self.annotators}")

        if corenlp_dir:
            if not os.path.exists(corenlp_dir):
                raise OSError("please check corenlp local path is correct! ")
            if self._launch_local_server(corenlp_dir, local_port, max_mem, threads):
                self.url = f"http://127.0.0.1:{local_port}"
                self._request_corenlp(data="", annotators=self.annotators)

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        """
        cleanup corenlp server's process
        """
        if self.corenlp_subprocess:
            self.corenlp_subprocess.kill()
            self.corenlp_subprocess.wait()

    def __del__(self):
        """
        cleanup corenlp server's process
        """
        if self.corenlp_subprocess:
            self.corenlp_subprocess.kill()
            self.corenlp_subprocess.wait()

    def _check_annotators_format(self, annotators):
        annotators = annotators.split(",")
        for i in annotators:
            if i not in self.annotators_list:
                return False
        return True

    def _reorganize_annotators(self, annotators):
        """
        re-organize annotators to standard format
        """
        annotators = annotators.split(",")
        max_annotator_idx = 0
        for annotator in annotators:
            annotator_idx = self.annotators_list.index(annotator)
            if annotator_idx > max_annotator_idx:
                max_annotator_idx = annotator_idx
        annotators = ",".join(self.annotators_list[:max_annotator_idx + 1])
        return annotators

    def _check_server_status(self):
        """
        check server's response first
        """
        if requests.get(self.url, verify=False).status_code != 200:
            raise ConnectionError(
                "please check your network connection, or the corenlp server is started before launching!")

    @staticmethod
    def _deal_path_suffix(path):
        """
        convert path
        """
        if "\\" in path:
            path = path.rstrip("\\") + "\\"
        else:
            path = path.rstrip("/") + "/"
        return path

    def _launch_local_server(self, corenlp_dir, port, max_mem, threads):
        """
        launch a corenlp server by subprocess
        """
        corenlp_dir = self._deal_path_suffix(os.path.abspath(corenlp_dir))
        tmp_dir = "tmp"
        if not os.path.exists("tmp"):
            os.mkdir(tmp_dir)
        try:
            os.system("java -version")
        except:
            raise AssertionError("Java is required to launch corenlp server! ")
        cmd = f'java -Djava.io.tmpdir={tmp_dir} -mx{max_mem}g ' + \
            f'-cp "{corenlp_dir}*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer ' + \
            f'-threads {threads} -port {port} -timeout 150000 -lazy false'
        print(cmd)
        cmd = shlex.split(cmd)
        self.corenlp_subprocess = Popen(cmd)
        time.sleep(1)
        return True

    def _request_corenlp(self, data, annotators):
        """
        sent requests
        """
        params = {"properties": '{"annotators": "%s"}' %
                  annotators, "pipelineLanguage": self.lang}
        res = requests.post(url=self.url, params=params, data=parse.quote(
            data), timeout=self.timeout, verify=False)
        ann_result = res.json()
        return ann_result

    def analyze(self, data: str):
        """
        analyze input data
        """
        res = self._request_corenlp(data, self.annotators)
        ana_result = CoreNLPAnaResult(res)
        return ana_result

    def tokenize(self, data, ssplit=True):
        """
        tokenize, if not ssplit, concat result to a single list
        """
        if ssplit:
            annotators = "tokenize,ssplit"
        else:
            annotators = "tokenize"
        ann_result = self._request_corenlp(data, annotators)
        if ssplit:
            annotation = [[token["word"] for token in sent["tokens"]]
                          for sent in ann_result["sentences"]]
        else:
            annotation = [token["word"] for token in ann_result["tokens"]]
        return annotation

    def pos_tag(self, data):
        """
        part-of-speech tagging
        """
        annotators = "tokenize,ssplit,pos"
        ann_result = self._request_corenlp(data, annotators)
        annotation = [[token["pos"] for token in sent["tokens"]]
                      for sent in ann_result["sentences"]]
        return annotation

    def ner(self, data):
        """
        name entity recognition
        """
        annotators = "tokenize,ssplit,pos,ner"
        ann_result = self._request_corenlp(data, annotators)
        annotation = []
        for sent in ann_result["sentences"]:
            sent_ner = []
            if "entitymentions" in sent:
                for entity in sent["entitymentions"]:
                    span = (entity["characterOffsetBegin"],
                            entity["characterOffsetEnd"])
                    ner = entity["ner"]
                    ner_entity = entity["text"]
                    sent_ner.append({(ner_entity, span): ner})
            annotation.append(sent_ner)
        return annotation

    @staticmethod
    def pretty_print_tree(tree):
        """
        re-format parse tree
        """
        Tree.fromstring(tree).pretty_print()

    def close(self):
        if self.corenlp_subprocess:
            self.corenlp_subprocess.kill()
            self.corenlp_subprocess.wait()
