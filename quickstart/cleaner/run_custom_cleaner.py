# -*- coding:UTF-8 -*-
from flagdata.cleaner.text_cleaner import DataCleaner
from flagdata.cleaner.utils.filter import BasicCleaner


class MyFakeFilter(BasicCleaner):
    def __init__(self, **args):
        pass

    def clean_article(self, article):
        cleaned_article = article
        return cleaned_article


# safe importing of main module in multi-processing
if __name__ == "__main__":
    cleaner = DataCleaner("flagdata/cleaner/config.yaml")
    cleaner.clean()
