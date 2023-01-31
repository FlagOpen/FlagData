import unittest
from flagdata.analysis.text_analyzer import CoreNLPAnalyzer


class AnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.analyzer = CoreNLPAnalyzer(url="https://corenlp.run", lang="zh", annotators="parse")

    def test_analysis(self):
        data = "FlagData is a fast and extensible toolkit for data processing provided by BAAI. Enjoy yourself! "
        ann = self.analyzer.analyze(data)
        print(ann.parse_tree)

    def tearDown(self):
        self.analyzer.close()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AnalysisTestCase('test_analysis'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
