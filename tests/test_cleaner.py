import unittest
from flagdata.cleaner.text_cleaner import DataCleaner
from tempfile import NamedTemporaryFile
import json


class CleanerTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_input_file = NamedTemporaryFile(mode="w")
        self.temp_output_file = NamedTemporaryFile(mode="w")
        self.cleaner = DataCleaner("./config/cleaner_config.yaml")
        self.cleaner.config["basic"]["input"] = self.temp_input_file.name
        self.cleaner.config["basic"]["output"] = self.temp_output_file.name
        self.setup_data()

    def setup_data(self):
        input_html = """
        <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta property="article:published_time" content="2022-12-05T00:52:04+08:00">
                <title>测试工具数据抽取能力</title>
            </head>
            <body>
                <p>2018年，北京智源人工智能研究院在科技部和北京市支持下，联合北京人工智能领域优势单位共建——汇集国际顶尖人工智能学者，聚焦核心技术与原始创新，旨在推动人工智能领域发展政策、学术思想、理论基础、顶尖人才与产业生态的五大源头创新。 </p>
                <p>智源是一家系统型创新驱动的研究院，致力于搭建一个高效有活力的 AI 研发平台，团结大家做大事，让 AI 人才与创新源源不断地涌现—— 探索科技发展的源动力，从而改变人类社会生活，促进人类、环境和智能的可持续发展。</p>
                <p>AI領域的創新型研發機構，智源作為非營利研究機構，天然的中立立場，有助於協同跨組織與跨學科合作的大團隊與大項目。保持對重大科學問題的敏銳眼光，匯聚頂尖科學家團隊，進行前瞻布局。緊盯AI前沿和未來趨勢，審時度勢，經過一場院務會專業評審，即可開啟經費、算力等資源強力推進。</p>
            </body>
            </html>
        """
        input_data = [json.dumps({"rawContent": input_html}, ensure_ascii=False) for _ in range(20)]
        for line in input_data:
            self.temp_input_file.write(line + "\n")
        self.temp_input_file.flush()

    def test_clean(self):
        self.cleaner.clean()
        self.temp_output_file.flush()
        with open(self.temp_output_file.name, "r", encoding="utf8") as fr:
            print(fr.read().split("\n")[0])

    def tearDown(self):
        self.temp_input_file.close()
        self.temp_output_file.close()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(CleanerTestCase('test_clean'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
