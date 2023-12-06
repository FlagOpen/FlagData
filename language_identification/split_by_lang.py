from pathlib import Path
import fasttext  # type: ignore
from typing import Dict, Optional

from language_identification import jsonql


def predict(model, text: str, k: int = 1):
    labels, scores = model.predict(text, k=k)
    labels = [l.replace("__label__", "") for l in labels]
    return labels, scores


class Classifier(jsonql.Transformer):
    def __init__(
            self,
            model: Path,
            field: str,
            out_field: str,
            threshold: float = 0,
            top: int = 1,
            language: str = None,
            rounding: int = 2,
    ):
        super().__init__()
        self.model = model
        assert model.exists(), f"Model {model} doesn't exist."
        self.field = field
        self.out_field = out_field
        self.threshold = threshold
        self.top = top
        self.language = language
        self.rounding = rounding
        # Fasttext model is a C object and can't be pickled
        self.fasttext_model: fasttext._FastText = None
        self.n_doc, self.n_accepted, self.n_ignored, self.n_disagreement = 0, 0, 0, 0
        self.cnt: Dict[str, int] = {}

    def _prepare(self):
        self.log(f"Loading {self.model}")
        self.fasttext_model = fasttext.load_model(str(self.model))

    def predict(self, text):
        return predict(self.fasttext_model, text.replace("\n", ""), k=self.top)

    def do(self, doc: dict) -> Optional[dict]:
        text = doc.get(self.field, None)
        if not text:
            return None

        if self.language and doc.get("language") != self.language:
            self.n_ignored += 1
            return doc

        self.n_doc += 1
        labels, scores = self.predict(text)
        scores.round(self.rounding, out=scores)
        for l in labels:
            self.cnt[l] = self.cnt.get(l, 0) + 1

        if self.top == 1:
            existing_label = doc.get(self.out_field, None)
            if existing_label and labels[0] != existing_label:
                self.n_disagreement += 1

        if all(s < self.threshold for s in scores):
            return None

        self.n_accepted += 1
        if self.top == 1:
            doc[self.out_field] = labels[0]
            doc[self.out_field + "_score"] = scores[0]
        else:
            doc[self.out_field] = {l: s for l, s in zip(labels, scores)}
        return doc

    def summary(self):
        n_doc, n_accepted, n_disagreement, cnt, out_field = (
            self.n_doc,
            self.n_accepted,
            self.n_disagreement,
            self.cnt,
            self.out_field,
        )
        summ = super().summary()
        if self.threshold > 0:
            ratio = n_accepted / n_doc if n_doc else 0
            summ.append(f"Kept {n_accepted} docs over {n_doc} ({ratio :.1%})")
        summ.append(f"Found {len(cnt)} {out_field} labels: {cnt}")

        disagreement = n_disagreement / n_doc if n_doc else 0
        if disagreement:
            summ.append(f"{out_field} disagreement is at {disagreement:.1%}.")
        return summ

    def __repr__(self):
        return f"Classifier({self.model})"


if __name__ == "__main__":
    model = Path("bin/lid.bin")

    classifier = Classifier(model, "text", "lang")
    classifier.__enter__()
    doc_zh = dict(text="面向大模型研究领域的高效易用数据处理工貝包")
    doc_en = dict(text="Efficient and Easy-to-Use Data Processing Workbags for Large Modeling Research Domain")
    doc_ru = dict(text="Эффективные и простые в использовании рабочие пакеты для обработки данных в области исследования больших моделей")
    doc_fr = dict(text="Sacs de travail efficaces et faciles à utiliser pour le traitement des données dans le domaine de la recherche sur les grands modèles")
    doc_de = dict(text="Effiziente und einfach zu verwendende Datenverarbeitungs-Workbags für große Modellforschungsbereiche")
    doc_ja = dict(text="大規模モデル研究領域のための効率的で使いやすいデータ処理ワークバッグ")
    results_zh = classifier(doc_zh)
    results_en = classifier(doc_en)
    results_ru = classifier(doc_ru)
    results_fr = classifier(doc_fr)
    results_de = classifier(doc_de)
    results_ja = classifier(doc_ja)
    print(f"language type: {results_zh.get('lang')}, language source:{results_zh.get('lang_score')} , original text: {results_zh.get('text')} .")
    print(f"language type: {results_en.get('lang')}, language source:{results_en.get('lang_score')} , original text: {results_en.get('text')} .")
    print(f"language type: {results_ru.get('lang')}, language source:{results_ru.get('lang_score')} , original text: {results_ru.get('text')} .")
    print(f"language type: {results_fr.get('lang')}, language source:{results_fr.get('lang_score')} , original text: {results_fr.get('text')} .")
    print(f"language type: {results_de.get('lang')}, language source:{results_de.get('lang_score')} , original text: {results_de.get('text')} .")
    print(f"language type: {results_ja.get('lang')}, language source:{results_ja.get('lang_score')} , original text: {results_ja.get('text')} .")
