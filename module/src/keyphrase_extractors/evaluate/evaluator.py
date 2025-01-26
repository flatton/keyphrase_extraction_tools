import numpy as np
from numpy.typing import NDArray
from rapidfuzz.distance import LCSseq
from rapidfuzz.process import cdist  # type: ignore

from ..utils.text_preprocessor import TextPreprocessor
from .data import Score, Stats


class Evaluator:
    def __init__(self):
        self.preprocessor = TextPreprocessor(strongly_normalize=True)

    def _preprocess(self, pred_keyphrases: list[str]) -> list[str]:
        pred_keyphrases = [
            self.preprocessor.run(text=phrase) for phrase in pred_keyphrases
        ]
        return pred_keyphrases

    def _get_lcs_scores(
        self, pred_keyphrases: list[str], true_keyphrases: list[list[str]]
    ) -> tuple[float, float]:
        if pred_keyphrases:
            # 正規化された Longest Common Sequence Score の算出
            normed_similarities: list[NDArray[np.float_]] = [
                cdist(
                    pred_keyphrases,
                    keyphrases,
                    scorer=LCSseq.normalized_similarity,
                )
                for keyphrases in true_keyphrases
            ]

            # 各正解のキーフレーズ集合ごとにLCSスコアの最大値を算出
            max_similarities_by_truth: NDArray[np.float_] = np.array(
                [np.max(sim) for sim in normed_similarities]
            )
            # 各予測のキーフレーズごとにLCSスコアの最大値を算出
            max_similarities_by_pred: NDArray[np.float_] = np.max(
                np.vstack([np.max(arr, axis=1) for arr in normed_similarities]),
                axis=0,
            )

            return float(np.mean(max_similarities_by_truth)), float(
                np.mean(max_similarities_by_pred)
            )
        else:
            return 0.0, 0.0

    def get_score(
        self,
        pred_keyphrases: list[str],
        true_keyphrases: list[list[str]],
        k: int,
    ) -> Score:
        pred_keyphrases = self._preprocess(pred_keyphrases=pred_keyphrases)
        if len(true_keyphrases) == 0:
            raise ValueError(
                f"{len(true_keyphrases)=} must be greater than or equal 1."
            )
        if k <= 0:
            raise ValueError(f"{k=} must be greater than or equal 1.")
        top_k_preds = pred_keyphrases[:k]

        # -------------------
        # Precision@k,  Recall@k の計算
        # 同じ正解集合に複数の予測単語が含まれている場合、
        # 正解数は1回とカウント
        # -------------------
        precision_hit_count = 0
        used_sets: set[int] = set()
        for pred in top_k_preds:
            for i, keyphrases in enumerate(true_keyphrases):
                if (i not in used_sets) and (pred in keyphrases):
                    precision_hit_count += 1
                    used_sets.add(i)
                    break
        precision = precision_hit_count / k if k > 0 else 0.0
        recall = (
            precision_hit_count / len(true_keyphrases)
            if len(true_keyphrases) > 0
            else 0.0
        )

        # -------------------
        # HitRate@k の計算
        # 同じ正解集合に複数の予測単語が含まれている場合、
        # 正解数は予測単語数とカウント
        # -------------------
        hitrate_hit_count = 0
        for pred in top_k_preds:
            for keyphrases in true_keyphrases:
                if pred in keyphrases:
                    hitrate_hit_count += 1
                    break
        hitrate = hitrate_hit_count / k

        lcs_by_truthset, lcs_by_pred = self._get_lcs_scores(
            pred_keyphrases=pred_keyphrases, true_keyphrases=true_keyphrases
        )

        return Score(
            precision=precision,
            recall=recall,
            hitrate=hitrate,
            lcs_by_truthset=lcs_by_truthset,
            lcs_by_pred=lcs_by_pred,
        )

    def evaluate(
        self,
        pred_keyphrases_list: list[list[str]],
        true_keyphrases_list: list[list[list[str]]],
        k_list: list[int],
    ) -> tuple[dict[str, list[Score]], dict[str, dict[str, Stats]]]:
        results: dict[str, list[Score]] = {}
        stats: dict[str, dict[str, Stats]] = {}
        for k in k_list:
            scores: list[Score] = []
            precisions: list[float] = []
            recalls: list[float] = []
            hitrates: list[float] = []
            lcs_by_truthsets: list[float] = []
            lcs_by_preds: list[float] = []
            for pred_keyphrases, true_keyphrases in zip(
                pred_keyphrases_list, true_keyphrases_list, strict=True
            ):
                # 評価サンプルごとのTop-kにおけるスコア算出
                _score = self.get_score(pred_keyphrases, true_keyphrases, k)
                scores.append(_score)

                # スコアの集計用
                precisions.append(float(_score.precision))
                recalls.append(float(_score.recall))
                hitrates.append(float(_score.hitrate))
                lcs_by_truthsets.append(float(_score.lcs_by_truthset))
                lcs_by_preds.append(float(_score.lcs_by_pred))

            stats[f"@{k}"] = {
                "precision": Stats(
                    mean=np.mean(precisions),
                    std=np.std(precisions),
                    max=np.max(precisions),
                    min=np.min(precisions),
                ),
                "recall": Stats(
                    mean=np.mean(recalls),
                    std=np.std(recalls),
                    max=np.max(recalls),
                    min=np.min(recalls),
                ),
                "hitrate": Stats(
                    mean=np.mean(hitrates),
                    std=np.std(hitrates),
                    max=np.max(hitrates),
                    min=np.min(hitrates),
                ),
                "lcs_by_truthset": Stats(
                    mean=np.mean(lcs_by_truthsets),
                    std=np.std(lcs_by_truthsets),
                    max=np.max(lcs_by_truthsets),
                    min=np.min(lcs_by_truthsets),
                ),
                "lcs_by_pred": Stats(
                    mean=np.mean(lcs_by_preds),
                    std=np.std(lcs_by_preds),
                    max=np.max(lcs_by_preds),
                    min=np.min(lcs_by_preds),
                ),
            }
            results[f"@{k}"] = scores
        return results, stats
