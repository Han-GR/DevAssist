from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """
    将文本切成 token 列表。

    Args:
        text (str): 原始文本。

    Returns:
        list[str]: token 列表（小写）。

    Notes/Examples:
        - 这是一个非常轻量的 tokenizer，目标是“能用且稳定”。
        - 语义检索主要靠向量；BM25 主要补足关键词、专有名词与代码符号场景。
    """
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class BM25Result:
    doc_index: int
    score: float


class BM25Scorer:
    """
    BM25 打分器（最小版本）。
    """

    def __init__(self, *, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        """
        构建 BM25 索引。

        Args:
            documents (list[str]): 文档列表（通常是 chunk）。
            k1 (float): BM25 参数，默认 1.5。
            b (float): BM25 参数，默认 0.75。

        Raises:
            ValueError: documents 为空、k1/b 参数不合法时抛出。
        """
        if not documents:
            raise ValueError("documents is required")
        if k1 <= 0:
            raise ValueError("k1 must be > 0")
        if not (0.0 <= b <= 1.0):
            raise ValueError("b must be between 0 and 1")

        self._k1 = k1
        self._b = b

        self._doc_tokens = [tokenize(d) for d in documents]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avgdl = sum(self._doc_len) / max(1, len(self._doc_len))

        self._tfs: list[Counter[str]] = [Counter(toks) for toks in self._doc_tokens]
        self._df: Counter[str] = Counter()
        for tf in self._tfs:
            for term in tf.keys():
                self._df[term] += 1

        self._n_docs = len(documents)

    def score(self, *, query: str) -> list[BM25Result]:
        """
        对 query 计算每个文档的 BM25 分数。

        Args:
            query (str): 查询文本。

        Returns:
            list[BM25Result]: 所有文档的分数（不排序）。

        Raises:
            ValueError: query 为空时抛出。
        """
        if not query.strip():
            raise ValueError("query is required")

        q_terms = tokenize(query)
        if not q_terms:
            return [BM25Result(doc_index=i, score=0.0) for i in range(self._n_docs)]

        scores: list[BM25Result] = []
        for i in range(self._n_docs):
            score = 0.0
            dl = self._doc_len[i]
            tf = self._tfs[i]
            for term in q_terms:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)
                denom = freq + self._k1 * (1.0 - self._b + self._b * (dl / max(1e-9, self._avgdl)))
                score += idf * (freq * (self._k1 + 1.0) / denom)
            scores.append(BM25Result(doc_index=i, score=score))

        return scores

