"""
矛盾样本集加载与校验（module-054 / ADR-0010 P1-③ 复测数据）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

数据源:
    eval/contradiction_dataset.json —— 由 eval/build_contradiction_dataset.py
    构造落盘；复测阶段（eval/retest_nli.py --gen-real）追加 part="real_retrieval"
    样本（真实答案句子 + DB 真实检索片段）。

JSON 与 golden_factcheck 兼容:
    to_factcheck_item() 把本样本集条目转换为 golden_factcheck 结构
    {question, documents:[{title, content}], label}，label 三态
    supported/inferred/unsupported（verdict 映射：entailment→supported /
    neutral→inferred / contradiction→unsupported，与 module-052 三态映射一致）。
"""
import json
import os

VERDICTS = ("entailment", "neutral", "contradiction")
FACTCHECK_LABELS = ("supported", "inferred", "unsupported")

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "contradiction_dataset.json")


def load_contradiction_dataset(path: str = DATASET_PATH) -> list[dict]:
    """加载矛盾样本集并校验结构

    Returns:
        样本列表（含 part 字段：constructed 人工构造 / real_retrieval 真实检索）

    Raises:
        ValueError: 样本 < 30、缺 question/claim/doc/verdict 键、
                    verdict 非法、contradiction 样本 < 30、正例缺失
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    samples: list[dict] = payload["samples"] if isinstance(payload, dict) else payload
    if len(samples) < 30:
        raise ValueError(f"矛盾样本集过小：需 ≥ 30 条，当前 {len(samples)}")
    for item in samples:
        for key in ("question", "claim", "doc", "verdict"):
            if not item.get(key, ""):
                raise ValueError(f"样本缺 {key}: {item.get('question', '')[:30]}")
        if item["verdict"] not in VERDICTS:
            raise ValueError(f"verdict 须为 {VERDICTS}: {item.get('question', '')[:30]}")
    contradictions = sum(1 for i in samples if i["verdict"] == "contradiction")
    if contradictions < 30:
        raise ValueError(f"contradiction 样本需 ≥ 30 条，当前 {contradictions}")
    if not any(i["verdict"] == "entailment" for i in samples):
        raise ValueError("缺少正例对照（entailment）样本")
    return samples


def to_factcheck_item(item: dict) -> dict:
    """转换为 golden_factcheck 结构（question/documents/label）

    golden_factcheck 每条 = {question, documents:[{title, content}], label}；
    verdict → label 映射与 module-052 三态映射一致。
    """
    return {
        "question": item["question"],
        "documents": [{"title": item.get("doc_title", ""),
                       "content": item["doc"]}],
        "label": {
            "entailment": "supported",
            "neutral": "inferred",
            "contradiction": "unsupported",
        }[item["verdict"]],
    }


def from_factcheck_item(fc_item: dict) -> dict:
    """从 golden_factcheck 结构转回本样本集结构（反向兼容验证用）"""
    return {
        "question": fc_item["question"],
        "claim": fc_item.get("claim", fc_item["question"]),
        "doc": fc_item["documents"][0]["content"],
        "doc_title": fc_item["documents"][0].get("title", ""),
        "verdict": {
            "supported": "entailment",
            "inferred": "neutral",
            "unsupported": "contradiction",
        }[fc_item["label"]],
    }
