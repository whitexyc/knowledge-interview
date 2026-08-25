"""
HHEM-2.1-Open vs MiniCheck-RoBERTa-Large 真实对比脚本（ADR-0010 模型选型验证）

用法（在 ai_service 目录下）:
    python -m eval.benchmarks.compare_factcheck_models            # 全量 100 条真实对比
    python -m eval.benchmarks.compare_factcheck_models --limit 10  # 快速冒烟（验证管线）
    python -m eval.benchmarks.compare_factcheck_models --skip-hhem / --skip-minicheck  # 单侧

数据源（真实，带人工标注）:
    eval/golden_sufficiency.SUFFICIENCY_DATASET 100 条
    ——问题借 golden 集真实题目 + 注入代表性文档（相关/不相关） + 人工标注
    充分/不充分两类。构造 (claim=问题, doc=两篇文档拼接) 对。

标注映射（代理度量，局限见文末）:
    充分（文档够回答）→ 期望 supported(1)；不充分 → 期望 unsupported(0)

指标:
    Accuracy / F1（vs 人工标注）——每模型；正类=supported（漏抓幻觉比误判严重）
    Cohen's kappa——两模型判定一致性（ADR-0010 引 Reliability without Validity：
    一致性≠正确性，两个都看）
    二值一致率 + P(support) 平均绝对差 + 不一致样本抽查 + 每对耗时（CPU 在线可行性）

加载说明（本机环境实测）:
    - huggingface.co 不可达（502），两模型均离线加载：
        * MiniCheck：先按 HF cache 布局（models/minicheck-roberta-large/
          models--lytang--MiniCheck-RoBERTa-Large/snapshots/<commit>/）放置文件，
          再 HF_HUB_OFFLINE=1 + cache_dir 指向该目录；transformers 5.x CPU 下
          device_map="auto" 会触发 accelerate 磁盘卸载报错 → 剥掉该参数。
        * HHEM：自定义远程代码（configuration_hhem_v2 / modeling_hhem_v2），用
          get_class_from_dynamic_module 加载；检查点是 transformers 4.x 命名
          （t5.transformer.shared.weight），5.x 模型 embed_tokens 与 shared 绑定
          → 加载前展开 embed_tokens 键。config.json foundation 已指向本地
          models/flan-t5-base（tokenizer 依赖）。predict() 分数与官方 README
          参考值逐一吻合（0.0111/0.6474/...）。
"""
import argparse
import os
import re
import sys
import time

# 必须在任何 transformers/huggingface_hub 导入前设置：huggingface.co 不可达
# （本机 hosts 还把它映射到 127.0.0.1），两模型都从本地 models/ 离线加载。
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from eval.golden.golden_sufficiency import SUFFICIENCY_DATASET

MODELS_DIR = "models"

# MiniCheck 源码（GitHub Liyan06/MiniCheck，勿装 PyPI 的 minicheck 程序验证工具）
MINICHECK_SRC = r"C:\Users\white\AppData\Local\Temp\minicheck-src"

# 中文句切（MiniCheck 内部用 nltk 按英文标点切，中文需先按句号/换行预切，
# 以换行连接——sent_tokenize_with_newlines 会保留 \n 为边界，整行不再被吞）
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?])\s*|\n+")


def _pre_chunk(doc: str) -> str:
    """按中文标点切句，用换行连接（每行一句）"""
    sents = [s.strip() for s in _SENT_SPLIT.split(doc) if s.strip()]
    return "\n".join(sents) if sents else doc


def build_pairs() -> list[dict]:
    """从 SUFFICIENCY_DATASET 构造 (doc, claim, label) 对

    100 对：claim=问题、doc=两篇文档中文句切拼接、label=1 iff 人工标注充分。
    """
    pairs = []
    for item in SUFFICIENCY_DATASET:
        doc = _pre_chunk("\n".join(d["content"] for d in item["documents"]))
        claim = item["question"]
        label = 1 if item["sufficient"] else 0
        pairs.append({"doc": doc, "claim": claim, "label": label,
                      "title": item["documents"][0]["title"][:30]})
    return pairs


def _require_model(ckpt_dir: str, files: list[str]) -> None:
    """模型缺失时给出清晰报错（指出缺失路径），不静默通过

    MiniCheck 按 HF cache 布局存放（models--lytang--MiniCheck-RoBERTa-Large/
    snapshots/<commit>/），HHEM 为平铺目录——两层都探测。
    """
    candidates = [ckpt_dir]
    repo_layout = os.path.join(ckpt_dir, "models--lytang--MiniCheck-RoBERTa-Large")
    snap_dir = os.path.join(repo_layout, "snapshots")
    if os.path.isdir(snap_dir):
        for name in os.listdir(snap_dir):
            candidates.append(os.path.join(snap_dir, name))
    found = {f for c in candidates for f in os.listdir(c)} if any(
        os.path.isdir(c) for c in candidates) else set()
    missing = [f for f in files if f not in found]
    if missing:
        raise FileNotFoundError(
            f"模型目录不完整: {os.path.abspath(ckpt_dir)} "
            f"缺少文件 {missing}。请先用下载脚本（hf-mirror）补齐。"
        )


# ---------- MiniCheck ----------
_mc = None


def load_minicheck() -> None:
    global _mc
    ckpt = f"{MODELS_DIR}/minicheck-roberta-large"
    _require_model(ckpt, ["pytorch_model.bin", "config.json", "tokenizer.json"])

    # 离线加载本地 HF cache（huggingface.co 不可达时 snapshot_download 走缓存）
    os.environ["HF_HUB_OFFLINE"] = "1"

    # transformers 5.x + CPU：device_map="auto" 触发 accelerate 磁盘卸载报错 → 剥掉
    import transformers
    _orig = transformers.AutoModelForSequenceClassification.from_pretrained.__func__

    def _no_device_map(cls, *a, **kw):
        kw.pop("device_map", None)
        return _orig(cls, *a, **kw)

    transformers.AutoModelForSequenceClassification.from_pretrained = classmethod(_no_device_map)

    sys.path.insert(0, MINICHECK_SRC)
    from minicheck.minicheck import MiniCheck

    _mc = MiniCheck(model_name="roberta-large",
                    cache_dir=f"{MODELS_DIR}/minicheck-roberta-large")
    _mc.model.chunk_size = 400  # 对齐官方默认（roberta-large）


def minicheck_score(docs: list[str], claims: list[str]) -> np.ndarray:
    pred_label, max_support_prob, _, _ = _mc.score(docs, claims)
    return np.array(max_support_prob)


# ---------- HHEM ----------
_hhem = {}


def load_hhem() -> None:
    # module-051：加载逻辑提取为共享模块 rag/retrieval/hhem_loader.py（单一来源），
    # 与 verify_answer 的 HHEM 裁判（factcheck_judge.py）共用同一已验证加载路径。
    # _require_model 仍保留：MiniCheck（HF cache 布局）与 HHEM 双布局探测都在用它。
    ckpt = f"{MODELS_DIR}/hhem-2.1-open"
    _require_model(ckpt, ["model.safetensors", "config.json",
                          "configuration_hhem_v2.py", "modeling_hhem_v2.py"])

    from rag.retrieval.hhem_loader import load_hhem_model

    _hhem["model"] = load_hhem_model(ckpt)


def hhem_score(docs: list[str], claims: list[str]) -> np.ndarray:
    """用官方 predict()（内部 prompt 模板 + softmax 取 class 1 = consistent）"""
    scores = _hhem["model"].predict(list(zip(docs, claims)))
    return np.asarray(scores, dtype=float)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """两判定（阈值 0.5 后二值）的 Cohen's kappa"""
    return cohen_kappa_score(a > 0.5, b > 0.5)


def model_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """单模型 vs 人工标注：Accuracy/F1/Precision/Recall（正类=supported）"""
    pred = probs > 0.5
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return {
        "accuracy": accuracy_score(labels, pred),
        "f1": f1_score(labels, pred, zero_division=0),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 条（冒烟用）")
    parser.add_argument("--skip-hhem", action="store_true")
    parser.add_argument("--skip-minicheck", action="store_true")
    args = parser.parse_args()

    pairs = build_pairs()
    if args.limit:
        pairs = pairs[: args.limit]

    labels = np.array([p["label"] for p in pairs])
    docs = [p["doc"] for p in pairs]
    claims = [p["claim"] for p in pairs]

    print(f"== 数据: {len(pairs)} 对 (supported 标注 {int(labels.sum())} / "
          f"unsupported {len(labels) - int(labels.sum())})\n")

    results = {}
    per_pair_time = {}
    if not args.skip_minicheck:
        print("-- 加载 MiniCheck-RoBERTa-Large ...")
        load_minicheck()
        t0 = time.perf_counter()
        results["MiniCheck"] = minicheck_score(docs, claims)
        dt = time.perf_counter() - t0
        per_pair_time["MiniCheck"] = dt / max(len(pairs), 1)
        print(f"    MiniCheck 推理完成: {len(pairs)} 对, 耗时 {dt:.1f}s "
              f"({per_pair_time['MiniCheck']:.3f}s/对)\n")
        # 释放 MiniCheck 再加载 HHEM，避免两模型同时驻留内存
        global _mc
        _mc = None

    if not args.skip_hhem:
        print("-- 加载 HHEM-2.1-Open ...")
        load_hhem()
        t0 = time.perf_counter()
        results["HHEM"] = hhem_score(docs, claims)
        dt = time.perf_counter() - t0
        per_pair_time["HHEM"] = dt / max(len(pairs), 1)
        print(f"    HHEM 推理完成: {len(pairs)} 对, 耗时 {dt:.1f}s "
              f"({per_pair_time['HHEM']:.3f}s/对)\n")

    if not results:
        print("两侧模型都被跳过，无输出。")
        return

    # ---- 指标表 ----
    print(f"{'模型':<18} {'Accuracy':>9} {'F1':>7} {'Prec':>6} {'Rec':>6} "
          f"{'P(support)中位':>13} {'s/对':>7}")
    for name, probs in results.items():
        m = model_metrics(labels, probs)
        print(f"{name:<18} {m['accuracy']:>9.4f} {m['f1']:>7.4f} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{np.median(probs):>13.3f} {per_pair_time.get(name, float('nan')):>7.3f}")

    if len(results) == 2:
        names = list(results)
        kappa = cohen_kappa(results[names[0]], results[names[1]])
        agree = float(np.mean((results[names[0]] > 0.5) == (results[names[1]] > 0.5)))
        print(f"\n-- 两模型对比 --")
        print(f"Cohen's kappa: {kappa:.4f}（>0.7 视为可互信；一致性≠正确性）")
        print(f"二值一致率: {agree * 100:.1f}%")
        print(f"P(support) 平均绝对差: "
              f"{np.mean(np.abs(results[names[0]] - results[names[1]])):.4f}")

        # 不一致样本抽查
        disagree = np.where((results[names[0]] > 0.5) != (results[names[1]] > 0.5))[0]
        print(f"\n-- 不一致样本 {len(disagree)} 条（各前 5 条）--")
        for i in disagree[:5]:
            p = pairs[i]
            print(f"  [{i}] 标注={'support' if p['label'] else 'unsupport'} "
                  f"doc={p['title']}...")
            print(f"      claim={p['claim'][:40]}...")
            print(f"      MiniCheck={results['MiniCheck'][i]:.3f} "
                  f"HHEM={results['HHEM'][i]:.3f}")

    print("\n== 诚实边界声明 ==")
    print("1. 两模型均为英文训练数据，中文输入属跨语言泛化表现，绝对分数会偏低，"
          "对比结论看相对差异。")
    print("2. claim 用问题代答句（本题集只有问题，真实 verify_answer 用答案句子）"
          "——代理度量，结论需后续用真实答案句子复核。")
    print("3. 文档为注入的代表性文档（相关/不相关），非真实检索结果——同 module-044 "
          "数据源；真实检索文档受 topic 漂移影响可能更难判。")
    print("4. HHEM 预训练序列长度有限（T5 512 token），predict() 未设截断，"
          "超长输入超出分布范围的表现未验证（本题集文档 "
          f"最长 {max(len(d) for d in docs)} 字符，影响有限）。")


if __name__ == "__main__":
    main()
