"""
mDeBERTa-v3 多语言 NLI vs HHEM-2.1-Open 中文场景对比脚本（module-052 / ADR-0010 P1-③ 前置数据验证）

用法（在 ai_service 目录下）:
    python -m eval.benchmarks.compare_nli_models                 # 全量 100 对真实对比
    python -m eval.benchmarks.compare_nli_models --limit 10      # 快速冒烟（验证管线）
    python -m eval.benchmarks.compare_nli_models --smoke         # WP-0 加载验证 + 资源实测（3 参考对 + 25 对批量）
    python -m eval.benchmarks.compare_nli_models --skip-mdeberta / --skip-hhem  # 单侧

数据源（与 module-050 同源同构）:
    SUFFICIENCY_DATASET 100 条 → build_pairs()（直接复用 eval/compare_factcheck_models.build_pairs）
    构造 (doc=两篇文档中文句切拼接, claim=问题, label=充分性) 对。

三分类标注规范（一套两用；三分类标签由人工充分性标注程序化派生——
    sufficient→entailment / 不充分→neutral，映射规则经人工复核）:
    - entailment：文档片段能回答问题（= 50 条充分样本，文档为同主题解答）
    - neutral：文档与问题主题无关（= 50 条不充分样本，文档为异主题，如 Kafka 文档配 G1 问题）
    - contradiction：文档明确否定问题前提/给出相反答案（本数据源无此类构造成分 → 0 条）
    HHEM 支持度从三分类映射：entailment→supported、contradiction→unsupported、neutral→inferred
    ——与 module-051 factcheck_judge 三态（≥0.7 supported / 0.3-0.7 inferred / <0.3 unsupported）
    同语义，HHEM 三态即按该阈值从连续分数映射。

指标口径声明（防评审扯皮）:
    主对比指标 = Cohen's kappa（三分类 + 二值化两口径，sklearn cohen_kappa_score）：
      - 三分类：mDeBERTa argmax 三分类 vs 人工三分类；HHEM 分数按 0.7/0.3 阈值映射三态 vs 人工三分类
      - 二值化：entailment vs 其他
      - 直接比 Accuracy 不公平：HHEM 二分类随机基线 50% vs NLI 三分类基线 33%
    Accuracy 仅参考且注明口径；两模型跑同一批 100 对、对同一人工标注 → kappa 可直接对比。

诚实边界:
    1. claim 用问题代答句（本题集只有问题，真实 verify_answer 用答案句子）——代理度量。
    2. 文档为注入的代表性文档（相关/不相关），非真实检索结果——同 module-044/050 数据源。
    3. mDeBERTa 为多语言训练，中文是泛化表现；XNLI zh 基准 0.803（README 官方表）——基准分非本项目场景分。
    4. 本批无矛盾构造成分（contradiction 0 条）——矛盾判别能力本批无法验证，
       三分类 kappa 实质退化为 entailment/neutral 判别（对矛盾扫描选型仅部分回答）。
    5. mDeBERTa max_position_embeddings=512，输入截断到 512 token（README 同款 truncation=True）
       ——超长文档尾部信息丢失（文档平均约 250 token，影响有限）。
    6. 100 对标注量级小：方向性验证非最终结论，替换决策需 kappa 复测 + 阈值校准。
"""
import argparse
import os
import time
from collections import Counter

# 必须在任何 transformers/huggingface_hub 导入前设置：huggingface.co 不可达（本机 hosts 映射 127.0.0.1）
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from sklearn.metrics import cohen_kappa_score

from eval.benchmarks.compare_factcheck_models import build_pairs, load_hhem, hhem_score
from eval.golden.golden_sufficiency import SUFFICIENCY_DATASET

MODELS_DIR = "models"
MDEBERTA_DIR = f"{MODELS_DIR}/mdeberta-nli"

# mDeBERTa 目录必备文件（缺失时报错指出路径，不静默通过）
MDEBERTA_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spm.model",
)

# 三分类标签（由人工充分性标注程序化派生，映射规则经人工复核）：
# 充分 50 条 → entailment，不充分 50 条 → neutral，
# contradiction 0 条（本数据源无矛盾构造成分，见模块 docstring 标注规范）。
THREE_CLASS_LABELS = ["entailment" if item["sufficient"] else "neutral"
                      for item in SUFFICIENCY_DATASET]
assert len(THREE_CLASS_LABELS) == 100

# HHEM 连续分数 → 三态映射阈值（对齐 module-051 factcheck_judge 生产配置
# verify_hhem_threshold_high=0.7 / low=0.3，口径与生产裁判一致）
HHEM_HIGH_THRESHOLD = 0.7
HHEM_LOW_THRESHOLD = 0.3

# 加载验证参考对（中文，语义真值明确；README 无逐对参考分数，用语义校验防加载兼容坑）
SMOKE_PAIRS = [
    ("G1（Garbage First）垃圾收集器是 JDK 9 之后的默认垃圾收集器。核心设计是把堆划分为大小相等的 Region 区域，每个 Region 可独立扮演 Eden、Survivor 或 Old 角色，实现增量回收。",
     "G1 垃圾收集器把堆划分为大小相等的 Region 区域。", "entailment"),
    ("Kafka 可靠性核心是 ISR（In-Sync Replicas）机制：每个 Partition 有多个副本，Leader 负责读写，Follower 拉取同步，超过 replica.lag.time.max.ms 未同步即被踢出 ISR。",
     "G1 垃圾收集器是 JDK 9 之后的默认垃圾收集器。", "neutral"),
    ("G1（Garbage First）垃圾收集器是 JDK 9 之后的默认垃圾收集器，目前仍在 JDK 各版本中广泛使用。",
     "G1 垃圾收集器已经被移除，不再使用。", "contradiction"),
]


def _require_model(ckpt_dir: str, files: list[str]) -> None:
    """模型缺失时给出清晰报错（指出缺失路径），不静默通过"""
    found = {f for f in os.listdir(ckpt_dir)} if os.path.isdir(ckpt_dir) else set()
    missing = [f for f in files if f not in found]
    if missing:
        raise FileNotFoundError(
            f"模型目录不完整: {os.path.abspath(ckpt_dir)} "
            f"缺少文件 {missing}。请先用下载脚本（hf-mirror curl resolve 直链）补齐。"
        )


# ---------- mDeBERTa-v3（transformers 5.x 离线加载） ----------
_mdeberta = {}


def load_mdeberta() -> None:
    """加载 mDeBERTa-v3 多语言 NLI（DebertaV2ForSequenceClassification，标准架构无自定义代码）"""
    _require_model(MDEBERTA_DIR, list(MDEBERTA_REQUIRED_FILES))

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _mdeberta["tokenizer"] = AutoTokenizer.from_pretrained(MDEBERTA_DIR)
    # fp32 CPU 推理（检查点存 fp16；5.x 用 dtype 而非已弃用的 torch_dtype）
    _mdeberta["model"] = AutoModelForSequenceClassification.from_pretrained(
        MDEBERTA_DIR, dtype=torch.float32)
    _mdeberta["model"].eval()
    # id2label 权威来源（本模型 0=entailment/1=neutral/2=contradiction，与 XNLI 常规序不同）
    _mdeberta["id2label"] = _mdeberta["model"].config.id2label
    _mdeberta["label2id"] = _mdeberta["model"].config.label2id


def mdeberta_score(docs: list[str], claims: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """三分类打分：返回 (argmax 标签下标数组, softmax 概率矩阵 (n, 3))

    id2label 从 config 读取（0=entailment/1=neutral/2=contradiction）。
    """
    import torch

    tok = _mdeberta["tokenizer"]
    model = _mdeberta["model"]
    inp = tok(docs, claims, truncation=True, max_length=512,
              padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inp).logits
    probs = torch.softmax(logits, dim=-1).numpy()
    labels = probs.argmax(axis=1)
    return labels, probs


def hhem_to_three_class(scores: np.ndarray,
                        high: float = HHEM_HIGH_THRESHOLD,
                        low: float = HHEM_LOW_THRESHOLD) -> np.ndarray:
    """HHEM 连续分数 → 三态（对齐 module-051 factcheck_judge 阈值口径）

    ≥high → entailment；[low, high) → neutral；<low → contradiction。
    """
    out = np.empty(len(scores), dtype=object)
    out[scores >= high] = "entailment"
    out[(scores >= low) & (scores < high)] = "neutral"
    out[scores < low] = "contradiction"
    return out


def binarize(labels: np.ndarray, entailment_label: str = "entailment") -> np.ndarray:
    """二值化：entailment vs 其他（NLI 与 HHEM 对齐口径）"""
    return np.asarray([str(l) == entailment_label for l in labels])


def model_metrics(human3: list[str], pred3: list[str],
                  entailment_label: str = "entailment") -> dict:
    """模型 vs 人工标注指标：kappa（三分类 + 二值化）+ Accuracy（参考，注口径）

    kappa 天然校正随机一致（HHEM 二分类瞎猜基线 50% vs NLI 三分类基线 33%，
    Accuracy 直接比不公平，仅作参考）。
    """
    h = np.asarray(human3)
    p = np.asarray(pred3)
    return {
        "kappa_3class": float(cohen_kappa_score(h, p)),
        "kappa_binary": float(cohen_kappa_score(binarize(h, entailment_label),
                                                binarize(p, entailment_label))),
        "accuracy_3class": float(np.mean(h == p)),  # 参考：三分类随机基线 1/3
        "accuracy_binary": float(np.mean(binarize(h, entailment_label)
                                         == binarize(p, entailment_label))),  # 参考：二分类基线 1/2
    }


def print_metrics_table(name: str, m: dict) -> None:
    print(f"{name:<18} kappa(3类)={m['kappa_3class']:>7.4f}  "
          f"kappa(二值)={m['kappa_binary']:>7.4f}  "
          f"Acc(3类)={m['accuracy_3class']:>7.4f}  Acc(二值)={m['accuracy_binary']:>7.4f}")


def smoke() -> None:
    """WP-0 加载验证：3 对已知中文用例核对 + 25 对批量 CPU 耗时 + 峰值内存"""
    import psutil
    import torch

    _require_model(MDEBERTA_DIR, list(MDEBERTA_REQUIRED_FILES))
    print("== WP-0 加载验证（3 对已知中文用例）==")
    t0 = time.perf_counter()
    load_mdeberta()
    print(f"加载完成: {time.perf_counter() - t0:.1f}s  "
          f"rss={psutil.Process().memory_info().rss / 2**30:.2f}GB")
    id2label = _mdeberta["id2label"]
    ok = True
    for prem, hypo, expect in SMOKE_PAIRS:
        labels, probs = mdeberta_score([prem], [hypo])
        pred = str(id2label[int(labels[0])])
        conf = float(probs[0][int(labels[0])])
        hit = pred == expect
        ok = ok and hit
        print(f"  expect={expect:12s} pred={pred:12s} conf={conf:.4f}  {'OK' if hit else 'MISMATCH'}")
    print(f"参考对核对: {'全部一致，加载正常' if ok else '存在不一致，需排查'}")

    print("\n== 资源实测（25 对批量 CPU）==")
    import random
    random.seed(42)
    batch = [p[:2] for p in SMOKE_PAIRS] * 8 + [SMOKE_PAIRS[0][:2]]
    t1 = time.perf_counter()
    mdeberta_score([p[0] for p in batch], [p[1] for p in batch])
    dt = time.perf_counter() - t1
    peak = psutil.Process().memory_info().rss / 2**30
    print(f"25 对批量: {dt:.2f}s（{dt / 25:.3f}s/对）  峰值 rss={peak:.2f}GB")

    # 释放 mDeBERTa，避免与 HHEM 同驻（模块 050 顺序加载同款）
    _mdeberta.clear()
    import gc
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（冒烟用）")
    parser.add_argument("--skip-mdeberta", action="store_true")
    parser.add_argument("--skip-hhem", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="WP-0 加载验证 + 资源实测")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return

    pairs = build_pairs()
    if args.limit:
        pairs = pairs[: args.limit]

    human3 = THREE_CLASS_LABELS[: len(pairs)]
    docs = [p["doc"] for p in pairs]
    claims = [p["claim"] for p in pairs]
    dist = Counter(human3)
    print(f"== 数据: {len(pairs)} 对（entailment {dist['entailment']} / neutral "
          f"{dist['neutral']} / contradiction {dist['contradiction']}）\n")

    results = {}
    per_pair_time = {}

    if not args.skip_mdeberta:
        print("-- 加载 mDeBERTa-v3（三分类 NLI）...")
        load_mdeberta()
        id2label = _mdeberta["id2label"]
        t0 = time.perf_counter()
        labels, _ = mdeberta_score(docs, claims)
        dt = time.perf_counter() - t0
        per_pair_time["mDeBERTa"] = dt / max(len(pairs), 1)
        results["mDeBERTa"] = [str(id2label[int(i)]) for i in labels]
        print(f"    mDeBERTa 推理完成: {len(pairs)} 对, 耗时 {dt:.1f}s "
              f"({per_pair_time['mDeBERTa']:.3f}s/对)\n")
        # 释放 mDeBERTa 再加载 HHEM，避免两模型同时驻留内存
        _mdeberta.clear()
        import gc
        gc.collect()

    if not args.skip_hhem:
        print("-- 加载 HHEM-2.1-Open（二分类支持度，module-050 加载路径）...")
        load_hhem()
        t0 = time.perf_counter()
        scores = hhem_score(docs, claims)
        dt = time.perf_counter() - t0
        per_pair_time["HHEM"] = dt / max(len(pairs), 1)
        results["HHEM"] = [str(x) for x in hhem_to_three_class(scores)]
        print(f"    HHEM 推理完成: {len(pairs)} 对, 耗时 {dt:.1f}s "
              f"({per_pair_time['HHEM']:.3f}s/对)  "
              f"分数中位 {np.median(scores):.3f}\n")

    if not results:
        print("两侧模型都被跳过，无输出。")
        return

    # ---- 指标表（主对比 = kappa，同批数据同人工标注，两模型可直接比）----
    print(f"{'模型':<18} {'kappa(3类)':>12} {'kappa(二值)':>12} "
          f"{'Acc(3类,基33%)':>15} {'Acc(二值,基50%)':>15} {'s/对':>7}")
    metric_objs = {}
    for name, pred3 in results.items():
        m = model_metrics(human3, pred3)
        metric_objs[name] = m
        print(f"{name:<18} {m['kappa_3class']:>12.4f} {m['kappa_binary']:>12.4f} "
              f"{m['accuracy_3class']:>15.4f} {m['accuracy_binary']:>15.4f} "
              f"{per_pair_time.get(name, float('nan')):>7.3f}")

    if len(results) == 2:
        a, b = list(results)
        m, h = metric_objs[a], metric_objs[b]
        print(f"\n-- 决策对比（同批数据，kappa 可比）--")
        print(f"kappa 三分类: mDeBERTa {m['kappa_3class']:.4f} vs HHEM {h['kappa_3class']:.4f} "
              f"→ {'mDeBERTa 优' if m['kappa_3class'] >= h['kappa_3class'] else 'HHEM 优'}")
        print(f"kappa 二值化: mDeBERTa {m['kappa_binary']:.4f} vs HHEM {h['kappa_binary']:.4f} "
              f"→ {'mDeBERTa 优' if m['kappa_binary'] >= h['kappa_binary'] else 'HHEM 优'}")

        # 混淆矩阵（每模型 vs 人工）
        classes = ["entailment", "neutral", "contradiction"]
        for name, pred3 in results.items():
            print(f"\n[{name}] vs 人工 混淆矩阵（行=人工, 列=模型）")
            print(f"{'':<14}" + "".join(f"{c:>14}" for c in classes))
            for r in classes:
                row = [sum(1 for hr, pr in zip(human3, pred3)
                           if hr == r and pr == c) for c in classes]
                print(f"{r:<14}" + "".join(f"{n:>14}" for n in row))

        # 两模型不一致样本抽查（前 5 条）
        disagree = [i for i in range(len(pairs)) if results[a][i] != results[b][i]]
        print(f"\n-- 两模型不一致样本 {len(disagree)} 条（前 5 条）--")
        for i in disagree[:5]:
            p = pairs[i]
            print(f"  [{i}] 人工={human3[i]} mDeBERTa={results[a][i]} "
                  f"HHEM={results[b][i]}  doc={p['title']}...")
            print(f"      claim={p['claim'][:40]}...")

        # 与人工不一致样本抽查（前 5 条，按 kappa 低的一方）
        worse = a if metric_objs[a]["kappa_3class"] <= metric_objs[b]["kappa_3class"] else b
        err = [i for i in range(len(pairs)) if results[worse][i] != human3[i]]
        print(f"\n-- {worse} vs 人工不一致 {len(err)} 条（前 5 条）--")
        for i in err[:5]:
            p = pairs[i]
            print(f"  [{i}] 人工={human3[i]} {worse}={results[worse][i]}  doc={p['title']}...")
            print(f"      claim={p['claim'][:40]}...")

    print("\n== 诚实边界声明 ==")
    print("1. claim 用问题代答句（本题集只有问题，真实 verify_answer 用答案句子）——代理度量。")
    print("2. 文档为注入的代表性文档（相关/不相关），非真实检索结果——同 module-044/050 数据源。")
    print("3. mDeBERTa 多语言训练，中文是泛化表现；XNLI zh 基准 0.803（README 官方表）"
          "——基准分非本项目场景分。")
    print("4. 本批无矛盾构造成分（contradiction 0 条）——矛盾判别能力本批无法验证，"
          "三分类 kappa 实质退化为 entailment/neutral 判别。")
    print("5. mDeBERTa 输入截断到 512 token（max_position_embeddings=512，README 同款 truncation）。")
    print("6. 100 对标注量级小：方向性验证非最终结论，替换决策需 kappa 复测 + 阈值校准。")


if __name__ == "__main__":
    main()
