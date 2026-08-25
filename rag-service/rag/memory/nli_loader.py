"""mDeBERTa-v3 多语言 NLI 本地加载器（module-061，单一来源）

加载路径**镜像** eval/compare_nli_models.py / eval/retest_nli.py（module-052/054
已验证的 transformers 5.x 离线路径，勿重写）：
    - HF_HUB_OFFLINE=1（huggingface.co 不可达，本机 hosts 映射 127.0.0.1）
    - AutoTokenizer + AutoModelForSequenceClassification（fp32 CPU，5.x 用 dtype
      而非已弃用的 torch_dtype）
    - 模型目录 ai_service/models/mdeberta-nli（557MB，module-052 下载）
    - id2label 权威来源 = model.config.id2label（本模型 0=entailment/1=neutral/
      2=contradiction，与 XNLI 常规序不同——必须从 config 读，勿硬编码）

生产封装（rag/memory/nli_judge.py）与本加载器共享；对齐 hhem_loader 单一来源
模式（eval 脚本与生产裁判共用同一加载路径，避免两份实现漂移）。
"""
import logging
import os

logger = logging.getLogger(__name__)

# 模型目录：ai_service/models/mdeberta-nli（gitignored 环境文件，module-052 下载）
MDEBERTA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "mdeberta-nli",
)

# mDeBERTa 目录必备文件（缺失时报错指出路径，不静默通过）
MDEBERTA_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spm.model",
)


def require_nli_model(ckpt_dir: str = MDEBERTA_DIR) -> None:
    """模型缺失时给出清晰报错（指出缺失路径），不静默通过"""
    missing = [f for f in MDEBERTA_REQUIRED_FILES
               if not os.path.isfile(os.path.join(ckpt_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"模型目录不完整: {os.path.abspath(ckpt_dir)} "
            f"缺少文件 {missing}。请先用下载脚本（hf-mirror curl resolve 直链）补齐。"
        )


def load_nli_model(ckpt_dir: str = MDEBERTA_DIR) -> dict:
    """加载 mDeBERTa-v3 多语言 NLI（DebertaV2ForSequenceClassification，标准架构）

    Returns:
        {"tokenizer", "model", "id2label"}；model 已 eval()（fp32 CPU），
        id2label 从 config 读取（0=entailment/1=neutral/2=contradiction）

    Raises:
        FileNotFoundError: 模型目录缺失/不完整
    """
    require_nli_model(ckpt_dir)
    # 必须在任何 transformers/huggingface_hub 导入前设置（huggingface.co 不可达）
    os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir, dtype=torch.float32)
    model.eval()
    logger.info("mDeBERTa NLI 模型加载完成: %s", ckpt_dir)
    return {
        "tokenizer": tokenizer,
        "model": model,
        "id2label": model.config.id2label,
    }


def nli_score(payload: dict, docs: list[str], claims: list[str]) -> tuple:
    """三分类打分：返回 (argmax 标签下标数组, softmax 概率矩阵 (n, 3))

    与 eval/compare_nli_models.mdeberta_score 同款实现（输入截断 512 token，
    max_position_embeddings=512，README 同款 truncation=True）。
    """
    import torch

    tok = payload["tokenizer"]
    model = payload["model"]
    inp = tok(docs, claims, truncation=True, max_length=512,
              padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inp).logits
    probs = torch.softmax(logits, dim=-1).numpy()
    labels = probs.argmax(axis=1)
    return labels, probs
