"""HHEM-2.1-Open 本地加载器（module-050 已验证路径，单一来源）

eval/compare_factcheck_models.py（module-050）与 rag/retrieval/factcheck_judge.py
（module-051）共享此加载函数，避免两份实现漂移（WP1 单一来源约束）。

加载路径（module-050 实测验证，勿重新发明）：
    - HF_HUB_OFFLINE=1（huggingface.co 不可达，本机 hosts 映射 127.0.0.1）
    - get_class_from_dynamic_module 加载自定义远程代码
      （configuration_hhem_v2.HHEMv2Config / modeling_hhem_v2.HHEMv2ForSequenceClassification）
    - safetensors load_file 手动加载 + embed_tokens 键展开
      （检查点是 transformers 4.x 命名 t5.transformer.shared.weight，
      5.x 模型 embed_tokens 与 shared 绑定 → 展开成对键）
    - config.json foundation 已指向本地 models/flan-t5-base（tokenizer 依赖）
    - 分数与官方 README 参考值逐一吻合（0.0111/0.6474/...），权重加载正确性有外部基准背书
"""
import logging
import os

logger = logging.getLogger(__name__)

# HHEM 模型目录必备文件（缺失时报错指出路径，不静默通过）
HHEM_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "configuration_hhem_v2.py",
    "modeling_hhem_v2.py",
)


def load_hhem_model(ckpt_dir: str):
    """加载 HHEM-2.1-Open 模型（transformers 5.x CPU 适配）

    Args:
        ckpt_dir: 模型目录（含 config.json / model.safetensors / 自定义代码）

    Returns:
        已 eval() 的 HHEMv2ForSequenceClassification 实例
        （predict(list[tuple(doc, claim)]) → 0-1 分数数组，class 1 = consistent）

    Raises:
        FileNotFoundError: 模型目录缺失/不完整（指出缺失路径）
    """
    missing = [f for f in HHEM_REQUIRED_FILES
               if not os.path.isfile(os.path.join(ckpt_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"模型目录不完整: {os.path.abspath(ckpt_dir)} "
            f"缺少文件 {missing}。请先用下载脚本（hf-mirror）补齐。"
        )

    # 必须在任何 transformers/huggingface_hub 导入前设置（huggingface.co 不可达）
    os.environ["HF_HUB_OFFLINE"] = "1"

    from safetensors.torch import load_file
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    HHEMv2Config = get_class_from_dynamic_module(
        "configuration_hhem_v2.HHEMv2Config", ckpt_dir)
    HHEMv2ForSequenceClassification = get_class_from_dynamic_module(
        "modeling_hhem_v2.HHEMv2ForSequenceClassification", ckpt_dir)

    cfg = HHEMv2Config.from_pretrained(ckpt_dir)
    model = HHEMv2ForSequenceClassification(cfg)
    state = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    # 检查点（transformers 4.x）只有 shared，5.x 模型 embed_tokens 与 shared 绑定 → 展开
    state["t5.transformer.encoder.embed_tokens.weight"] = state["t5.transformer.shared.weight"]
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("HHEM-2.1-Open 模型加载完成: %s", ckpt_dir)
    return model
