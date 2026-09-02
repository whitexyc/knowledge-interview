"""HHEM-2.1-Open 本地加载器（module-050 已验证路径，单一来源）

eval/compare_factcheck_models.py（module-050）与 rag/retrieval/factcheck_judge.py
（module-051）共享此加载函数，避免两份实现漂移（WP1 单一来源约束）。

双路径加载（2026-09-01 扩展）：
    - ONNX 路径（首选，FamiliarTools/HHEM-2.1-Open-onnx）：
      model.onnx + tokenizer.json + tokenizer_config.json + special_tokens_map.json
      直接 T5ForTokenClassification 导出（opset 17），输入 input_ids/attention_mask
      [batch, seq] 动态，输出 logits [batch, seq, 2]，分数 = softmax(logits[:, 0, :])[1]。
      无需 PyTorch / trust_remote_code，内存更省。

    - safetensors 路径（回退，原 module-050 实测路径）：
      HF_HUB_OFFLINE + get_class_from_dynamic_module 自定义代码
      + safetensors load_file + embed_tokens 键展开 + config foundation 本地 tokenizer

统一契约：返回带 predict(list[tuple(doc, claim)]) -> list[0-1 float] 的对象，
与官方分数口径一致（prompt 模板 <pad> Determine if the hypothesis is true
given the premise?\\n\\nPremise: {text1}\\n\\nHypothesis: {text2}）。
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

# ONNX 路径必备文件（2026-09-01：FamiliarTools/HHEM-2.1-Open-onnx）
HHEM_ONNX_REQUIRED_FILES = (
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

# 官方 prompt 模板（configuration_hhem_v2.py 一致：T5 padding + premise/hypothesis）
_HHEM_PROMPT = ("<pad> Determine if the hypothesis is true given the premise?\n\n"
                "Premise: {text1}\n\nHypothesis: {text2}")


class HHEMOnnxModel:
    """ONNX 推理包装：predict(pairs) -> 0-1 分数数组（class 1 = consistent）

    推理契约（README 实测）：
        - 输入 input_ids / attention_mask（[batch, seq] 动态）
        - 输出 logits [batch, seq, 2]
        - 一致性分数 = softmax(logits[:, 0, :])[:, 1]
    线程安全由调用方（HHEMJudge threading.Lock 串行）保证。
    """

    def __init__(self, model_dir: str):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"),
                                          providers=["CPUExecutionProvider"])
        self._input_names = [inp.name for inp in self._sess.get_inputs()]
        logger.info("HHEM-2.1-Open ONNX 模型加载完成: %s", model_dir)

    def predict(self, pairs: list) -> list:
        """批量打分：官方 prompt 模板 + softmax(logits[:,0,:])[1]"""
        import numpy as np

        if not pairs:
            return []
        texts = [self._prompt(doc, claim) for doc, claim in pairs]
        enc = self._tokenizer(texts, padding=True, truncation=True,
                              max_length=512, return_tensors="np")
        feeds = {name: np.ascontiguousarray(enc[name], dtype=np.int64) for name in self._input_names}
        logits = self._sess.run(None, feeds)[0]
        probs = _softmax(logits[:, 0, :])
        return [float(p) for p in probs[:, 1]]

    @staticmethod
    def _prompt(doc: str, claim: str) -> str:
        return _HHEM_PROMPT.format(text1=doc, text2=claim)


def _softmax(x):
    """逐行 softmax（numpy）"""
    import numpy as np

    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def load_hhem_model(ckpt_dir: str):
    """加载 HHEM-2.1-Open 模型（优先 ONNX，回退 safetensors）

    Args:
        ckpt_dir: 模型目录（含 model.onnx + tokenizer 三件套，或
                  config.json / model.safetensors / 自定义代码）

    Returns:
        predict(list[tuple(doc, claim)]) -> 0-1 分数数组的对象

    Raises:
        FileNotFoundError: 两种路径的必备文件都缺失时指出缺失路径
    """
    onnx_missing = [f for f in HHEM_ONNX_REQUIRED_FILES
                    if not os.path.isfile(os.path.join(ckpt_dir, f))]
    if not onnx_missing:
        return HHEMOnnxModel(ckpt_dir)

    missing = [f for f in HHEM_REQUIRED_FILES
               if not os.path.isfile(os.path.join(ckpt_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"模型目录不完整: {os.path.abspath(ckpt_dir)} "
            f"缺 ONNX 文件 {onnx_missing} 或缺原版文件 {missing}。"
            "请下载 FamiliarTools/HHEM-2.1-Open-onnx 或原版 HHEM-2.1-Open。"
        )

    # safetensors 回退路径（module-050 实测原始实现）
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
    logger.info("HHEM-2.1-Open safetensors 模型加载完成: %s", ckpt_dir)
    return model