"""Model construction, faithful to Knowledgator/GLiClass train.py (main @ 6a00f80, 16 Oct 2026).

The pip package `gliclass` (0.1.21) ships GLiClassModel / GLiClassModelConfig / GLiClassDataset /
Trainer but not train.py itself, so the backbone-specific bits of train.py (special tokens and their
embedding init) are reproduced here unchanged in behaviour.
"""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from gliclass import GLiClassModel, GLiClassModelConfig

SPECIAL_TOKENS = ["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"]
# GLiClass train.py: SPECIAL_TOKEN_INIT_WORDS
SPECIAL_TOKEN_INIT_WORDS = {"<<LABEL>>": "label", "<<SEP>>": "text", "<<EXAMPLE>>": "example"}


def check_backbone_support(encoder_name: str):
    """GLiClassUniEncoder loads non-DeBERTa/T5 backbones with AutoModel.from_pretrained(name) and no
    trust_remote_code, so UltraBERT needs native transformers support (>= 5.20)."""
    cfg = AutoConfig.from_pretrained(encoder_name)
    if "ultrabert" in encoder_name.lower() and "ultrabert" not in CONFIG_MAPPING_NAMES:
        raise RuntimeError("UltraBERT needs transformers>=5.20 (native support); GLiClass does not pass "
                           "trust_remote_code. Install transformers==5.20.2.")
    return cfg


def init_special_token_embeddings(embeddings, text_tokenizer, new_ids, num_pretrained_rows):
    """Copied in behaviour from GLiClass train.py: seed each new token from a descriptive word's embedding,
    rescaled to the median norm of the pretrained vocabulary."""
    weight = embeddings.weight
    with torch.no_grad():
        target_norm = weight[:num_pretrained_rows].float().norm(dim=-1).median()
        for token, token_id in new_ids.items():
            word = SPECIAL_TOKEN_INIT_WORDS.get(token, token.strip("<>").lower())
            word_ids = [i for i in text_tokenizer(word, add_special_tokens=False)["input_ids"]
                        if i < num_pretrained_rows]
            if not word_ids:
                continue
            vector = weight[word_ids].float().mean(dim=0)
            weight[token_id] = (vector * target_norm / vector.norm()).to(weight.dtype)
            print(f"Initialized {token} (id {token_id}) from '{word}', norm {target_norm:.3f}")


def freeze_unused(model) -> list[str]:
    """`logit_scale` only enters the forward pass when config.normalize_features is True. Otherwise it
    never gets a gradient, and DDP with find_unused_parameters=False fails (CPU smoke test). Freezing it
    is cheaper than turning on unused-parameter detection."""
    frozen = []
    if not getattr(model.config, "normalize_features", False):
        for name, p in model.named_parameters():
            if name.endswith("logit_scale"):
                p.requires_grad_(False)
                frozen.append(name)
    return frozen


def build_model(mcfg: dict):
    """Return (model, tokenizer). mcfg = the `model:` block of a stage config."""
    if mcfg.get("init_from"):
        path = mcfg["init_from"]
        model = GLiClassModel.from_pretrained(
            path,
            focal_loss_alpha=mcfg.get("focal_loss_alpha", 0.75),
            focal_loss_gamma=mcfg.get("focal_loss_gamma", -1),
            focal_loss_reduction=mcfg.get("focal_loss_reduction", "none"),
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Initialized from GLiClass checkpoint {path}")
        freeze_unused(model)
        return model, tokenizer

    name = mcfg["encoder_model_name"]
    encoder_config = check_backbone_support(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    config = GLiClassModelConfig(
        encoder_config=encoder_config,
        encoder_model=name,
        label_model_name=None,
        label_model_config=None,
        class_token_index=len(tokenizer),
        text_token_index=len(tokenizer) + 1,
        example_token_index=len(tokenizer) + 2,
        pooling_strategy=mcfg.get("pooler_type", "first"),
        class_token_pooling=mcfg.get("class_token_pooling", "first"),
        scorer_type=mcfg.get("scorer_type", "mlp"),
        use_lstm=mcfg.get("use_lstm", False),
        focal_loss_alpha=mcfg.get("focal_loss_alpha", 0.75),
        focal_loss_gamma=mcfg.get("focal_loss_gamma", -1),
        focal_loss_reduction=mcfg.get("focal_loss_reduction", "none"),
        contrastive_loss_coef=mcfg.get("contrastive_loss_coef", 0.0),
        normalize_features=mcfg.get("normalize_features", False),
        extract_text_features=mcfg.get("extract_text_features", False),
        architecture_type=mcfg.get("architecture_type", "uni-encoder"),
        prompt_first=mcfg.get("prompt_first", True),
        squeeze_layers=False,
        layer_wise=False,
        encoder_layer_id=-1,
        shuffle_labels=mcfg.get("shuffle_labels", True),
        dropout=mcfg.get("dropout", 0.3),
        use_segment_embeddings=False,
        multimodal=False,
        max_num_classes=mcfg.get("max_num_classes", 50),
    )
    if config.architecture_type != "uni-encoder":
        raise NotImplementedError("Stage plan starts with uni-encoder; bi-encoder is a later experiment.")
    model = GLiClassModel(config, from_pretrained=True).to(dtype=torch.float32)

    num_pretrained_rows = len(tokenizer)
    added = [w for w in SPECIAL_TOKENS if w not in tokenizer.get_vocab()]
    tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    embeddings = model.resize_token_embeddings(len(tokenizer))
    init_special_token_embeddings(embeddings, tokenizer,
                                  {w: tokenizer.convert_tokens_to_ids(w) for w in added}, num_pretrained_rows)
    freeze_unused(model)
    return model, tokenizer


def no_decay_norm_params(model) -> set[str]:
    """UltraBERT uses RMSNorm, which transformers' ALL_LAYERNORM_LAYERS (nn.LayerNorm) does not cover, so
    GLiClass's optimizer would decay the norm gains. GLiClass's create_optimizer honours
    `model.no_weight_decay_param_names`, so norm weights are excluded from decay here."""
    names = {n for n, _ in model.named_parameters() if "norm" in n.lower()}
    model.no_weight_decay_param_names = set(getattr(model, "no_weight_decay_param_names", set())) | names
    return names
