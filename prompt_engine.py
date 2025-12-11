# ml_prompt_engine.py
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np



# Optional T5 paraphrasing model
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
except Exception:
    t5_tokenizer = None
    t5_model = None


# OPTION 1 -- USING THE BERT-BASED SENTENCE TRANSFORMERS

# Optional BERT-like sentence embedding model
try:
    from sentence_transformers import SentenceTransformer

    # You can swap this for another model if you like, e.g. "all-mpnet-base-v2"
    _bert_model: SentenceTransformer | None = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _bert_model = None




# OPTION 2 -- USING THE HANMADE - BAG OF WORDS APPROACH
# ---------------------------
# Basic text utilities
# ---------------------------

STOPWORDS = set("""
a an the and or but so to of for in on with at by from as that which who whom whose
be am is are was were been being have has had do does did will would should can could
""".split())


def compact_sentence(s: str) -> str:
    """
    Light-weight compressor:
    - collapse spaces
    - apply a few textual replacements
    - drop stopwords
    """
    s = " ".join(s.split())
    lower = s.lower()

    replacements = {
        "please ": "",
        " kindly ": "",
        " in order to ": " to ",
        " basically ": " ",
        " actually ": " ",
        " really ": " ",
        " step-by-step ": " stepwise ",
    }
    for k, v in replacements.items():
        lower = lower.replace(k, v)

    tokens = lower.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def _tokenize(text: str) -> List[str]:
    return [t for t in text.replace("\n", " ").split() if t]


def _split_sentences(text: str) -> List[str]:
    # very rough sentence split
    parts = []
    for chunk in text.split("\n"):
        parts.extend(
            [p.strip() for p in chunk.replace("?", ".").replace("!", ".").split(".") if p.strip()]
        )
    return parts


# ---------------------------
# Feature extraction
# ---------------------------

def compute_features(text: str) -> Dict[str, float]:
    """
    Richer feature set:

    - tokens
    - type_token_ratio
    - avg_sentence_len
    - stopword_ratio
    - sections (label-like lines)
    """
    tokens = _tokenize(text)
    num_tokens = len(tokens)
    unique_tokens = set(tokens)

    if num_tokens == 0:
        return {
            "tokens": 0.0,
            "type_token_ratio": 0.0,
            "avg_sentence_len": 0.0,
            "stopword_ratio": 0.0,
            "sections": 0.0,
        }

    # Type-token ratio (vocabulary richness)
    ttr = len(unique_tokens) / num_tokens

    # Sentences & average length
    sentences = _split_sentences(text)
    sent_lens = [len(_tokenize(s)) for s in sentences] or [0]
    avg_sent_len = float(sum(sent_lens) / len(sent_lens))

    # Stopword ratio
    stop_count = sum(1 for t in tokens if t.lower() in STOPWORDS)
    stop_ratio = stop_count / num_tokens

    # Sections: heuristic based on ":" early in the line
    sections = 0
    for line in text.splitlines():
        if ":" in line[:30]:
            sections += 1

    return {
        "tokens": float(num_tokens),
        "type_token_ratio": float(ttr),
        "avg_sentence_len": float(avg_sent_len),
        "stopword_ratio": float(stop_ratio),
        "sections": float(sections),
    }


# ---------------------------
# Similarity functions
# ---------------------------

def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(_tokenize(a.lower()))
    set_b = set(_tokenize(b.lower()))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)



# FOR THE BERT MODEL USAGE
def _bert_embed(text: str) -> np.ndarray:
    """
    BERT-style sentence embedding using sentence-transformers.

    - Returns a dense normalized vector
    - Falls back to a zero vector if the model isn't available
    """
    if _bert_model is None:
        return np.zeros(1, dtype=float)

    # sentence-transformers returns a 1D numpy array
    emb = _bert_model.encode(text, normalize_embeddings=True)
    return np.asarray(emb, dtype=float)


def _embed(text: str) -> np.ndarray:
    """
    Lightweight hashed bag-of-words "embedding":

    - No external models
    - Fixed small dimensionality
    - Normalized for cosine similarity
    """
    tokens = _tokenize(text.lower())
    if not tokens:
        return np.zeros(1, dtype=float)

    dim = 256  # small & cheap
    vec = np.zeros(dim, dtype=float)
    for t in tokens:
        idx = hash(t) % dim
        vec[idx] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec



# FOR THE BERT MODEL USAGE -SEMATIC SIMILARITY
def semantic_similarity(a: str, b: str) -> float:
    """
    Semantic similarity between two prompts using BERT-style sentence embeddings.

    - Primary: cosine similarity on BERT embeddings (sentence-transformers)
    - Fallback: hashed bag-of-words + cosine, then Jaccard if needed
    """
    # If we have a BERT model, use it
    if _bert_model is not None:
        va = _bert_embed(a)
        vb = _bert_embed(b)
    else:
        # Fallback to the old lightweight embedding
        va = _embed(a)
        vb = _embed(b)

    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        # Last-resort fallback
        return jaccard_similarity(a, b)

    return float(np.dot(va, vb) / denom)


def t5_simplify(text: str, max_len: int = 128) -> str:
    """
    Uses T5 to paraphrase/simplify text while preserving meaning.
    If the model is unavailable, returns the original text.
    """
    try:
        if t5_model is None or t5_tokenizer is None or not text.strip():
            return text

        input_ids = t5_tokenizer(
            "paraphrase: " + text,
            return_tensors="pt",
            truncation=True
        ).input_ids

        outputs = t5_model.generate(
            input_ids,
            max_length=max_len,
            num_beams=4,
            early_stopping=True
        )

        return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception:
        return text


# ---------------------------
# Energy model
# ---------------------------

def predict_energy_kwh(
    features: Dict[str, float],
    layers: int,
    training_hours: float,
    flops_per_hour: float,
) -> float:
    """
    Energy estimate that *does* depend on prompt length.

    Base is proportional to layers * time * FLOPs * normalized token length,
    then adjusted by a complexity factor.
    """
    
    tokens = features["tokens"]
    # normalize tokens so we don't explode energy
    token_factor = 1.0 + min(tokens / 200.0, 2.0)   # 1.0–3.0 range

    base = layers * training_hours * flops_per_hour * 1e-23 * token_factor

    complexity = (
        0.3 * features["type_token_ratio"]
        + 0.3 * (features["avg_sentence_len"] / 40.0)
        + 0.2 * features["sections"]
        + 0.2 * (1.0 - features["stopword_ratio"])
    )

    complexity = max(0.0, min(complexity, 2.0))
    return float(base * (1.0 + complexity))


# ---------------------------
# Prompt variants & optimizer
# ---------------------------

@dataclass
class PromptVariant:
    role: str
    context: str
    expectations: str

    def combined(self) -> str:
        parts = []
        if self.role.strip():
            parts.append(f"Role: {self.role.strip()}")
        if self.context.strip():
            parts.append(f"Context: {self.context.strip()}")
        if self.expectations.strip():
            parts.append(f"Expectations: {self.expectations.strip()}")
        return "\n\n".join(parts)


def _compress_field(text: str, level: str) -> str:
    """
    level ∈ {"none", "mild", "strong"}
    """
    text = text.strip()
    if not text:
        return text

    if level == "none":
        return text

    base = compact_sentence(text)
    words = base.split()
    if not words:
        return base

    if level == "mild":
        keep = max(1, int(len(words) * 0.8))
    else:  # "strong"
        keep = max(1, int(len(words) * 0.5))

    return " ".join(words[:keep])


def generate_variants(prompt: PromptVariant) -> List[PromptVariant]:
    """
    Generate a small grid of compressed variants for (role, context, expectations).
    Use milder compression for expectations to keep user-facing text readable.
    """
    levels_role = ["none", "mild", "strong"]
    levels_context = ["none", "mild", "strong"]
    levels_expect = ["none", "mild"] 
    
    variants: List[PromptVariant] = []
    for lr in levels_role:
        for lc in levels_context:
            for le in levels_expect:
                variants.append(
                    PromptVariant(
                        role=_compress_field(prompt.role, lr),
                        context=_compress_field(prompt.context, lc),
                        expectations=_compress_field(prompt.expectations, le),
                    )
                )
    return variants


def optimize_prompt(
    prompt: PromptVariant,
    layers: int,
    training_hours: float,
    flops_per_hour: float,
    similarity_min: float = 0.90,  # can change this if you want the improved prompt to stay very close to the original meaning
) -> Dict[str, Any]:
    """
    Multi-variant optimizer:

    - Generate several compressed variants.
    - Predict energy for each variant.
    - Compute semantic similarity to the original.
    - Prefer prompts that:
        * stay close to the target similarity_min
        * have lower predicted energy
        * use fewer tokens
    """
    # ----- Original prompt baseline -----
    orig_text = prompt.combined()
    before_features = compute_features(orig_text)
    before_kwh = predict_energy_kwh(
        before_features, layers, training_hours, flops_per_hour
    )

    # ----- Generate candidate compressed variants -----
    variants = generate_variants(prompt)
    candidates: List[
        Tuple[float, float, float, PromptVariant, Dict[str, float], float, float]
    ] = []

    for cand in variants:
        cand_text = cand.combined()
        after_features = compute_features(cand_text)
        after_kwh = predict_energy_kwh(
            after_features, layers, training_hours, flops_per_hour
        )

        sim = semantic_similarity(orig_text, cand_text)

        # (distance_to_target_similarity, energy, tokens, variant, features, energy, sim)
        distance_to_target = abs(similarity_min - sim)
        tokens = after_features["tokens"]

        candidates.append(
            (distance_to_target, after_kwh, tokens, cand, after_features, after_kwh, sim)
        )

    # ----- Pick best candidate by (distance to similarity target, energy, tokens) -----
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, _, best_variant, best_features, best_energy, best_sim = candidates[0]

    # -------------------------------------------
    # Optional T5 paraphrasing of the best variant
    # -------------------------------------------
    # Only try if T5 is available
    if t5_model is not None and t5_tokenizer is not None:
        # Paraphrase each field separately to keep structure
        t5_role = t5_simplify(best_variant.role)
        t5_context = t5_simplify(best_variant.context)
        t5_expectations = t5_simplify(best_variant.expectations)

        t5_variant = PromptVariant(
            role=t5_role,
            context=t5_context,
            expectations=t5_expectations,
        )

        t5_combined = t5_variant.combined()

        # Compute similarity between original and T5 output
        sim_t5 = semantic_similarity(orig_text, t5_combined)

        # Only accept T5 version if similarity is high enough
        if sim_t5 >= similarity_min:
            # Recompute features and energy for the T5 version
            feats_t5 = compute_features(t5_combined)
            energy_t5 = predict_energy_kwh(
                feats_t5, layers, training_hours, flops_per_hour
            )

            # Only switch to T5 version if it's not more energy-expensive
            if energy_t5 <= best_energy:
                best_variant = t5_variant
                best_features = feats_t5
                best_energy = energy_t5
                best_sim = sim_t5

    # ----- Final result -----
    return {
        "improved": best_variant,              # PromptVariant object
        "predicted_kwh_before": before_kwh,
        "predicted_kwh_after": best_energy,
        "similarity": best_sim,
        "features_before": before_features,
        "features_after": best_features,
    }
