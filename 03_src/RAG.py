"""
AGRONA — retrieval-augmented recommendation layer.
Flow (the customer journey on the site):
    image -> leaf gate -> disease classifier -> [RAG] retrieve grounding -> generate advice
Design goals:
  * Works out of the box on a free CPU Space with NO API keys and NO heavy
    downloads. Retrieval is lexical; generation is a deterministic, grounded
    composer that only ever uses text from the knowledge base (no hallucination).
  * Upgrades automatically:
      - if `sentence-transformers` is installed -> semantic (embedding) retrieval
      - if ANTHROPIC_API_KEY is set as a Space secret -> a hosted Claude model
        writes the advice, grounded ONLY in the retrieved KB entry. If the image
        is passed, the call is multimodal (VLM) and adds a one-line visual check.
"""

import os, re, json, base64, io

# ---------------------------------------------------------------- knowledge base
_KB_PATH = os.path.join(os.path.dirname(__file__), "kb_diseases.json")
with open(_KB_PATH, "r", encoding="utf-8") as f:
    KB = json.load(f)

_STOP = {"leaf", "leaves", "plant", "the", "and", "of", "a", "to", "with"}


def _tokens(text):
    return [w for w in re.findall(r"[a-z]+", str(text).lower()) if w not in _STOP]


def _doc_text(entry):
    return " ".join([
        entry.get("crop", ""), entry.get("condition", ""), entry.get("type", ""),
        entry.get("symptoms", ""), entry.get("management", ""), entry.get("note", ""),
        entry.get("label", ""),
    ])


# ---------------------------------------------------------------- optional embeddings
_EMBEDDER = None
_KB_VECS = None


def _try_load_embedder():
    """Load a small sentence-transformer if available; silently skip otherwise."""
    global _EMBEDDER, _KB_VECS
    if _EMBEDDER is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        labels = list(KB.keys())
        _KB_VECS = (labels, _EMBEDDER.encode(
            [_doc_text(KB[l]) for l in labels], normalize_embeddings=True
        ))
    except Exception:
        _EMBEDDER = False  # mark "tried and unavailable"


# ---------------------------------------------------------------- retrieval
def retrieve(raw_label, top3_labels=None):
    """Return (entry, score, method) for the best-matching KB record."""
    # 1) exact / normalized key match — the predicted label IS a KB key in the
    #    normal case, so this is both fastest and most accurate.
    if raw_label in KB:
        return KB[raw_label], 1.0, "exact"
    norm = {k.lower().strip(): k for k in KB}
    if raw_label.lower().strip() in norm:
        k = norm[raw_label.lower().strip()]
        return KB[k], 1.0, "exact"

    # 2) semantic retrieval if embeddings are available
    _try_load_embedder()
    if _EMBEDDER:
        try:
            import numpy as np
            labels, mat = _KB_VECS
            q = _EMBEDDER.encode([raw_label], normalize_embeddings=True)[0]
            sims = mat @ q
            i = int(np.argmax(sims))
            return KB[labels[i]], float(sims[i]), "embedding"
        except Exception:
            pass

    # 3) lexical fallback — token overlap, weighted toward crop + condition words
    query = set(_tokens(raw_label))
    for t in (top3_labels or []):
        query |= set(_tokens(t))
    best, best_score = None, -1.0
    for label, entry in KB.items():
        doc = set(_tokens(label)) | set(_tokens(entry.get("crop", ""))) \
            | set(_tokens(entry.get("condition", "")))
        if not doc:
            continue
        score = len(query & doc) / len(doc)
        if score > best_score:
            best, best_score = entry, score
    return best, max(best_score, 0.0), "lexical"


# ---------------------------------------------------------------- hosted generation (optional VLM/LLM)
def _image_to_b64(image, max_side=512):
    try:
        from PIL import Image  # noqa
        img = image.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((int(w * s), int(h * s)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _hosted_generate(entry, pretty_label, confidence, top3, image=None):
    """Ask a hosted Claude model to write grounded advice. Returns text or None."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import requests
        model = os.environ.get("AGRONA_LLM_MODEL", "claude-haiku-4-5-20251001")
        context = json.dumps(entry, ensure_ascii=False, indent=2)
        top3_txt = ", ".join(f"{k} ({v*100:.0f}%)" for k, v in (top3 or {}).items())
        prompt = (
            "You are an agricultural assistant. Using ONLY the knowledge-base "
            "entry below, write 2-3 short, practical sentences for a farmer about "
            "the predicted condition. Do not invent facts beyond the entry. End by "
            "noting that an expert should confirm before any chemical treatment.\n\n"
            f"Predicted: {pretty_label} (confidence {confidence*100:.0f}%).\n"
            f"Model's top-3: {top3_txt}.\n\n"
            f"Knowledge-base entry:\n{context}"
        )
        content = [{"type": "text", "text": prompt}]
        b64 = _image_to_b64(image) if image is not None else None
        if b64:
            content.insert(0, {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
            content[1]["text"] = (
                "Briefly note (one clause) whether the leaf in the image is "
                "visually consistent with the predicted condition, then:\n\n" + prompt
            )
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={"model": model, "max_tokens": 300,
                  "messages": [{"role": "user", "content": content}]},
            timeout=30,
        )
        r.raise_for_status()
        parts = r.json().get("content", [])
        txt = "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
        return txt or None
    except Exception:
        return None


# ---------------------------------------------------------------- offline grounded composer
def _offline_generate(entry, pretty_label, confidence):
    bits = []
    cond_type = entry.get("type", "")
    if cond_type == "healthy":
        bits.append(entry.get("symptoms", ""))
        bits.append(entry.get("management", ""))
    else:
        bits.append(f"Likely {entry.get('crop','')} — {entry.get('condition','')} "
                    f"({cond_type}).")
        if entry.get("symptoms"):
            bits.append("What you may see: " + entry["symptoms"])
        if entry.get("management"):
            bits.append("Suggested steps: " + entry["management"])
        bits.append("Confirm with an agricultural expert before any chemical treatment.")
    if confidence < 0.70 and cond_type != "healthy":
        bits.insert(0, "This is a tentative reading — please verify.")
    return " ".join(b for b in bits if b)


# ---------------------------------------------------------------- public API
def explain(pretty_label, raw_label, confidence, top3=None, image=None):
    """
    Returns (recommendation_text, sources_html).
    `pretty_label` is the display label, `raw_label` is the model's class string.
    """
    entry, score, method = retrieve(raw_label, list((top3 or {}).keys()))
    if entry is None:
        return ("Use a clearer image for confirmation and consult an agricultural "
                "expert before treatment.", "")

    text = _hosted_generate(entry, pretty_label, confidence, top3, image) \
        or _offline_generate(entry, pretty_label, confidence)

    gen = "Claude (grounded)" if os.environ.get("ANTHROPIC_API_KEY") else "offline composer"
    sources_html = (
        "<div class='card-label' style='margin-top:12px;'>Knowledge source</div>"
        "<div style='font-size:13px;color:#4b6357;line-height:1.55;'>"
        f"<b>{entry.get('crop','')} → {entry.get('condition','')}</b> "
        f"<span style='color:#8a978f;'>· {entry.get('type','')}, "
        f"severity {entry.get('severity','')}</span><br>"
        f"<span style='color:#8a978f;'>retrieval: {method} · generation: {gen}</span>"
        "</div>"
    )
    return text, sources_html
