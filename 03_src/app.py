import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import gradio as gr

import rag  # retrieval-augmented recommendation layer

# =========================
# Files
# =========================
GATE_W = "leaf_gate.pth"

if os.path.exists("disease_model.pth") and os.path.exists("disease_meta.json"):
    DIS_W = "disease_model.pth"
    DISEASE_BACKBONE = json.load(open("disease_meta.json"))["disease_backbone"]
else:
    DIS_W = "disease_baseline.pth"
    DISEASE_BACKBONE = "mobilenetv2_100"

gate_classes = json.load(open("gate_classes.json"))
disease_classes = json.load(open("disease_classes.json"))

NOT_LEAF_IDX = gate_classes.index("not_leaf")
NUM_G = len(gate_classes)
NUM_D = len(disease_classes)

GATE_THRESH = 0.60
DISEASE_THRESH = 0.55

# =========================
# Helpers
# =========================
def prettify(s):
    return (
        s.replace("___", " → ")
         .replace("__", " → ")
         .replace("_", " ")
         .replace("  ", " ")
    )

def build_model(name, n, dropout=0.4):
    model = timm.create_model(name, pretrained=False, num_classes=n)

    if isinstance(getattr(model, "classifier", None), nn.Linear):
        inf = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(inf, n))

    elif isinstance(getattr(model, "fc", None), nn.Linear):
        inf = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(inf, n))

    return model

def is_healthy_class(label):
    text = label.lower()
    disease_words = [
        "blight", "spot", "rust", "scab", "rot", "mold", "mildew",
        "virus", "bacterial", "mite", "septoria", "curl"
    ]
    return not any(w in text for w in disease_words)

def guidance_for(label):
    text = label.lower()

    if is_healthy_class(label):
        return "No clear disease pattern was detected. Keep monitoring the plant and maintain good airflow."

    if "early blight" in text:
        return "Remove infected leaves, avoid overhead watering, and improve airflow."
    if "late blight" in text:
        return "Remove heavily infected parts and avoid wet foliage. Expert review is recommended."
    if "rust" in text:
        return "Remove infected leaves and improve air circulation. A suitable fungicide may be needed."
    if "scab" in text:
        return "Remove fallen infected leaves and improve airflow around the plant."
    if "bacterial" in text:
        return "Avoid overhead watering, remove infected debris, and use clean tools."
    if "mold" in text or "mildew" in text:
        return "Improve ventilation, reduce humidity, and remove severely infected leaves."
    if "virus" in text or "curl" in text:
        return "There is usually no direct cure. Remove infected plants and control insect vectors."
    if "rot" in text:
        return "Remove infected parts, reduce excess moisture, and improve sanitation."

    return "Use a clearer image for confirmation and consult an agricultural expert before treatment."

# =========================
# Load models
# =========================
gate = build_model("mobilenetv2_100", NUM_G)
gate.load_state_dict(torch.load(GATE_W, map_location="cpu"))
gate.eval()

disease_model = build_model(DISEASE_BACKBONE, NUM_D)
disease_model.load_state_dict(torch.load(DIS_W, map_location="cpu"))
disease_model.eval()

tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# Prediction
# =========================
def predict(image):
    if image is None:
        return (
            "<div class='status-box neutral'>Upload a crop leaf image to begin.</div>",
            "<div class='result-box neutral'><b>No image yet.</b><br>"
            "Drop a clear photo of a single leaf on the left, then press <b>Analyze Leaf</b>.</div>",
            {},
            "—",
            "Your treatment guidance will appear here after analysis.",
            ""
        )

    x = tf(image).unsqueeze(0)

    with torch.no_grad():
        gate_probs = F.softmax(gate(x), dim=1)[0]

    not_leaf_conf = float(gate_probs[NOT_LEAF_IDX])

    # ---- Stage 1: leaf / not-leaf gate ----
    if not_leaf_conf >= GATE_THRESH:
        status = "<div class='status-box ok'>Image checked. Stage 1 (leaf gate) complete.</div>"
        result = (
            "<div class='result-box reject'>"
            "<span class='tag tag-reject'>Rejected</span>"
            "<div class='result-title'>This does not look like a plant leaf</div>"
            f"<div class='result-sub'>Not-a-leaf confidence: {not_leaf_conf * 100:.1f}%</div>"
            "</div>"
        )
        return (
            status,
            result,
            {},
            f"{not_leaf_conf * 100:.1f}% not-a-leaf",
            "Please upload a clear, well-lit photo of a single crop leaf.",
            ""
        )

    # ---- Stage 2: disease classifier ----
    with torch.no_grad():
        disease_probs = F.softmax(disease_model(x), dim=1)[0]

    order = sorted(range(NUM_D), key=lambda i: float(disease_probs[i]), reverse=True)
    top3 = {
        prettify(disease_classes[i]): float(disease_probs[i])
        for i in order[:3]
    }

    best_idx = order[0]
    best_label = disease_classes[best_idx]
    best_conf = float(disease_probs[best_idx])
    pretty_label = prettify(best_label)

    status = "<div class='status-box ok'>Analysis complete. Stage 2 (disease classifier) done.</div>"

    sources_html = ""

    if best_conf < DISEASE_THRESH:
        result = (
            "<div class='result-box warning'>"
            "<span class='tag tag-warning'>Low confidence</span>"
            f"<div class='result-title'>{pretty_label}</div>"
            f"<div class='result-sub'>Top guess at {best_conf * 100:.1f}% — not reliable enough.</div>"
            "</div>"
        )
        confidence = f"{best_conf * 100:.1f}%  (below {int(DISEASE_THRESH*100)}% threshold)"
        recommendation = (
            "The model is not confident enough. Try a clearer, closer photo with a single "
            "leaf filling the frame and even lighting."
        )
    else:
        healthy = is_healthy_class(best_label)
        tag_html = (
            "<span class='tag tag-ok'>Healthy</span>" if healthy
            else "<span class='tag tag-disease'>Disease detected</span>"
        )
        title = f"Healthy / no clear disease: {pretty_label}" if healthy else pretty_label
        result = (
            "<div class='result-box success'>"
            f"{tag_html}"
            f"<div class='result-title'>{title}</div>"
            f"<div class='result-sub'>Confidence {best_conf * 100:.1f}%</div>"
            "</div>"
        )
        confidence = f"{best_conf * 100:.1f}%"
        # ---- RAG: retrieve grounding + generate the recommendation ----
        recommendation, sources_html = rag.explain(
            pretty_label, best_label, best_conf, top3, image
        )

    return status, result, top3, confidence, recommendation, sources_html

# =========================
# UI
# =========================
custom_css = """
:root {
    --bg:        #f4faf6;
    --panel:     #ffffff;
    --ink:       #14532d;
    --ink-soft:  #4b6357;
    --line:      #d9ece0;
    --brand:     #1b8a4b;
    --brand-2:   #2fb86a;
}
/* fill the laptop width instead of a thin centered strip */
.gradio-container {
    max-width: 1280px !important;
    width: 100% !important;
    margin: 0 auto !important;
    background: var(--bg);
    font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.gradio-container .prose :is(h1,h2,h3) { color: var(--ink); }
/* ---------- header ---------- */
#app-header {
    display: flex; align-items: center; justify-content: center;
    gap: 12px; padding: 14px 8px 6px;
}
#app-header .logo { font-size: 30px; line-height: 1; }
#app-header h1 { margin: 0; font-size: 27px; font-weight: 800; color: var(--ink); letter-spacing: .3px; }
#app-header p  { margin: 2px 0 0; font-size: 14px; color: var(--ink-soft); }
#app-header .titles { display: flex; flex-direction: column; }
/* ---------- cards ---------- */
.card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(20,83,45,.04);
}
.card-label {
    font-size: 12px; font-weight: 700; letter-spacing: .6px;
    text-transform: uppercase; color: var(--ink-soft); margin-bottom: 10px;
}
/* upload column compact */
#analyze-btn { margin-top: 10px; }
/* ---------- status / result ---------- */
.status-box {
    padding: 10px 14px; border-radius: 10px; font-size: 14px; margin-bottom: 12px;
    border-left: 5px solid #4ca8b8; background: #e4f4f7; color: #155e6b;
}
.status-box.ok      { border-left-color: var(--brand); background: #e6f6ec; color: var(--ink); }
.status-box.neutral { border-left-color: #b8c4be; background: #eef2f0; color: #56635c; }
.result-box {
    padding: 16px 18px; border-radius: 14px; margin-bottom: 4px;
    border-left: 6px solid #b8c4be; background: #f3f6f4;
}
.result-box.success { border-left-color: var(--brand); background: #e6f6ec; }
.result-box.warning { border-left-color: #e0a800; background: #fff6da; }
.result-box.reject  { border-left-color: #d6453f; background: #fde6e5; }
.result-box.neutral { border-left-color: #b8c4be; background: #f3f6f4; }
.result-title { font-size: 21px; font-weight: 800; color: var(--ink); margin: 8px 0 2px; }
.result-sub   { font-size: 14px; color: var(--ink-soft); }
.tag {
    display: inline-block; font-size: 11px; font-weight: 800; letter-spacing: .5px;
    text-transform: uppercase; padding: 3px 9px; border-radius: 999px;
}
.tag-ok      { background: #cdeed8; color: #14532d; }
.tag-disease { background: #f7d9c0; color: #8a3b12; }
.tag-warning { background: #fbe7ad; color: #6b4e00; }
.tag-reject  { background: #f6c9c6; color: #7f1d1d; }
/* tighten Gradio defaults for a shorter page */
.gradio-container .gap { gap: 14px !important; }
.gradio-container label { color: var(--ink-soft) !important; }
#footer {
    text-align: center; color: #8a978f; font-size: 13px;
    margin: 18px 0 6px; padding-top: 14px; border-top: 1px solid var(--line);
}
/* responsive: stack on narrow screens, side-by-side on laptops */
@media (max-width: 900px) {
    #main-row { flex-direction: column !important; }
}
"""

with gr.Blocks(title="AGRONA", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div id="app-header">
        <div class="logo">🌿</div>
        <div class="titles">
            <h1>AGRONA</h1>
            <p>Crop leaf disease detection &amp; recommendation</p>
        </div>
    </div>
    """)

    with gr.Row(elem_id="main-row", equal_height=False):
        # ---- LEFT: input only (keeps the page short) ----
        with gr.Column(scale=5, min_width=320):
            with gr.Group(elem_classes="card"):
                gr.HTML("<div class='card-label'>Upload</div>")
                img = gr.Image(type="pil", label="Crop leaf image", height=300)
                btn = gr.Button("Analyze Leaf", variant="primary", size="lg", elem_id="analyze-btn")
                gr.HTML("<div style='font-size:13px;color:#8a978f;margin-top:8px;text-align:center;'>"
                        "One clear, well-lit leaf works best.</div>")

        # ---- RIGHT: all results, laid out horizontally ----
        with gr.Column(scale=7, min_width=420):
            status_box = gr.HTML("<div class='status-box neutral'>Waiting for an image…</div>")
            result_box = gr.HTML(
                "<div class='result-box neutral'><b>No image yet.</b><br>"
                "Upload a leaf and press Analyze Leaf.</div>"
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=220):
                    with gr.Group(elem_classes="card"):
                        gr.HTML("<div class='card-label'>Top-3 predictions</div>")
                        out_top = gr.Label(num_top_classes=3, show_label=False)
                with gr.Column(scale=1, min_width=220):
                    with gr.Group(elem_classes="card"):
                        gr.HTML("<div class='card-label'>Confidence</div>")
                        out_conf = gr.Textbox(show_label=False, interactive=False)
                        gr.HTML("<div class='card-label' style='margin-top:12px;'>Recommendation</div>")
                        out_recommendation = gr.Textbox(
                            show_label=False, interactive=False, lines=4
                        )
                        out_sources = gr.HTML("")

    gr.HTML("""
    <div id="footer">
        AGRONA is an AI support tool and does not replace expert agricultural diagnosis.
    </div>
    """)

    btn.click(
        predict,
        inputs=img,
        outputs=[status_box, result_box, out_top, out_conf, out_recommendation, out_sources]
    )
    # also run on upload so the result fills the right panel immediately
    img.change(
        predict,
        inputs=img,
        outputs=[status_box, result_box, out_top, out_conf, out_recommendation, out_sources]
    )

demo.launch(ssr_mode=False)
