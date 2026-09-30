"""Gradio web app for the E-commerce Review Sentiment & CS Flagging System.

Usage:
    python app.py              # http://127.0.0.1:7860
    python app.py --share      # temporary public link
"""

import argparse
import html

import gradio as gr

from src import config
from src.inference import AVAILABLE_MODELS, analyze_review, ensure_distilbert_weights

EXAMPLES = [
    ["The product arrived completely broken and smells dangerous, I want a refund now!", "DistilBERT"],
    ["The product is excellent. Great quality and fast delivery!", "DistilBERT"],
    ["The product quality is disappointing. I don't really like it.", "DistilBERT"],
    ["This item looks fake and I want a refund immediately.", "BiLSTM"],
    ["The refund process was simple and the support team was great.", "BiLSTM"],
]  # fmt: skip


def confidence_bar(confidence: float, sentiment: str) -> str:
    color = "#16a34a" if sentiment == config.ID2LABEL[1] else "#dc2626"
    pct = confidence * 100
    return f"""
<div style="font-family: inherit;">
  <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
    <span>{html.escape(sentiment)}</span><strong>{pct:.1f}%</strong>
  </div>
  <div style="background:#e5e7eb; border-radius:6px; height:14px; overflow:hidden;">
    <div style="width:{pct:.1f}%; background:{color}; height:100%;"></div>
  </div>
</div>"""


def predict(text: str, model_name: str):
    if not text or not text.strip():
        raise gr.Error("Please enter a review.")
    try:
        result = analyze_review(text, model_name)
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc

    pred, flag = result.prediction, result.flag
    keywords = ", ".join(flag.matched_keywords) or "none"
    details = f"**Emergency keywords:** {keywords}  \n**Why:** {flag.reason}"
    return pred.sentiment, confidence_bar(pred.confidence, pred.sentiment), flag.label, details


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Review Sentiment & CS Flagging") as demo:
        gr.Markdown(
            "# 🛒 E-commerce Review Sentiment & CS Flagging\n"
            "Classify a product review and automatically escalate urgent complaints "
            f"(Negative + confidence > {config.FLAG_CONFIDENCE_THRESHOLD:.0%} + emergency keyword) "
            "to Customer Service."
        )
        with gr.Row():
            with gr.Column():
                review = gr.Textbox(label="Review Text", lines=5, placeholder="Paste a customer review...")
                model = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS), value="DistilBERT", label="Model Selection"
                )
                submit = gr.Button("Analyze", variant="primary")
            with gr.Column():
                sentiment = gr.Textbox(label="Sentiment", interactive=False)
                confidence = gr.HTML(label="Confidence Score")
                cs_flag = gr.Textbox(label="CS Flag", interactive=False)
                details = gr.Markdown()

        outputs = [sentiment, confidence, cs_flag, details]
        submit.click(predict, inputs=[review, model], outputs=outputs)
        review.submit(predict, inputs=[review, model], outputs=outputs)
        gr.Examples(examples=EXAMPLES, inputs=[review, model])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    try:
        ensure_distilbert_weights()  # first run only: fetch weights from the GitHub Release
    except FileNotFoundError as exc:
        print(f"Warning: {exc}\nDistilBERT is unavailable; BiLSTM still works.")
    build_demo().launch(server_port=args.port, share=args.share)
