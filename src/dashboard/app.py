import random

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.pipeline.annotate import MoodMosaicPipeline, GOEMO_LABELS
from src.models.personality_big5 import PersonalityRegressorBig5, BIG5_TRAITS as BIG5_TRAITS_BIG5

# Tone labels for the Stanford model (binary)
TONE_LABELS = ["impolite", "polite"]


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    # Provide all keys that annotate.py expects.
    # We still only *display* emotion from this pipeline; tone and
    # personality for the dashboard come from the new models below.
    paths = {
        "emotion": "experiments/checkpoints/emotion/best",
        "tone": "experiments/checkpoints/tone/best",
        "personality_sbert": "sentence-transformers/all-mpnet-base-v2",
        "personality_ckpt": "experiments/checkpoints/personality/best.pt",
    }
    return MoodMosaicPipeline(paths)


@st.cache_resource(show_spinner=True)
def load_big5_model():
    ckpt_path = "experiments/checkpoints/personality_big5/best_big5.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder_name = ckpt.get("encoder_name", "sentence-transformers/all-mpnet-base-v2")
    model = PersonalityRegressorBig5(encoder_name=encoder_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=True)
def load_tone_stanford():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
    )
    ckpt_path = "experiments/checkpoints/tone_stanford/best_stanford.bin"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return tokenizer, model, device


pipe = load_pipeline()
big5_model, big5_device = load_big5_model()
tone_tokenizer, tone_model, tone_device = load_tone_stanford()

# ------------------------------------------------------------------
# Sidebar: description + presets
# ------------------------------------------------------------------
st.sidebar.title("MoodMosaic")
st.sidebar.write(
    "Paste one message per line to see the emotion mosaic, "
    "tone labels from a Stanford Politeness model, "
    "and a Big Five personality sketch."
)

example_choice = st.sidebar.selectbox(
    "Load an example",
    [
        "None",
        "Positive day",
        "Frustrated and stressed",
        "Mixed polite and impolite",
        "Random personality sample",
    ],
)

# Example blocks
positive_day = """I am really excited about how this semester is going.
I feel proud of the work I put into my projects.
Thank you so much for helping me with the assignment.
I am grateful for my friends checking in on me today.
I am curious to see how far I can push this project."""

frustrated_block = """Today was exhausting and nothing worked the way I expected.
I am annoyed that my code keeps breaking right before the deadline.
This whole situation makes me nervous and a little scared.
I feel disappointed in myself for not planning better.
Please just fix this issue, I do not have the energy to debug again."""

mixed_tone_block = """Could you please review my pull request when you get time.
Thanks a lot for taking the extra effort on this task.
This is completely wrong, you did not follow the instructions.
Why did you ignore my message, this is really upsetting.
I appreciate your patience and all the feedback you gave me."""

# Big Five flavored sentences
personality_snippets = [
    "I love learning new ideas and exploring different research topics. I enjoy reading, reflecting, and trying creative approaches in my projects.",
    "I always plan my work carefully, make detailed schedules, and double check everything before I submit.",
    "I feel energized when I am around people and I like being the one who starts conversations in group projects.",
    "I try to be patient and supportive with my teammates, and I care a lot about keeping a friendly atmosphere.",
    "I often worry about results and deadlines, and sometimes I feel stressed or anxious when things are uncertain.",
]

if example_choice == "Positive day":
    default_text = positive_day
elif example_choice == "Frustrated and stressed":
    default_text = frustrated_block
elif example_choice == "Mixed polite and impolite":
    default_text = mixed_tone_block
elif example_choice == "Random personality sample":
    k = random.randint(3, len(personality_snippets))
    chosen = random.sample(personality_snippets, k=k)
    default_text = "\n".join(chosen)
else:
    default_text = ""

st.sidebar.markdown("---")
st.sidebar.caption(
    "Models: RoBERTa (emotion on GoEmotions), "
    "DistilBERT (tone on Stanford Politeness), "
    "SBERT + MLP (personality on Essays Big Five)."
)

# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
st.title("MoodMosaic Dashboard")

col_input, col_meta = st.columns([3, 2])

with col_input:
    st.subheader("Input text")
    raw_text = st.text_area(
        "Enter one message per line:",
        value=default_text,
        height=200,
        label_visibility="collapsed",
    )
    analyze_btn = st.button("Analyze")

with col_meta:
    st.subheader("Session info")
    st.write("Emotion device:", str(pipe.device))
    st.write("Tone device:", str(tone_device))
    st.write("Personality device:", str(big5_device))
    st.write("Number of emotion labels:", len(GOEMO_LABELS))
    st.write("Tone classes:", ", ".join(TONE_LABELS))
    st.write("Personality traits:", ", ".join(BIG5_TRAITS_BIG5))

# ------------------------------------------------------------------
# Helper functions for plots and table
# ------------------------------------------------------------------
def build_emotion_summary(per_message):
    probs = np.array([row["emotion_probs"] for row in per_message])
    mean_probs = probs.mean(axis=0)
    dom_idx = probs.argmax(axis=1)
    dom_labels = [GOEMO_LABELS[i] for i in dom_idx]
    return mean_probs, dom_labels


def build_personality_series_big5(scores_vec):
    # scores_vec: numpy array of shape (5,) from Big Five model (roughly in [-1, 1])
    scores_01 = (scores_vec + 1.0) / 2.0
    return pd.Series(scores_01, index=BIG5_TRAITS_BIG5)


def emotion_radar(mean_probs):
    df = pd.DataFrame(
        {"emotion": GOEMO_LABELS, "score": mean_probs},
    )
    df = df.sort_values("score", ascending=False).head(8)
    fig = px.line_polar(
        df,
        r="score",
        theta="emotion",
        line_close=True,
        range_r=[0, df["score"].max() * 1.1 if df["score"].max() > 0 else 1],
    )
    fig.update_traces(fill="toself")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig


def personality_radar(series):
    df = pd.DataFrame({"trait": series.index, "score": series.values})
    fig = px.line_polar(
        df,
        r="score",
        theta="trait",
        line_close=True,
        range_r=[0, 1],
    )
    fig.update_traces(fill="toself")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig


def predict_tone_stanford(messages):
    enc = tone_tokenizer(
        messages,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(tone_device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = tone_model(**enc)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
    return preds.cpu().numpy()


def predict_big5_scores(messages):
    # messages: list of strings, we join them to one block
    joined = " ".join(messages)
    with torch.no_grad():
        preds = big5_model([joined]).cpu().numpy()[0]
    return preds


def build_per_message_df(texts, dom_emotions, tone_labels):
    rows = []
    for text, emo_label, tone_idx in zip(texts, dom_emotions, tone_labels):
        tone_label = TONE_LABELS[int(tone_idx)] if 0 <= tone_idx < len(TONE_LABELS) else str(
            tone_idx
        )
        rows.append(
            {
                "Text": text,
                "Dominant emotion": emo_label,
                "Tone": tone_label,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------------
if analyze_btn:
    lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
    if not lines:
        st.warning("Please enter at least one non empty line.")
    else:
        with st.spinner("Running MoodMosaic pipeline..."):
            # Use pipeline for emotion only
            result = pipe.analyze_messages(lines)

        per_message = result["per_message"]
        emo_mean, dom_emotions = build_emotion_summary(per_message)

        # New tone and personality from real models
        tone_preds = predict_tone_stanford(lines)
        big5_vec = predict_big5_scores(lines)
        personality = build_personality_series_big5(big5_vec)

        df_msgs = build_per_message_df(lines, dom_emotions, tone_preds)

        # Quick text summary
        top_emo_idx = emo_mean.argsort()[::-1][:3]
        top_emos = [GOEMO_LABELS[i] for i in top_emo_idx]
        top_trait_idx = personality.values.argsort()[::-1][:2]
        top_traits = [personality.index[i] for i in top_trait_idx]

        st.markdown("### Summary")
        st.write(
            f"Top emotions across all messages: **{', '.join(top_emos)}**."
        )
        st.write(
            "Strongest personality traits (approximate, Essays Big Five): "
            f"**{', '.join(top_traits)}**."
        )

        # Tabs: Summary vs Per message details
        tab_summary, tab_details = st.tabs(["Overview", "Per message details"])

        with tab_summary:
            col_emo, col_pers = st.columns(2)
            with col_emo:
                st.subheader("Emotion mosaic")
                st.plotly_chart(emotion_radar(emo_mean), use_container_width=True)
            with col_pers:
                st.subheader("Personality profile (Big Five)")
                st.plotly_chart(personality_radar(personality), use_container_width=True)

        with tab_details:
            st.subheader("Per message analysis")

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                tone_filter = st.multiselect(
                    "Filter by tone (Stanford)",
                    options=TONE_LABELS,
                    default=TONE_LABELS,
                )
            with col_f2:
                emo_filter = st.multiselect(
                    "Filter by dominant emotion",
                    options=sorted(set(dom_emotions)),
                    default=sorted(set(dom_emotions)),
                )

            df_filtered = df_msgs[
                df_msgs["Tone"].isin(tone_filter)
                & df_msgs["Dominant emotion"].isin(emo_filter)
            ]
            st.dataframe(df_filtered, use_container_width=True, height=300)

            with st.expander("Show full table (no filters)"):
                st.dataframe(df_msgs, use_container_width=True, height=300)
else:
    st.info(
        "Enter one message per line on the left, optionally choose a preset example "
        "in the sidebar, and click **Analyze** to see the mood mosaic."
    )
