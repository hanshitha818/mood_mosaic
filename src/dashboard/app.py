import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline.annotate import (
    MoodMosaicPipeline,
    GOEMO_LABELS,
    POLITE_LABELS,
    BIG5_TRAITS,
)

# ------------------------------------------------------------------
# Cached pipeline loader
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    paths = {
        "emotion": "experiments/checkpoints/emotion/best",
        "tone": "experiments/checkpoints/tone/best",
        "personality_sbert": "sentence-transformers/all-mpnet-base-v2",
        "personality_ckpt": "experiments/checkpoints/personality/best.pt",
    }
    return MoodMosaicPipeline(paths)


pipe = load_pipeline()

# ------------------------------------------------------------------
# Sidebar: description + presets
# ------------------------------------------------------------------
st.sidebar.title("MoodMosaic")
st.sidebar.write(
    "Paste one message per line to see the emotion mosaic, "
    "tone labels, and a simple Big Five personality sketch."
)

example_choice = st.sidebar.selectbox(
    "Load an example",
    [
        "None",
        "Positive day",
        "Frustrated and stressed",
        "Mixed polite and impolite",
    ],
)

if example_choice == "Positive day":
    default_text = """I am really excited about how this semester is going.
I feel proud of the work I put into my projects.
Thank you so much for helping me with the assignment.
I am grateful for my friends checking in on me today.
I am curious to see how far I can push this project."""
elif example_choice == "Frustrated and stressed":
    default_text = """Today was exhausting and nothing worked the way I expected.
I am annoyed that my code keeps breaking right before the deadline.
This whole situation makes me nervous and a little scared.
I feel disappointed in myself for not planning better.
Please just fix this issue, I do not have the energy to debug again."""
elif example_choice == "Mixed polite and impolite":
    default_text = """Could you please review my pull request when you get time.
Thanks a lot for taking the extra effort on this task.
This is completely wrong, you did not follow the instructions.
Why did you ignore my message, this is really upsetting.
I appreciate your patience and all the feedback you gave me."""
else:
    default_text = ""

st.sidebar.markdown("---")
st.sidebar.caption("Models: RoBERTa (emotion), DistilBERT (tone), "
                   "SBERT + MLP (personality).")

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
    st.write("Device:", str(pipe.device))
    st.write("Number of emotion labels:", len(GOEMO_LABELS))
    st.write("Tone classes:", ", ".join(POLITE_LABELS))
    st.write("Personality traits:", ", ".join(BIG5_TRAITS))

# ------------------------------------------------------------------
# Helper functions for plots and table
# ------------------------------------------------------------------
def build_emotion_summary(per_message):
    """Compute mean emotion probabilities and dominant emotion per row."""
    probs = np.array([row["emotion_probs"] for row in per_message])
    mean_probs = probs.mean(axis=0)
    dom_idx = probs.argmax(axis=1)
    dom_labels = [GOEMO_LABELS[i] for i in dom_idx]
    return mean_probs, dom_labels


def build_personality_series(personality_dict):
    """Turn personality dict into a pandas Series in fixed trait order."""
    values = [personality_dict[t] for t in BIG5_TRAITS]
    return pd.Series(values, index=BIG5_TRAITS)


def emotion_radar(mean_probs):
    df = pd.DataFrame(
        {"emotion": GOEMO_LABELS, "score": mean_probs},
    )
    # Show only the top 8 emotions to keep the chart readable
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


def build_per_message_df(per_message, dom_emotions):
    rows = []
    for row, emo_label in zip(per_message, dom_emotions):
        text = row["text"]
        tone_idx = int(row["tone_label"])
        tone_label = POLITE_LABELS[tone_idx] if 0 <= tone_idx < len(POLITE_LABELS) else str(
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
            result = pipe.analyze_messages(lines)

        per_message = result["per_message"]
        # Use our own aggregation for the radar plots
        emo_mean, dom_emotions = build_emotion_summary(per_message)
        personality = build_personality_series(result["summary"]["personality"])
        df_msgs = build_per_message_df(per_message, dom_emotions)

        # Quick text summary
        top_emo_idx = emo_mean.argsort()[::-1][:3]
        top_emos = [GOEMO_LABELS[i] for i in top_emo_idx]
        top_trait_idx = personality.values.argsort()[::-1][:2]
        top_traits = [BIG5_TRAITS[i] for i in top_trait_idx]

        st.markdown("### Summary")
        st.write(
            f"Top emotions across all messages: **{', '.join(top_emos)}**."
        )
        st.write(
            f"Strongest personality traits (approximate): "
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
                st.subheader("Personality profile")
                st.plotly_chart(personality_radar(personality), use_container_width=True)

        with tab_details:
            st.subheader("Per message analysis")

            # Filters
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                tone_filter = st.multiselect(
                    "Filter by tone",
                    options=POLITE_LABELS,
                    default=POLITE_LABELS,
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
