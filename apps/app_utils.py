"""
Utility functions for Streamlit apps
"""
import streamlit as st
import pandas as pd


def create_sidebar_menu():
    """Create standard sidebar menu"""
    st.sidebar.header("📊 Menu")
    return st.sidebar.radio("Pilih Menu:", ["🏠 Home", "🔍 Prediksi", "📈 Model Info"])


def display_metrics(metrics):
    """
    Display metrics in columns
    Args:
        metrics: list of tuples (label, value)
    """
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)


def display_footer(dataset_name, sample_count):
    """Display standard footer"""
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center'>
        <p>Dibuat dengan ❤️ menggunakan Streamlit | Dataset: {dataset_name} ({sample_count} samples)</p>
    </div>
    """, unsafe_allow_html=True)


def create_prediction_button(text="🔮 Prediksi"):
    """Create standard prediction button"""
    return st.button(text, use_container_width=True, type="primary")


def section_divider():
    """Add section divider"""
    st.markdown("---")


def display_probabilities(classes, probabilities, emoji_map=None):
    """
    Display probability bars for all classes
    Args:
        classes: list of class names
        probabilities: list of probabilities
        emoji_map: dict mapping class name to emoji (optional)
    """
    for cls, prob in zip(classes, probabilities):
        emoji = emoji_map.get(cls, '⚪') if emoji_map else ''
        st.write(f"{emoji} **{cls.title()}**: {prob:.1%}")


def display_prediction_result(prediction, confidence, emoji='', description=''):
    """Display prediction result with progress bar"""
    st.markdown(f"### {emoji} {description}")
    st.progress(confidence)
    st.write(f"**Confidence:** {confidence:.1%}")


def load_and_preprocess_model(model_path, dataset_path, train_func):
    """
    Generic model loading with auto-training
    Args:
        model_path: path to saved model
        dataset_path: path to dataset
        train_func: function to train model if not exists
    """
    import pickle
    import os

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        model_data = train_func(dataset_path)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        return model_data
