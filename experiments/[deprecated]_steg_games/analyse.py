import pandas as pd
import pathlib
import streamlit as st
import plotly.express as px

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

def main():
    st.set_page_config(layout="wide")
    st.title("Steg Task Analysis")

    # Load and preprocess data
    df = pd.read_csv(results_dir / "results.csv")
    df['stealth_score'] = 100-df['stego_score']

    # Print error rate 
    error = df["status"] == "error"
    st.write(f"Task error rate: {(error.sum() / len(df)):.2%}")


    select_df = df[[
        'encoder',
        'plaintext',
        'recovered_message',
        'fidelity_score',
        'stealth_score',
        'stegotext',
    ]]
    
    col1, col2, col3 = st.columns(3)

    with col1:
        # Histogram of fidelity
        fig = px.histogram(select_df, x='fidelity_score', nbins=100, color='encoder')
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig)

    with col2:
        # Histogram of stego_score
        fig = px.histogram(select_df, x='stealth_score', nbins=100, color='encoder')
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig)

    with col3:
        # Scatter plot of stego_score vs fidelity_score
        fig = px.scatter(select_df, x='stealth_score', y='fidelity_score', color='encoder')
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig)

    st.dataframe(select_df)


if __name__ == "__main__":
    main()