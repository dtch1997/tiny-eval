import pandas as pd
import pathlib
import streamlit as st
import plotly.express as px

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

def main():
    st.title("Steg Task Analysis")

    df = pd.read_csv(results_dir / "results.csv")
    select_df = df[[
        'encoder',
        'plaintext',
        'recovered_message',
        'fidelity_score',
        'stego_score',
        'stegotext',
    ]]
    
    # Histogram of fidelity
    fig = px.histogram(select_df, x='fidelity_score', nbins=100)
    st.plotly_chart(fig)

    # Histogram of stego_score
    fig = px.histogram(select_df, x='stego_score', nbins=100)
    st.plotly_chart(fig)

    # Scatter plot of stego_score vs fidelity_score
    fig = px.scatter(select_df, x='stego_score', y='fidelity_score', color='encoder')
    st.plotly_chart(fig)

    st.dataframe(select_df)


if __name__ == "__main__":
    main()