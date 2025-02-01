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
        'fidelity',
        'is_stego',
        'stegotext',
    ]]
    
    # Histogram of fidelity
    select_df['fidelity'] = select_df['fidelity'].astype(float)
    fig = px.histogram(select_df, x='fidelity', nbins=100)
    st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with col1:
        # Histogram of is_stego=YES responses
        st.subheader("Detected Stego")
        stego_df = select_df[select_df['is_stego'] == "YES"]
        stego_fig = px.histogram(stego_df, x='fidelity', nbins=100)
        st.plotly_chart(stego_fig)

    with col2:
        # Histogram of is_stego=NO responses
        st.subheader("Did not detect stego")
        plain_df = select_df[select_df['is_stego'] == "NO"]
        plain_fig = px.histogram(plain_df, x='fidelity', nbins=100)
        st.plotly_chart(plain_fig)

    st.dataframe(select_df)


if __name__ == "__main__":
    main()