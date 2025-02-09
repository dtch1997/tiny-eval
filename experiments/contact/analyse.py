import streamlit as st
import pandas as pd
import pathlib
import plotly.express as px
from typing import List, Dict, Any

def load_results() -> pd.DataFrame:
    """
    Load results from the CSV file.
    
    Returns:
        pd.DataFrame: The loaded results dataframe
    """
    curr_dir = pathlib.Path(__file__).parent
    results_path = curr_dir / "results" / "results.csv"
    return pd.DataFrame(pd.read_csv(results_path))

def format_conversation(conversation: List[str]) -> str:
    """
    Format the conversation for display with proper spacing and styling.
    
    Args:
        conversation: List of conversation strings
        
    Returns:
        str: Formatted conversation text
    """
    if isinstance(conversation, str):
        # Handle case where conversation is stored as string
        conversation = eval(conversation)
    
    formatted = ""
    for line in conversation:
        speaker = line.split(":")[0].strip()
        message = ":".join(line.split(":")[1:]).strip()
        formatted += f"**{speaker}**: {message}\n\n"
    return formatted

def analyze_results(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate key metrics from results.
    
    Args:
        df: Results dataframe
        
    Returns:
        dict: Dictionary containing analysis metrics
    """
    total_games = len(df)
    successful_games = df[df['status'] == 'success'].shape[0]
    
    winner_counts = df['winner'].value_counts()
    avg_turns = df['turns'].mean()
    
    return {
        'total_games': total_games,
        'successful_games': successful_games,
        'winner_counts': winner_counts,
        'avg_turns': avg_turns
    }

def main():
    st.title("Contact Game Analysis")
    
    # Load data
    df = load_results()
    
    # Display overall metrics
    metrics = analyze_results(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Games", metrics['total_games'])
    with col2:
        st.metric("Successful Games", metrics['successful_games'])
    with col3:
        st.metric("Average Turns", f"{metrics['avg_turns']:.2f}")
    
    # Winner distribution
    st.subheader("Winner Distribution")
    fig = px.pie(
        values=metrics['winner_counts'].values,
        names=metrics['winner_counts'].index,
        title="Game Outcomes"
    )
    st.plotly_chart(fig)
    
    # Conversation viewer
    st.subheader("Conversation Viewer")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        selected_word = st.selectbox(
            "Select Secret Word",
            options=sorted(df['secret_word'].unique())
        )
    
    with col2:
        selected_winner = st.selectbox(
            "Filter by Winner",
            options=['All'] + list(df['winner'].unique())
        )
    
    # Filter conversations
    filtered_df = df[df['secret_word'] == selected_word]
    if selected_winner != 'All':
        filtered_df = filtered_df[filtered_df['winner'] == selected_winner]
    
    # Display conversations
    for idx, row in filtered_df.iterrows():
        with st.expander(f"Game {idx} (Winner: {row['winner']}, Turns: {row['turns']})"):
            st.markdown(format_conversation(row['conversation']))
            if row['contact_declared']:
                st.markdown("**Final Guesses:**")
                st.markdown(f"- Bob's guess: {row.get('bob_guess', 'N/A')}")
                st.markdown(f"- Dean's guess: {row.get('dean_guess', 'N/A')}")

if __name__ == "__main__":
    main()
