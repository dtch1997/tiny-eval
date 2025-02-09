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

def format_message(speaker: str, message: str) -> None:
    """Display a single message in chat bubble style."""
    # Create columns for message alignment
    left, right = st.columns([6, 4]) if speaker.lower() == "alice" else st.columns([4, 6])
    
    # Style for message bubbles with text color
    bubble_style = {
        "alice": """
            padding: 10px;
            border-radius: 15px;
            background-color: #DCF8C6;
            color: #000000;
            margin-bottom: 10px;
            display: inline-block;
            max-width: 100%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        """,
        "bob": """
            padding: 10px;
            border-radius: 15px;
            background-color: #E8E8E8;
            color: #000000;
            margin-bottom: 10px;
            display: inline-block;
            max-width: 100%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        """,
        "system": """
            padding: 10px;
            border-radius: 15px;
            background-color: #FFE4E1;
            color: #000000;
            margin: 10px auto;
            text-align: center;
            max-width: 80%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        """
    }
    
    if speaker.lower() == "alice":
        with left:
            st.markdown(f"""
                <div style="{bubble_style['alice']}">
                    <strong style="color: #000000;">{speaker}</strong><br>
                    {message}
                </div>
            """, unsafe_allow_html=True)
    elif speaker.lower() == "dean":
        st.markdown(f"""
            <div style="{bubble_style['system']}">
                <strong style="color: #000000;">Overseer</strong><br>
                {message}
            </div>
        """, unsafe_allow_html=True)
    else:  # Bob
        with right:
            st.markdown(f"""
                <div style="{bubble_style['bob']}">
                    <strong style="color: #000000;">{speaker}</strong><br>
                    {message}
                </div>
            """, unsafe_allow_html=True)

def format_conversation(
    conversation: List[str], 
    full_interactions: Dict[str, List[Dict[str, Any]]] | str | None = None,
    show_details: bool = False
) -> None:
    """
    Format the conversation for display with chat-like styling.
    
    Args:
        conversation: List of conversation strings
        full_interactions: Optional dict or string containing detailed interaction logs
        show_details: Whether to show detailed interaction logs
    """
    if isinstance(conversation, str):
        conversation = eval(conversation)
    
    # Process conversation messages
    for line in conversation:
        speaker = line.split(":")[0].strip()
        message = ":".join(line.split(":")[1:]).strip()
        
        # Add final guesses to conversation if this is a CONTACT message
        if "contact" in message.lower():
            format_message(speaker, message)
            
            # Add final guesses if available
            if full_interactions:
                if isinstance(full_interactions, str):
                    try:
                        full_interactions = eval(full_interactions)
                    except:
                        continue
                
                # Get Bob's final guess
                bob_final = next((i for i in reversed(full_interactions['player_2']) 
                                if i['type'] == 'final_guess'), None)
                if bob_final:
                    format_message("Bob", f"Final Guess: {bob_final['content']}")
                
                # Get Dean's guess
                dean_final = next((i for i in reversed(full_interactions['overseer']) 
                                 if i['type'] == 'final_guess'), None)
                if dean_final:
                    format_message("Dean", f"Final Guess: {dean_final['content']}")
        else:
            format_message(speaker, message)
    
    # Add detailed interactions if requested
    if full_interactions and show_details:
        st.markdown("### Detailed Interactions")
        if isinstance(full_interactions, str):
            try:
                full_interactions = eval(full_interactions)
            except:
                return
        
        for player, interactions in full_interactions.items():
            with st.expander(f"{player.title()} Details"):
                for interaction in interactions:
                    st.markdown(f"**Type**: {interaction['type']}")
                    if interaction['type'] == 'assistant':
                        st.markdown(f"*Raw Response*:")
                        st.code(interaction['raw_response'])
                        st.markdown(f"*Extracted Message*: {interaction['extracted_message']}")
                    elif interaction['type'] in ['system', 'final_guess']:
                        st.markdown(f"*Content*: {interaction['content']}")
                    st.markdown("---")

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

def display_conversation_page(filtered_df: pd.DataFrame, page_idx: int):
    """Display a single conversation with navigation."""
    if len(filtered_df) == 0:
        st.warning("No conversations match the selected filters.")
        return
    
    # Ensure page_idx is within bounds
    page_idx = max(0, min(page_idx, len(filtered_df) - 1))
    
    # Get current row
    row = filtered_df.iloc[page_idx]
    
    # Navigation controls
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Previous", key="prev", disabled=page_idx == 0):
            st.session_state.page_idx = max(0, page_idx - 1)
            st.rerun()
    
    with col2:
        st.write(f"Conversation {page_idx + 1} of {len(filtered_df)}")
    
    with col3:
        if st.button("Next →", key="next", disabled=page_idx == len(filtered_df) - 1):
            st.session_state.page_idx = min(len(filtered_df) - 1, page_idx + 1)
            st.rerun()
    
    # Display conversation details
    st.markdown(f"### Game {row.name}")
    
    # Display model information using the direct fields
    st.markdown(f"""
    **Models**:
    - Alice: {row['alice_model']}
    - Bob: {row['bob_model']}
    - Dean (Overseer): {row['dean_model']}
    """)
    
    st.markdown(f"**Winner**: {row['winner']} | **Turns**: {row['turns']} | **Secret Word**: {row['secret_word']}")
    
    # Display conversation
    st.markdown("### Conversation")
    show_details = st.checkbox("Show detailed interactions", key=f"details_{page_idx}")
    format_conversation(
        row['conversation'], 
        row.get('full_interactions', None),
        show_details=show_details
    )

def main():
    st.title("Contact Game Analysis")
    
    # Initialize session state for pagination
    if 'page_idx' not in st.session_state:
        st.session_state.page_idx = 0
    
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
    
    # Display paginated conversation
    display_conversation_page(filtered_df, st.session_state.page_idx)
    
    # Add keyboard navigation
    st.markdown("""
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft') {
                document.querySelector('button[kind="secondary"]:first-of-type').click();
            } else if (e.key === 'ArrowRight') {
                document.querySelector('button[kind="secondary"]:last-of-type').click();
            }
        });
        </script>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
