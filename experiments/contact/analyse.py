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

def format_conversation(conversation: List[str], full_interactions: Dict[str, List[Dict[str, Any]]] | str | None = None) -> str:
    """
    Format the conversation for display with proper spacing and styling.
    
    Args:
        conversation: List of conversation strings
        full_interactions: Optional dict or string containing detailed interaction logs
        
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
        
    # Add detailed interactions if available
    if full_interactions:
        # Parse full_interactions if it's a string
        if isinstance(full_interactions, str):
            try:
                full_interactions = eval(full_interactions)
            except:
                # If parsing fails, skip detailed interactions
                return formatted
        
        formatted += "\n### Detailed Interactions\n\n"
        for player, interactions in full_interactions.items():
            formatted += f"\n#### {player.title()}\n\n"
            for interaction in interactions:
                formatted += f"**Type**: {interaction['type']}\n\n"
                if interaction['type'] == 'assistant':
                    formatted += f"*Raw Response*:\n```\n{interaction['raw_response']}\n```\n\n"
                    formatted += f"*Extracted Message*: {interaction['extracted_message']}\n\n"
                elif interaction['type'] in ['system', 'final_guess']:
                    formatted += f"*Content*: {interaction['content']}\n\n"
                formatted += "---\n\n"
    
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
    st.markdown(f"**Winner**: {row['winner']}")
    st.markdown(f"**Turns**: {row['turns']}")
    st.markdown(f"**Secret Word**: {row['secret_word']}")
    
    # Display conversation
    st.markdown("### Conversation")
    
    # Add toggle for detailed view
    show_details = st.checkbox("Show detailed interactions", key=f"details_{page_idx}")
    
    if show_details:
        st.markdown(format_conversation(
            row['conversation'], 
            row.get('full_interactions', None)
        ))
    else:
        # Show just the basic conversation without detailed interactions
        if isinstance(row['conversation'], str):
            conversation = eval(row['conversation'])
        else:
            conversation = row['conversation']
        
        for line in conversation:
            speaker = line.split(":")[0].strip()
            message = ":".join(line.split(":")[1:]).strip()
            st.markdown(f"**{speaker}**: {message}")
    
    if row['contact_declared']:
        st.markdown("### Final Guesses")
        
        # Create columns for each player's guess
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Alice")
            st.markdown(f"**Word**: {row['secret_word']}")
            # Get Alice's final message
            if isinstance(row['conversation'], str):
                conversation = eval(row['conversation'])
            else:
                conversation = row['conversation']
            alice_final = next((msg for msg in reversed(conversation) 
                              if msg.startswith("Alice:")), "No final message")
            st.markdown(f"**Final Message**: {alice_final.split(':', 1)[1].strip()}")
            
        with col2:
            st.markdown("#### Bob")
            st.markdown(f"**Guess**: {row.get('bob_guess', 'N/A')}")
            # Get Bob's reasoning from full interactions
            if row.get('full_interactions'):
                interactions = row['full_interactions']
                if isinstance(interactions, str):
                    interactions = eval(interactions)
                bob_final = next((i for i in reversed(interactions['player_2']) 
                                if i['type'] == 'final_guess'), None)
                if bob_final:
                    st.markdown(f"**Reasoning**: {bob_final['content']}")
            
        with col3:
            st.markdown("#### Dean")
            st.markdown(f"**Guess**: {row.get('dean_guess', 'N/A')}")
            # Get Dean's reasoning from full interactions
            if row.get('full_interactions'):
                interactions = row['full_interactions']
                if isinstance(interactions, str):
                    interactions = eval(interactions)
                dean_final = next((i for i in reversed(interactions['overseer']) 
                                 if i['type'] == 'final_guess'), None)
                if dean_final:
                    st.markdown(f"**Reasoning**: {dean_final['content']}")

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
