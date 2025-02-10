import streamlit as st
import pandas as pd
import pathlib
import plotly.express as px
from typing import List, Dict, Any

def load_results() -> pd.DataFrame:
    """
    Load results from the CSV file.
    
    Returns:
        pd.DataFrame: The loaded results dataframe with proper data types
    """
    curr_dir = pathlib.Path(__file__).parent
    results_path = curr_dir / "results" / "results.csv"
    df = pd.read_csv(results_path)
    
    # Convert string representations of lists/dicts back to Python objects
    for col in ['conversation', 'full_interactions']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) and x.strip() else x)
    
    return df

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
                bob_final = next((i for i in reversed(full_interactions.get('bob', []))
                                if i['type'] == 'final_guess'), None)
                if bob_final:
                    format_message("Bob", f"Final Guess: {bob_final['content']}")
                
                # Get Dean's guess
                dean_final = next((i for i in reversed(full_interactions.get('overseer', []))
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
        
        # Map internal names to display names
        player_names = {
            'alice': 'Alice',
            'bob': 'Bob',
            'overseer': 'Dean (Overseer)'
        }
        
        for player, interactions in full_interactions.items():
            display_name = player_names.get(player, player.title())
            with st.expander(f"{display_name} Details"):
                for interaction in interactions:
                    st.markdown(f"**Type**: {interaction['type']}")
                    if interaction['type'] == 'assistant':
                        st.markdown(f"*Raw Response*:")
                        st.code(interaction['raw_response'])
                        st.markdown(f"*Extracted Message*: {interaction['extracted_message']}")
                    elif interaction['type'] in ['system', 'final_guess']:
                        st.markdown(f"*Content*: {interaction['content']}")
                    st.markdown("---")

def format_conversation_page(filtered_df: pd.DataFrame, page_idx: int, full_df: pd.DataFrame):
    """Display a single conversation with navigation."""
    if len(filtered_df) == 0:
        st.warning("No conversations match the selected filters.")
        return
    
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
    
    # Display game details with updated model information
    st.markdown(f"### Game {row.name}")
    
    # Display model information using the direct fields from the updated schema
    st.markdown(f"""
    **Models**:
    - Alice: {row['alice_model']}
    - Bob: {row['bob_model']}
    - Dean (Overseer): {row['dean_model']}
    """)
    
    # Display game outcome information
    outcome_details = [
        f"**Winner**: {row['winner']}",
        f"**Turns**: {row['turns']}",
        f"**Secret Word**: {row['secret_word']}"
    ]
    
    # Add guesses if available
    if 'bob_guess' in row and pd.notna(row['bob_guess']):
        outcome_details.append(f"**Bob's Guess**: {row['bob_guess']}")
    if 'dean_guess' in row and pd.notna(row['dean_guess']):
        outcome_details.append(f"**Dean's Guess**: {row['dean_guess']}")
    
    st.markdown(" | ".join(outcome_details))
    
    # Display error if present
    if pd.notna(row.get('error')):
        st.error(f"Error occurred: {row['error']}")
    
    # Display conversation
    st.markdown("### Conversation")
    show_details = st.checkbox("Show detailed interactions", key=f"details_{page_idx}")
    
    # Pass the full interactions if available
    format_conversation(
        row['conversation'],
        row.get('full_interactions', None),
        show_details=show_details
    )

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
    error_games = df[df['status'] == 'error'].shape[0]
    
    # Count winners excluding error games
    winner_counts = df[df['status'] == 'success']['winner'].value_counts()
    
    # Calculate average turns for successful games
    avg_turns = df[df['status'] == 'success']['turns'].mean()
    
    # Calculate success rate by model combination
    model_success = df.groupby(['alice_model', 'bob_model', 'dean_model']).agg({
        'status': lambda x: (x == 'success').mean()
    }).reset_index()
    
    # Calculate guess outcomes for successful games
    successful_games_df = df[df['status'] == 'success'].copy()
    
    # Create guess outcome categories
    successful_games_df['guess_outcome'] = successful_games_df.apply(
        lambda row: (
            "Both Correct" if row['bob_guess'] == row['secret_word'] and row['dean_guess'] == row['secret_word']
            else "Both Wrong" if row['bob_guess'] != row['secret_word'] and row['dean_guess'] != row['secret_word']
            else "Only Bob Correct" if row['bob_guess'] == row['secret_word']
            else "Only Dean Correct"
        ),
        axis=1
    )
    
    guess_outcomes = successful_games_df['guess_outcome'].value_counts().to_dict()

    return {
        'total_games': total_games,
        'successful_games': successful_games,
        'error_games': error_games,
        'winner_counts': winner_counts,
        'avg_turns': avg_turns,
        'model_success': model_success,
        'guess_outcomes': guess_outcomes
    }

def main():
    st.title("Contact Game Analysis")
    
    # Initialize session state for pagination
    if 'page_idx' not in st.session_state:
        st.session_state.page_idx = 0
    
    # Load data
    df = load_results()
    
    # Display overall metrics
    metrics = analyze_results(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Games", metrics['total_games'])
    with col2:
        st.metric("Successful Games", metrics['successful_games'])
    with col3:
        st.metric("Error Games", metrics['error_games'])
    with col4:
        st.metric("Average Turns", f"{metrics['avg_turns']:.2f}")
    
    # Winner distribution
    st.subheader("Winner Distribution")
    fig = px.pie(
        values=metrics['winner_counts'].values,
        names=metrics['winner_counts'].index,
        title="Game Outcomes"
    )
    st.plotly_chart(fig)
    
    # Add model success rate visualization
    st.subheader("Success Rate by Model Configuration")
    success_fig = px.bar(
        metrics['model_success'],
        x='alice_model',
        y='status',
        color='dean_model',
        barmode='group',
        title="Success Rate by Model Configuration",
        labels={'status': 'Success Rate', 'alice_model': 'Player Model', 'dean_model': 'Overseer Model'}
    )
    st.plotly_chart(success_fig)
    
    # Add guess outcomes visualization
    st.subheader("Guess Outcomes")
    guess_outcomes = pd.DataFrame(
        list(metrics['guess_outcomes'].items()),
        columns=['Outcome', 'Count']
    )
    
    guess_fig = px.bar(
        guess_outcomes,
        x='Outcome',
        y='Count',
        title="Distribution of Guess Outcomes",
        labels={'Count': 'Number of Games', 'Outcome': 'Guess Outcome'},
        color='Outcome',
        color_discrete_map={
            'Both Correct': '#2ecc71',
            'Both Wrong': '#e74c3c',
            'Only Bob Correct': '#3498db',
            'Only Dean Correct': '#f1c40f'
        }
    )
    guess_fig.update_layout(
        xaxis_title="Outcome",
        yaxis_title="Number of Games",
        showlegend=False
    )
    st.plotly_chart(guess_fig)
    
    # Conversation viewer
    st.subheader("Conversation Viewer")
    
    # Filter only by winner
    selected_winner = st.selectbox(
        "Filter by Winner",
        options=['All'] + list(df['winner'].unique())
    )
    
    # Filter conversations
    filtered_df = df if selected_winner == 'All' else df[df['winner'] == selected_winner]
    
    # Display paginated conversation
    format_conversation_page(filtered_df, st.session_state.page_idx, df)
    
    # Add keyboard navigation
    st.markdown("""
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft' || e.key === 'h' || e.key === 'H') {
                e.preventDefault();
                const prevButton = document.querySelector('button:has(div:contains("← Previous"))');
                if (prevButton && !prevButton.hasAttribute('disabled')) {
                    prevButton.click();
                }
            } else if (e.key === 'ArrowRight' || e.key === 'l' || e.key === 'L') {
                e.preventDefault();
                const nextButton = document.querySelector('button:has(div:contains("Next →"))');
                if (nextButton && !nextButton.hasAttribute('disabled')) {
                    nextButton.click();
                }
            }
        });
        </script>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
