import streamlit as st
import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Setup paths
curr_dir = pathlib.Path(__file__).parent
results_path = curr_dir / "results" / "results.csv"

def load_data() -> pd.DataFrame:
    """Load and preprocess the results dataframe"""
    if not results_path.exists():
        st.error("No results file found. Please run main.py first.")
        st.stop()
        
    df = pd.read_csv(results_path)
    
    # Convert TaskResult data structure
    if 'data' in df.columns:
        # Expand the data column
        data_df = pd.json_normalize(df['data'].apply(eval))
        # Drop the original data column and combine with status and error
        df = pd.concat([df[['status', 'error']], data_df], axis=1)
    
    # Convert boolean columns if present
    if 'is_correct' in df.columns:
        df['is_correct'] = df['is_correct'].astype(bool)
    return df

def create_performance_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Create a bar plot showing success rates between storyteller and guesser models with error bars
    
    Args:
        df: DataFrame containing the results
        
    Returns:
        matplotlib.figure.Figure: The bar plot figure
    """
    # Filter for successful attempts only
    success_df = df[df['status'] == 'success']
    
    # Calculate statistics for each model combination
    stats = []
    for storyteller in success_df['storyteller_model'].unique():
        for guesser in success_df['guesser_model'].unique():
            data = success_df[
                (success_df['storyteller_model'] == storyteller) & 
                (success_df['guesser_model'] == guesser)
            ]['is_correct']
            
            if len(data) > 0:
                mean = data.mean()
                stderr = data.std() / np.sqrt(len(data))
                stats.append({
                    'storyteller_model': storyteller,
                    'guesser_model': guesser,
                    'mean': mean,
                    'stderr': stderr,
                    'count': len(data)
                })
    
    stats_df = pd.DataFrame(stats)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up bar positions
    guessers = stats_df['guesser_model'].unique()
    storytellers = stats_df['storyteller_model'].unique()
    x = np.arange(len(guessers))
    width = 0.8 / len(storytellers)  # Adjust bar width based on number of storytellers
    
    # Plot bars for each storyteller
    for i, storyteller in enumerate(storytellers):
        storyteller_data = stats_df[stats_df['storyteller_model'] == storyteller]
        
        # Plot bars with error bars
        bars = ax.bar(
            x + i * width - (len(storytellers)-1) * width/2, 
            storyteller_data['mean'],
            width,
            label=f'Storyteller: {storyteller}',
            yerr=storyteller_data['stderr'],
            capsize=5
        )
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{height:.1%}',
                ha='center',
                va='bottom',
                size=8
            )
    
    # Customize appearance
    ax.set_ylabel('Success Rate')
    ax.set_xlabel('Guesser Model')
    ax.set_title('Model Performance by Storyteller-Guesser Pairs\n(mean ± standard error)')
    ax.set_xticks(x)
    ax.set_xticklabels(guessers, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)  # Set y-axis from 0 to 1
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    
    return fig

def format_story(story: str) -> None:
    """Display a story with proper text wrapping"""
    st.markdown(
        f"""
        <div style="
            white-space: pre-wrap;
            background-color: var(--st-color-background-secondary);
            border: 1px solid var(--st-color-border-light);
            padding: 1rem;
            border-radius: 0.5rem;
            font-family: 'Source Code Pro', monospace;
            line-height: 1.5;
            color: var(--st-color-text);
        ">
        {story}
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.title("📚 Story Concept Game!")
    
    # Add description
    st.markdown("""
    ### 🎮 How It Works
    
    Each story is created through an AI-powered two-step process:
    1. A "Storyteller" AI model creates a short story themed around a target concept
    2. A "Guesser" AI model attempts to identify the concept without knowing it beforehand
    
    The Storyteller is instructed to create engaging stories that:
    - Are 3-4 paragraphs long
    - Never directly mention the concept or obvious synonyms
    - Use metaphors and situations that relate to the concept
    - Make the concept discoverable through careful reading
    
    This creates an interesting challenge where we can see how well one AI model can understand
    the thematic elements created by another!
    """)
    
    # Load data
    df = load_data()
    
    # Display overall statistics
    with st.expander("📊 Show Statistics", expanded=True):
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            success_rate = (df['status'] == 'success').mean()
            st.metric("Success Rate", f"{success_rate:.1%}")
        with col2:
            accuracy = df[df['status'] == 'success']['is_correct'].mean()
            st.metric("Concept Guessing Accuracy", f"{accuracy:.1%}")
        with col3:
            total_stories = len(df)
            st.metric("Total Stories", total_stories)
            
        st.markdown("### 🎯 Model Performance Matrix")
        st.markdown("This chart shows how well each guesser model performs at identifying concepts in stories created by different storyteller models:")
        
        # Create and display performance plot
        fig = create_performance_plot(df)
        st.pyplot(fig)
        
        st.markdown("""
        **How to read the chart:**
        - Each group shows the success rate for a storyteller model
        - Different colored bars represent different guesser models
        - Higher bars indicate better performance
        - Success rate is the percentage of correctly guessed concepts
        """)
    
    # Initialize session state for revealed stories
    if 'revealed_answers' not in st.session_state:
        st.session_state.revealed_answers = set()
    
    # Filters
    st.markdown("### 🔍 Filter Stories")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        st.markdown("**Correctness**")
        show_correct = st.checkbox("Show correct", True)
        show_incorrect = st.checkbox("Show incorrect", True)
    
    with filter_col2:
        st.markdown("**Storyteller Model**")
        storyteller_models = ['All'] + sorted(df['storyteller_model'].unique().tolist())
        selected_storyteller = st.selectbox(
            "Select Storyteller Model",
            options=storyteller_models,
            key="storyteller_filter"
        )
    
    with filter_col3:
        st.markdown("**Guesser Model**")
        guesser_models = ['All'] + sorted(df['guesser_model'].unique().tolist())
        selected_guesser = st.selectbox(
            "Select Guesser Model",
            options=guesser_models,
            key="guesser_filter"
        )
    
    # Filter based on selections
    filtered_df = df[df['status'] == 'success']
    
    if show_correct and not show_incorrect:
        filtered_df = filtered_df[filtered_df['is_correct']]
    elif show_incorrect and not show_correct:
        filtered_df = filtered_df[~filtered_df['is_correct']]
    elif not show_correct and not show_incorrect:
        filtered_df = filtered_df[filtered_df['is_correct'] != filtered_df['is_correct']]
    
    if selected_storyteller != 'All':
        filtered_df = filtered_df[filtered_df['storyteller_model'] == selected_storyteller]
    
    if selected_guesser != 'All':
        filtered_df = filtered_df[filtered_df['guesser_model'] == selected_guesser]
    
    # Display individual stories
    st.markdown("### 📖 Can You Guess The Concepts?")
    for idx, row in filtered_df.iterrows():
        st.markdown("---")
        st.markdown(f"### Story #{idx + 1}")
        
        # Display the story
        format_story(row['story'])
        
        col1, col2 = st.columns([1, 4])
        
        if idx not in st.session_state.revealed_answers:
            if col1.button(f"🔍 Reveal Concept #{idx + 1}"):
                st.session_state.revealed_answers.add(idx)
                st.rerun()
        else:
            if col1.button(f"🙈 Hide Concept #{idx + 1}"):
                st.session_state.revealed_answers.remove(idx)
                st.rerun()
        
        if idx in st.session_state.revealed_answers:
            st.markdown("#### Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Target Concept**")
                st.info(row['target_concept'])
                
            with col2:
                st.markdown("**🤖 Model's Guess**")
                if row['is_correct']:
                    st.success(row['answer'])
                else:
                    st.error(row['answer'])
            
            # Add storyteller's reasoning
            with st.expander("✍️ See Storyteller's Approach", expanded=False):
                if row.get('storyteller_reasoning'):
                    st.markdown("**Creative Approach and Metaphors:**")
                    st.text(row['storyteller_reasoning'])
                else:
                    st.info("No storyteller reasoning available for this story.")
            
            # Existing guesser analysis
            with st.expander("🤔 See Guesser's Analysis", expanded=False):
                st.text(row['guesser_response'])
                
            st.markdown("##### Models Used")
            st.text(f"Storyteller: {row['storyteller_model']}\nGuesser: {row['guesser_model']}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Story Concept Game",
        page_icon="📚",
        layout="wide"
    )
    main() 