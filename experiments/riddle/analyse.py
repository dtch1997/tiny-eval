import streamlit as st
import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt

# Setup paths
curr_dir = pathlib.Path(__file__).parent
results_path = curr_dir / "results" / "results.csv"

def load_data() -> pd.DataFrame:
    """Load and preprocess the results dataframe"""
    if not results_path.exists():
        st.error("No results file found. Please run main.py first.")
        st.stop()
        
    df = pd.read_csv(results_path)
    # Convert boolean columns if present
    if 'is_correct' in df.columns:
        df['is_correct'] = df['is_correct'].astype(bool)
    return df

def create_performance_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Create a heatmap showing success rates between riddler and solver models
    
    Args:
        df: DataFrame containing the results
        
    Returns:
        matplotlib.figure.Figure: The heatmap figure
    """
    # Filter for successful attempts only
    success_df = df[df['status'] == 'success']
    
    # Create pivot table of success rates
    pivot = pd.pivot_table(
        success_df,
        values='is_correct',
        index='riddler_model',
        columns='solver_model',
        aggfunc='mean'
    )
    
    # Create figure and axis with smaller size
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create heatmap
    sns.heatmap(
        pivot,
        annot=True,  # Show values in cells
        fmt='.1%',   # Format as percentages
        cmap='RdYlGn',  # Red to Yellow to Green colormap
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Success Rate'},
        ax=ax,
        square=True  # Make cells square for better appearance
    )
    
    # Customize appearance
    plt.title('Model Performance Matrix', pad=10)
    plt.xlabel('Solver Model', labelpad=10)
    plt.ylabel('Riddler Model', labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def main():
    st.title("🧩 Riddle Me This!")
    
    # Add description of riddle generation process
    st.markdown("""
    ### 🎮 How It Works
    
    Each riddle is created through an AI-powered two-step process:
    1. A "Riddler" AI model creates a 4-line rhyming riddle for a target word
    2. A "Solver" AI model attempts to solve the riddle without knowing the target word
    
    The Riddler is instructed to create challenging but solvable riddles that follow strict rules:
    - Exactly 4 lines with rhyming structure
    - No obvious synonyms or direct references
    - Must have exactly one correct answer
    
    This creates an interesting dynamic where we can see how well one AI model can solve
    riddles created by another!
    """)
    
    # Load data
    df = load_data()
    
    # Display overall statistics in a collapsed section
    with st.expander("📊 Show Statistics", expanded=True):
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            success_rate = (df['status'] == 'success').mean()
            st.metric("Success Rate", f"{success_rate:.1%}")
        with col2:
            accuracy = df[df['status'] == 'success']['is_correct'].mean()
            st.metric("Model Accuracy", f"{accuracy:.1%}")
        with col3:
            total_riddles = len(df)
            st.metric("Total Riddles", total_riddles)
            
        # Add spacing
        st.markdown("### 🎯 Model Performance Matrix")
        st.markdown("This heatmap shows how well each solver model performs against riddles created by different riddler models:")
        
        # Create and display heatmap
        fig = create_performance_heatmap(df)
        st.pyplot(fig)
        
        # Add explanation of the heatmap
        st.markdown("""
        **How to read the heatmap:**
        - Each cell shows the success rate of a solver model (columns) solving riddles created by a riddler model (rows)
        - Darker green indicates higher success rates
        - Darker red indicates lower success rates
        """)
    
    # Initialize session state for tracking revealed answers
    if 'revealed_answers' not in st.session_state:
        st.session_state.revealed_answers = set()
    
    # Filters
    st.markdown("### 🔍 Filter Riddles")
    col1, col2 = st.columns(2)
    with col1:
        show_correct = st.checkbox("Show only solved correctly", False)
    with col2:
        show_incorrect = st.checkbox("Show only solved incorrectly", False)
    
    # Filter based on checkboxes
    filtered_df = df[df['status'] == 'success']  # Start with successful attempts
    if show_correct and not show_incorrect:
        filtered_df = filtered_df[filtered_df['is_correct']]
    elif show_incorrect and not show_correct:
        filtered_df = filtered_df[~filtered_df['is_correct']]
    
    # Display individual riddles
    st.markdown("### 🎯 Can You Solve These Riddles?")
    for idx, row in filtered_df.iterrows():
        st.markdown("---")
        st.markdown(f"### Riddle #{idx + 1}")
        
        # Display the riddle in a prominent way
        st.markdown(f"```\n{row['riddle']}\n```")
        
        # Create columns for reveal/hide buttons
        col1, col2 = st.columns([1, 4])
        
        # Toggle button for showing/hiding answer
        if idx not in st.session_state.revealed_answers:
            if col1.button(f"🔍 Reveal Answer #{idx + 1}"):
                st.session_state.revealed_answers.add(idx)
                st.rerun()
        else:
            if col1.button(f"🙈 Hide Answer #{idx + 1}"):
                st.session_state.revealed_answers.remove(idx)
                st.rerun()
        
        # Show answer and details if revealed
        if idx in st.session_state.revealed_answers:
            st.markdown("#### Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Correct Answer**")
                st.info(row['target_word'])
                
            with col2:
                st.markdown("**🤖 Model's Guess**")
                if row['is_correct']:
                    st.success(row['answer'])
                else:
                    st.error(row['answer'])
            
            # Show model's full reasoning in a collapsible section
            with st.expander("🤔 See Model's Reasoning", expanded=False):
                st.text(row['solver_response'])
                
            # Show model information
            st.markdown("##### Models Used")
            st.text(f"Riddler: {row['riddler_model']}\nSolver: {row['solver_model']}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Riddle Game",
        page_icon="🧩",
        layout="wide"
    )
    main() 