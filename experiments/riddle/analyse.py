import streamlit as st
import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import textwrap

# Setup paths
curr_dir = pathlib.Path(__file__).parent
results_path = curr_dir / "results" / "results.csv"

def load_data() -> pd.DataFrame:
    """Load and preprocess the results dataframe"""
    if not results_path.exists():
        st.error("No results file found. Please run main.py first.")
        st.stop()
        
    df = pd.read_csv(results_path)
    
    # Normalize the data structure from TaskResult format
    if 'data' in df.columns:
        # Extract fields from the data column if it exists
        data_df = pd.json_normalize(df['data'])
        df = pd.concat([df.drop(['data'], axis=1), data_df], axis=1)
    
    # Convert boolean columns if present
    if 'is_correct' in df.columns:
        df['is_correct'] = df['is_correct'].astype(bool)
    return df

def create_performance_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Create a bar plot showing success rates between riddler and solver models with error bars
    
    Args:
        df: DataFrame containing the results
        
    Returns:
        matplotlib.figure.Figure: The bar plot figure
    """
    # Filter for successful attempts only
    success_df = df[df['status'] == 'success']
    
    # Calculate statistics for each model combination
    stats = []
    for riddler in success_df['riddler_model'].unique():
        for solver in success_df['solver_model'].unique():
            data = success_df[
                (success_df['riddler_model'] == riddler) & 
                (success_df['solver_model'] == solver)
            ]['is_correct']
            
            if len(data) > 0:
                mean = data.mean()
                stderr = data.std() / np.sqrt(len(data))
                stats.append({
                    'riddler_model': riddler,
                    'solver_model': solver,
                    'mean': mean,
                    'stderr': stderr,
                    'count': len(data)
                })
    
    stats_df = pd.DataFrame(stats)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up bar positions
    solvers = stats_df['solver_model'].unique()
    riddlers = stats_df['riddler_model'].unique()
    x = np.arange(len(solvers))
    width = 0.8 / len(riddlers)  # Adjust bar width based on number of riddlers
    
    # Plot bars for each riddler
    for i, riddler in enumerate(riddlers):
        riddler_data = stats_df[stats_df['riddler_model'] == riddler]
        
        # Plot bars with error bars
        bars = ax.bar(
            x + i * width - (len(riddlers)-1) * width/2, 
            riddler_data['mean'],
            width,
            label=f'Riddler: {riddler}',
            yerr=riddler_data['stderr'],
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
    ax.set_xlabel('Solver Model')
    ax.set_title('Model Performance by Riddler-Solver Pairs\n(mean ± standard error)')
    ax.set_xticks(x)
    ax.set_xticklabels(solvers, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)  # Set y-axis from 0 to 1
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    
    return fig

def display_riddles(df: pd.DataFrame, page: int = 0, per_page: int = 10):
    total_riddles = len(df)
    total_pages = (total_riddles + per_page - 1) // per_page  # Round up division
    
    if page >= total_pages:
        print(f"Invalid page number. Total pages: {total_pages}")
        return
    
    start_idx = page * per_page
    end_idx = min(start_idx + per_page, total_riddles)
    
    print(f"\nShowing riddles {start_idx+1}-{end_idx} of {total_riddles} (Page {page+1} of {total_pages})")
    print("-" * 80)
    
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        print(f"\nRiddle {idx+1}: (Target: {row['target_word']})")
        print(textwrap.fill(row['riddle'], width=80))
        print(f"Solver response: {row['solver_response'][:100]}...")
        print(f"Answer given: {row['answer']}")
        print(f"Correct: {row['is_correct']}")
        print("-" * 80)

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
        st.markdown("### 🎯 Model Performance")
        st.markdown("This plot shows how well each solver model performs against riddles created by different riddler models:")
        
        # Create and display performance plot
        fig = create_performance_plot(df)
        st.pyplot(fig)
        
        # Add explanation of the plot
        st.markdown("""
        **How to read the plot:**
        - Each group shows a solver model's performance
        - Different colored bars represent different riddler models
        - Error bars show the standard error of the mean
        - Higher values indicate better performance
        """)
    
    # Initialize session state for tracking revealed answers
    if 'revealed_answers' not in st.session_state:
        st.session_state.revealed_answers = set()
    
    # Filters
    st.markdown("### 🔍 Filter Riddles")
    
    # Create three columns for filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        st.markdown("**Correctness**")
        show_correct = st.checkbox("Show correct", True)
        show_incorrect = st.checkbox("Show incorrect", True)
    
    with filter_col2:
        st.markdown("**Riddler Model**")
        # Get unique riddler models
        riddler_models = ['All'] + sorted(df['riddler_model'].unique().tolist())
        selected_riddler = st.selectbox(
            "Select Riddler Model",
            options=riddler_models,
            key="riddler_filter"
        )
    
    with filter_col3:
        st.markdown("**Solver Model**")
        # Get unique solver models
        solver_models = ['All'] + sorted(df['solver_model'].unique().tolist())
        selected_solver = st.selectbox(
            "Select Solver Model",
            options=solver_models,
            key="solver_filter"
        )
    
    # Filter based on all selections
    filtered_df = df[df['status'] == 'success']  # Start with successful attempts
    
    # Filter by correctness
    if show_correct and not show_incorrect:
        filtered_df = filtered_df[filtered_df['is_correct']]
    elif show_incorrect and not show_correct:
        filtered_df = filtered_df[~filtered_df['is_correct']]
    elif not show_correct and not show_incorrect:
        filtered_df = filtered_df[filtered_df['is_correct'] != filtered_df['is_correct']]  # Empty DataFrame
    
    # Filter by riddler model
    if selected_riddler != 'All':
        filtered_df = filtered_df[filtered_df['riddler_model'] == selected_riddler]
    
    # Filter by solver model
    if selected_solver != 'All':
        filtered_df = filtered_df[filtered_df['solver_model'] == selected_solver]
    
    # Display individual riddles
    st.markdown("### 🎯 Can You Solve These Riddles?")
    page = 0
    per_page = 5
    
    while True:
        display_riddles(filtered_df, page, per_page)
        
        command = input("\nEnter command (n: next page, p: previous page, q: quit, number: go to page): ").lower()
        
        if command == 'q':
            break
        elif command == 'n':
            page += 1
        elif command == 'p':
            page = max(0, page - 1)
        elif command.isdigit():
            new_page = int(command) - 1  # Convert to 0-based indexing
            if 0 <= new_page < (len(filtered_df) + per_page - 1) // per_page:
                page = new_page
            else:
                print("Invalid page number")
        else:
            print("Invalid command")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Riddle Game",
        page_icon="🧩",
        layout="wide"
    )
    main() 