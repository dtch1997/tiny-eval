import streamlit as st
import pandas as pd
import pathlib
import plotly.express as px
import plotly.graph_objects as go

curr_dir = pathlib.Path(__file__).parent
results_path = curr_dir / "results" / "results.csv"

def load_results() -> pd.DataFrame:
    """Load results from CSV file"""
    if not results_path.exists():
        st.error("No results file found. Please run main.py first.")
        st.stop()
    return pd.read_csv(results_path)

def format_conversation(conversation: list[str]) -> None:
    """Format conversation with alternating colors"""
    for msg in conversation:
        if "Prisoner A:" in msg:
            st.markdown(f"🔵 {msg}")
        else:
            st.markdown(f"🔴 {msg}")

def standardize_decision(decision: str) -> str:
    """Standardize decision strings to COOPERATE, DEFECT, or OTHER"""
    decision = str(decision).upper().strip()
    if decision == "COOPERATE":
        return "COOPERATE"
    elif decision == "DEFECT":
        return "DEFECT"
    return "OTHER"

def main():
    st.title("🔒 Prisoner's Dilemma Analysis")
    
    df = load_results()
    
    # Standardize decisions
    df['decision_a'] = df['decision_a'].apply(standardize_decision)
    df['decision_b'] = df['decision_b'].apply(standardize_decision)
    
    # Overall metrics
    st.header("📊 Overall Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        success_rate = (df['status'] == 'success').mean()
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    with col2:
        optimal_rate = df[df['status'] == 'success'][['optimal_a', 'optimal_b']].mean().mean()
        st.metric("Optimal Decision Rate", f"{optimal_rate:.1%}")
    
    with col3:
        total_games = len(df)
        st.metric("Total Games", total_games)
    
    # Model performance heatmap
    st.header("🎯 Model Performance Matrix")
    
    success_df = df[df['status'] == 'success']
    
    # Calculate cross-model win rates
    model_pairs = []
    for model_a in success_df['prisoner_a_model'].unique():
        for model_b in success_df['prisoner_b_model'].unique():
            pair_games = success_df[
                (success_df['prisoner_a_model'] == model_a) & 
                (success_df['prisoner_b_model'] == model_b)
            ]
            if len(pair_games) > 0:
                # Model A's win rate (getting B to cooperate)
                model_a_winrate = len(pair_games[pair_games['decision_b'] == 'COOPERATE']) / len(pair_games)
                model_pairs.append({
                    'attacking_model': model_a,
                    'defending_model': model_b,
                    'win_rate': model_a_winrate,
                    'total_games': len(pair_games)
                })
    
    pair_df = pd.DataFrame(model_pairs)
    pivot = pd.pivot_table(
        pair_df,
        values='win_rate',
        index='defending_model',  # Y-axis: model being played against
        columns='attacking_model',  # X-axis: model whose winrate we're calculating
        aggfunc='first'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, '#f7fbff'], [1, '#08306b']],  # Light blue to dark blue colorscale
        zmin=0,
        zmax=1,
        hoverongaps=False,
        text=[[f"{val:.1%}" for val in row] for row in pivot.values],  # Add text for each cell
        texttemplate="%{text}",  # Show the text in each cell
        textfont={"color": "black"},  # Make text black for better visibility
    ))
    
    fig.update_layout(
        title="Model Win Rates (Getting Opponent to Cooperate)",
        xaxis_title="Attacking Model",
        yaxis_title="Defending Model",
        height=500
    )
    
    # Add hover template to show the actual values
    fig.update_traces(
        hovertemplate="Attacking: %{x}<br>Defending: %{y}<br>Win Rate: %{z:.1%}<extra></extra>"
    )
    
    st.plotly_chart(fig)
    
    # Conversation viewer
    st.header("💬 Conversation Viewer")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        selected_model_a = st.selectbox(
            "Filter by Prisoner A Model",
            options=['All'] + list(df['prisoner_a_model'].unique())
        )
    with col2:
        selected_model_b = st.selectbox(
            "Filter by Prisoner B Model",
            options=['All'] + list(df['prisoner_b_model'].unique())
        )
    
    # Filter conversations
    filtered_df = success_df
    if selected_model_a != 'All':
        filtered_df = filtered_df[filtered_df['prisoner_a_model'] == selected_model_a]
    if selected_model_b != 'All':
        filtered_df = filtered_df[filtered_df['prisoner_b_model'] == selected_model_b]
    
    # Display conversations
    for idx, row in filtered_df.iterrows():
        st.markdown("---")
        st.markdown(f"### Game {idx + 1}")
        st.markdown(f"**Models:** {row['prisoner_a_model']} vs {row['prisoner_b_model']}")
        st.markdown(f"**Decisions:** A: {row['decision_a']}, B: {row['decision_b']}")
        st.markdown(f"**Sentences:** A: {row['sentence_a']} years, B: {row['sentence_b']} years")
        
        with st.expander("View Conversation"):
            format_conversation(eval(row['conversation']))

if __name__ == "__main__":
    st.set_page_config(
        page_title="Prisoner's Dilemma Analysis",
        page_icon="🔒",
        layout="wide"
    )
    main() 