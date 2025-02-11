import streamlit as st
import pandas as pd
import pathlib

import plotly.graph_objects as go
import ast

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
        if "Defender:" in msg:
            st.markdown(f"🛡️ {msg}")
        else:
            st.markdown(f"⚔️ {msg}")

def format_full_response(response_data: dict) -> None:
    """Format a single full response with detailed information"""
    role_icons = {
        "defender": "🛡️",
        "attacker": "⚔️"
    }
    
    st.markdown(f"**Turn:** {response_data['turn']}")
    st.markdown(f"**Role:** {role_icons[response_data['role']]} {response_data['role'].title()}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Full Response:**")
        st.text(response_data['full_response'])
    
    with col2:
        st.markdown("**Extracted Content:**")
        if 'extracted_message' in response_data:
            st.text(response_data['extracted_message'])
        elif 'extracted_decision' in response_data:
            st.text(f"Decision: {response_data['extracted_decision']}")

def standardize_decision(decision: str) -> str:
    """Standardize decision strings to COOPERATE, DEFECT, or OTHER"""
    decision = str(decision).upper().strip()
    if decision == "COOPERATE":
        return "COOPERATE"
    elif decision == "DEFECT":
        return "DEFECT"
    return "OTHER"

def extract_final_reasoning(conversation_str: str) -> tuple[str, str]:
    """
    Extract the final reasoning from the conversation.
    
    Args:
        conversation_str: String representation of the conversation list
        
    Returns:
        tuple[str, str]: Attacker's and defender's final reasoning
    """
    try:
        conversation = ast.literal_eval(conversation_str)
        attacker_reasoning = ""
        defender_reasoning = ""
        
        # Look for the last reasoning from each participant
        for msg in reversed(conversation):
            if "Reasoning:" in msg:
                if "Attacker:" in msg and not attacker_reasoning:
                    attacker_reasoning = msg.split("Reasoning:", 1)[1].strip()
                elif "Defender:" in msg and not defender_reasoning:
                    defender_reasoning = msg.split("Reasoning:", 1)[1].strip()
            
            if attacker_reasoning and defender_reasoning:
                break
                
        return attacker_reasoning, defender_reasoning
    except Exception:
        return "Error parsing conversation", "Error parsing conversation"

def format_final_reasoning(row: pd.Series) -> None:
    """Format the final reasoning from both participants"""
    attacker_reasoning, defender_reasoning = extract_final_reasoning(row['conversation'])
    
    st.markdown("#### Attacker's Final Reasoning")
    st.markdown(f"🔵 {attacker_reasoning}")
    
    st.markdown("#### Defender's Final Reasoning")
    st.markdown(f"🔴 {defender_reasoning}")

def format_conversation_with_decision(row: pd.Series) -> None:
    """Format conversation and highlight the final decision"""
    # Show full conversation first
    conversation = ast.literal_eval(row['conversation'])
    for msg in conversation:
        if "Defender:" in msg:
            st.markdown(f"🛡️ {msg}")
        else:
            st.markdown(f"⚔️ {msg}")
    
    # Get the full responses
    full_responses = ast.literal_eval(row['full_responses'])
    
    # Find the defender's final decision response
    final_response = next(
        (r for r in full_responses 
         if r['role'] == 'defender' and r['turn'] == 'final'),
        None
    )
    
    if final_response:
        st.markdown("---")
        st.markdown("### Final Decision")
        st.markdown("**Defender's Full Response:**")
        st.text(final_response['full_response'])
        st.markdown(f"**Extracted Decision:** {row['defender_decision']}")
        st.markdown(f"**Was this optimal?** {'Yes ✅' if row['optimal_decision'] else 'No ❌'}")
    else:
        st.error("Could not find defender's final decision response")

def main():
    st.title("🔒 Prisoner's Dilemma Analysis (asymmetric)")
    
    df = load_results()
    
    # Standardize decisions
    df['defender_decision'] = df['defender_decision'].apply(standardize_decision)
    
    # Overall metrics
    st.header("📊 Overall Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        success_rate = (df['status'] == 'success').mean()
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    with col2:
        optimal_rate = df[df['status'] == 'success']['optimal_decision'].mean()
        st.metric("Optimal Decision Rate", f"{optimal_rate:.1%}")
    
    with col3:
        total_games = len(df)
        st.metric("Total Games", total_games)
    
    # Model performance heatmap
    st.header("🎯 Model Performance Matrix")
    
    success_df = df[df['status'] == 'success']
    
    # Calculate cross-model deception rates
    model_pairs = []
    for attacker_model in success_df['attacker_model'].unique():
        for defender_model in success_df['defender_model'].unique():
            pair_games = success_df[
                (success_df['attacker_model'] == attacker_model) & 
                (success_df['defender_model'] == defender_model)
            ]
            if len(pair_games) > 0:
                # Attacker's success rate (getting defender to cooperate)
                deception_rate = (pair_games['defender_decision'] == 'COOPERATE').mean()
                model_pairs.append({
                    'attacker_model': attacker_model,
                    'defender_model': defender_model,
                    'deception_rate': deception_rate,
                    'total_games': len(pair_games)
                })
    
    pair_df = pd.DataFrame(model_pairs)
    pivot = pd.pivot_table(
        pair_df,
        values='deception_rate',
        index='defender_model',  # Y-axis: model being attacked
        columns='attacker_model',  # X-axis: model trying to deceive
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
        title="Model Deception Success Rates (Getting Defender to Cooperate)",
        xaxis_title="Attacker Model",
        yaxis_title="Defender Model",
        height=500
    )
    
    # Add hover template to show the actual values
    fig.update_traces(
        hovertemplate="Attacker: %{x}<br>Defender: %{y}<br>Deception Rate: %{z:.1%}<extra></extra>"
    )
    
    st.plotly_chart(fig)
    
    # Points Analysis
    st.header("🏆 Points Analysis")
    
    # Calculate average points per model as attacker and defender
    attacker_points = success_df.groupby('attacker_model')['attacker_points'].mean().round(2)
    defender_points = success_df.groupby('defender_model')['defender_points'].mean().round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Points as Attacker")
        st.dataframe(attacker_points)
    
    with col2:
        st.subheader("Average Points as Defender")
        st.dataframe(defender_points)
    
    # Conversation viewer
    st.header("💬 Game Analysis")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        selected_attacker = st.selectbox(
            "Filter by Attacker Model",
            options=['All'] + list(df['attacker_model'].unique())
        )
    with col2:
        selected_defender = st.selectbox(
            "Filter by Defender Model",
            options=['All'] + list(df['defender_model'].unique())
        )
    
    # Filter conversations
    filtered_df = df[df['status'] == 'success']
    if selected_attacker != 'All':
        filtered_df = filtered_df[filtered_df['attacker_model'] == selected_attacker]
    if selected_defender != 'All':
        filtered_df = filtered_df[filtered_df['defender_model'] == selected_defender]
    
    # Display conversations and decisions
    for idx, row in filtered_df.iterrows():
        st.markdown("---")
        st.markdown(f"### Game {idx + 1}")
        st.markdown(f"**Models:** Attacker: {row['attacker_model']} vs Defender: {row['defender_model']}")
        
        with st.expander("View Conversation and Decision"):
            format_conversation_with_decision(row)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Prisoner's Dilemma Analysis",
        page_icon="🔒",
        layout="wide"
    )
    main() 