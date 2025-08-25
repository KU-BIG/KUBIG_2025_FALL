import os
import io
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI

# --- optional: .env 지원 ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===== Project modules (your existing code) =====
from config import (
    PATH_BUSINESS, PATH_REVIEW,
    PATH_REVIEWS_FILTERED, PATH_SENT_WITH_SENT, PATH_WITH_TOPICS,
    SENTIMENT_MODEL, EMBEDDING_MODEL
)
from modules.find_business_ids import find_business_ids
from modules.filter_reviews import filter_reviews_by_business_ids
from modules.sentence_sentiment import run_sentence_sentiment
from modules.topic_model_final import apply_bertopic_for_business
from modules.business_meta import load_business_meta

# ========= Enhanced Dashboard Functions =========

def create_kpi_cards(summary_df: pd.DataFrame, sentences_df: pd.DataFrame) -> None:
    """Display key metrics in card format with full multi-label support"""
    
    # Calculate key metrics
    total_reviews = sentences_df['review_id'].nunique() if 'review_id' in sentences_df.columns else len(sentences_df)
    total_topics = len(summary_df[summary_df['topic_id'] != -1])
    avg_satisfaction = summary_df['stars_mean'].mean()
    overall_positive_rate = summary_df['pos'].mean()
    
    # Process labels for better analysis - NO primary label selection
    df = summary_df[summary_df['topic_id'] != -1].copy()
    df['display_label'] = df['label'].apply(lambda x: get_smart_display_label(x, 60))
    df['compact_label'] = df['label'].apply(lambda x: get_compact_display_label(x, 40))
    df['all_labels'] = df['label'].apply(lambda x: '_'.join(parse_multiple_labels(x)))
    df['label_count'] = df['label'].apply(lambda x: len(parse_multiple_labels(x)))
    
    # Most problematic and best topics (using ALL labels)
    worst_topic = df.loc[df['pos'].idxmin()] if not df.empty else None
    best_topic = df.loc[df['pos'].idxmax()] if not df.empty else None
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Reviews Analyzed", 
            value=f"{total_reviews:,}",
            help="Total number of reviews processed"
        )
    
    with col2:
        st.metric(
            label="🎯 Multi-Aspect Topics", 
            value=total_topics,
            help="Number of distinct multi-label topics identified"
        )
    
    with col3:
        st.metric(
            label="⭐ Average Satisfaction", 
            value=f"{avg_satisfaction:.1f}/5.0",
            delta=f"{(avg_satisfaction - 3.0):.1f} vs neutral",
            help="Average star rating across all topics"
        )
    
    with col4:
        st.metric(
            label="😊 Positive Sentiment Rate", 
            value=f"{overall_positive_rate:.1%}",
            delta=f"{(overall_positive_rate - 0.5):.1%} vs 50%",
            help="Overall positive sentiment rate"
        )
    
    # Alert cards for critical insights with full multi-label info
    if worst_topic is not None and best_topic is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Show all aspects for the worst topic
            worst_labels = parse_multiple_labels(worst_topic['label'])
            aspect_info = f"<br><small style='color: #888;'>Covers {len(worst_labels)} aspects: " + ", ".join(worst_labels[:2])
            if len(worst_labels) > 2:
                aspect_info += f" + {len(worst_labels)-2} more"
            aspect_info += "</small>"
            
            st.markdown(f"""
            <div style='padding: 1rem; background: linear-gradient(135deg, #fff2f2 0%, #ffe6e6 100%); border-left: 4px solid #ff4444; border-radius: 8px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4 style='color: #cc0000; margin: 0; font-size: 1.1rem;'>🚨 Most Critical Multi-Aspect Issue</h4>
                <p style='margin: 0.5rem 0; font-weight: bold; color: #333; line-height: 1.3;'>{worst_topic['compact_label']}</p>
                <p style='margin: 0; color: #666; font-size: 0.9rem;'>Positive Rate: {worst_topic['pos']:.1%} | Avg Stars: {worst_topic['stars_mean']:.1f}</p>
                {aspect_info}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Show all aspects for the best topic
            best_labels = parse_multiple_labels(best_topic['label'])
            aspect_info = f"<br><small style='color: #888;'>Covers {len(best_labels)} aspects: " + ", ".join(best_labels[:2])
            if len(best_labels) > 2:
                aspect_info += f" + {len(best_labels)-2} more"
            aspect_info += "</small>"
            
            st.markdown(f"""
            <div style='padding: 1rem; background: linear-gradient(135deg, #f2fff2 0%, #e6ffe6 100%); border-left: 4px solid #44ff44; border-radius: 8px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4 style='color: #006600; margin: 0; font-size: 1.1rem;'>✨ Strongest Multi-Aspect Performance</h4>
                <p style='margin: 0.5rem 0; font-weight: bold; color: #333; line-height: 1.3;'>{best_topic['compact_label']}</p>
                <p style='margin: 0; color: #666; font-size: 0.9rem;'>Positive Rate: {best_topic['pos']:.1%} | Avg Stars: {best_topic['stars_mean']:.1f}</p>
                {aspect_info}
            </div>
            """, unsafe_allow_html=True)

def parse_multiple_labels(label_str: str) -> List[str]:
    """Parse underscore-separated multiple labels from zero-shot classification"""
    if pd.isna(label_str) or not label_str:
        return []
    
    # Split by underscore and clean up
    labels = [label.strip() for label in str(label_str).split('_')]
    labels = [label for label in labels if label and not label.isdigit()]  # Remove empty strings and pure numbers
    return labels

def get_smart_display_label(label_str: str, max_length: int = 100) -> str:
    """Get a smart display version of all labels - no truncation of meaning"""
    labels = parse_multiple_labels(label_str)
    if not labels:
        return str(label_str)
    
    # If single label, show it fully
    if len(labels) == 1:
        return labels[0]
    
    # For multiple labels, show all but smartly formatted
    if len(labels) <= 3:
        # For 2-3 labels, show all separated by " + "
        return " + ".join(labels)
    else:
        # For 4+ labels, show first 2 and count
        return f"{labels[0]} + {labels[1]} + {len(labels)-2} more aspects"

def get_compact_display_label(label_str: str, max_length: int = 50) -> str:
    """Get compact version for charts while preserving meaning"""
    labels = parse_multiple_labels(label_str)
    if not labels:
        return str(label_str)
    
    if len(labels) == 1:
        # Extract main concept before parentheses
        main_part = labels[0].split('(')[0].strip()
        return main_part if main_part else labels[0]
    
    # For multiple labels, show main concepts
    main_concepts = []
    for label in labels[:2]:  # Show first 2
        main_part = label.split('(')[0].strip()
        main_concepts.append(main_part if main_part else label)
    
    if len(labels) > 2:
        return f"{' + '.join(main_concepts)} + {len(labels)-2} more"
    else:
        return " + ".join(main_concepts)

def create_topic_overview_chart(summary_df: pd.DataFrame) -> None:
    """Create comprehensive topic overview visualization with full multi-label support"""
    
    # Filter out noise topic
    df = summary_df[summary_df['topic_id'] != -1].copy()
    
    # Process multiple labels - show ALL labels meaningfully
    df['display_label'] = df['label'].apply(lambda x: get_smart_display_label(x, 120))
    df['compact_label'] = df['label'].apply(lambda x: get_compact_display_label(x, 60))
    df['all_labels'] = df['label'].apply(lambda x: '_'.join(parse_multiple_labels(x)))
    df['label_count'] = df['label'].apply(lambda x: len(parse_multiple_labels(x)))
    
    df = df.sort_values('n', ascending=True)  # Sort by volume for better readability
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Topic Performance Overview (All Aspects)', 'Topic Volume Distribution'),
        specs=[[{"secondary_y": True}, {"type": "pie"}]],
        column_widths=[0.7, 0.3]
    )
    
    # Left plot: Enhanced bubble chart with full multi-label info
    hover_text = []
    for _, row in df.iterrows():
        labels = parse_multiple_labels(row['label'])
        label_display = "<br>".join([f"• {label}" for label in labels])  # Show ALL labels, no truncation
        
        hover_text.append(
            f"<b>Multi-Aspect Topic</b><br>"
            f"All {len(labels)} Aspects:<br>{label_display}<br><br>"
            f"Avg Stars: {row['stars_mean']:.1f}<br>"
            f"Positive Rate: {row['pos']:.1%}<br>"
            f"Volume: {row['n']} reviews<br>"
            f"Share: {row['share']:.1%}"
        )
    
    fig.add_trace(
        go.Scatter(
            x=df['stars_mean'],
            y=df['pos'],
            mode='markers',  # Remove text overlay to avoid clutter
            marker=dict(
                size=df['n'],
                sizemode='diameter',
                sizeref=df['n'].max()/35,  # Adjusted size
                color=df['pos'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Positive Rate", x=0.45),
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_text,
            name="Multi-Label Topics"
        ),
        row=1, col=1
    )
    
    # Add text annotations separately for better control - use compact labels
    for _, row in df.iterrows():
        fig.add_annotation(
            x=row['stars_mean'],
            y=row['pos'],
            text=row['compact_label'],
            showarrow=False,
            font=dict(size=8, color='black'),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            row=1, col=1
        )
    
    # Right plot: Pie chart with display labels
    fig.add_trace(
        go.Pie(
            labels=df['compact_label'],
            values=df['n'],
            hovertemplate="<b>%{label}</b><br>Reviews: %{value}<br>Share: %{percent}<extra></extra>",
            name="Volume",
            textinfo='percent',
            textposition='auto',  # Changed to auto for better positioning
            textfont=dict(size=9)
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Average Star Rating", row=1, col=1, range=[0.5, 5.5])
    fig.update_yaxes(title_text="Positive Sentiment Rate", row=1, col=1, range=[0, 1])
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Multi-Label Topic Analysis Overview",
        title_x=0.5,
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show comprehensive label analysis
    with st.expander("🏷️ Comprehensive Multi-Label Analysis", expanded=True):
        st.subheader("All Zero-Shot Labels by Topic")
        
        # Create a comprehensive view
        for _, row in df.sort_values('n', ascending=False).head(10).iterrows():
            labels = parse_multiple_labels(row['label'])
            
            with st.container():
                # Header with metrics
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.markdown(f"**Topic {row['topic_id']}: {len(labels)} Aspects**")
                with col2:
                    st.metric("Reviews", f"{row['n']:,}")
                with col3:
                    st.metric("Positive", f"{row['pos']:.1%}")
                with col4:
                    st.metric("Stars", f"{row['stars_mean']:.1f}")
                
                # All labels in a nice layout
                st.markdown("**All Zero-Shot Classifications:**")
                for i, label in enumerate(labels, 1):
                    # Highlight main concept
                    main_concept = label.split('(')[0].strip()
                    detail_part = f"({label.split('(', 1)[1]}" if '(' in label else ""
                    
                    st.markdown(f"{i}. **{main_concept}** {detail_part}")
                
                st.markdown("---")
        
        # Label frequency analysis across all topics
        st.subheader("Most Common Aspect Categories")
        all_individual_labels = []
        for _, row in df.iterrows():
            labels = parse_multiple_labels(row['label'])
            for label in labels:
                # Extract main concept for categorization
                main_concept = label.split('(')[0].strip()
                all_individual_labels.append((main_concept, row['n']))  # Weight by review count
        
        if all_individual_labels:
            label_weights = {}
            for label, weight in all_individual_labels:
                label_weights[label] = label_weights.get(label, 0) + weight
            
            label_freq_df = pd.DataFrame([
                {'Aspect_Category': label, 'Weighted_Frequency': freq, 'Appears_In_Topics': 1} 
                for label, freq in label_weights.items()
            ]).sort_values('Weighted_Frequency', ascending=False)
            
            # Count how many topics each aspect appears in
            aspect_counts = {}
            for _, row in df.iterrows():
                labels = parse_multiple_labels(row['label'])
                for label in labels:
                    main_concept = label.split('(')[0].strip()
                    aspect_counts[main_concept] = aspect_counts.get(main_concept, 0) + 1
            
            label_freq_df['Appears_In_Topics'] = label_freq_df['Aspect_Category'].map(aspect_counts)
            
            st.dataframe(
                label_freq_df.head(15), 
                use_container_width=True,
                column_config={
                    "Aspect_Category": st.column_config.TextColumn("Aspect Category", width="large"),
                    "Weighted_Frequency": st.column_config.NumberColumn("Total Review Volume", format="%d"),
                    "Appears_In_Topics": st.column_config.NumberColumn("# Topics", width="small")
                }
            )

def create_priority_matrix(summary_df: pd.DataFrame) -> None:
    """Create actionable priority matrix with full multi-label support"""
    
    df = summary_df[summary_df['topic_id'] != -1].copy()
    df['display_label'] = df['label'].apply(lambda x: get_smart_display_label(x, 80))
    df['compact_label'] = df['label'].apply(lambda x: get_compact_display_label(x, 50))
    df['all_labels'] = df['label'].apply(lambda x: '_'.join(parse_multiple_labels(x)))
    df['label_count'] = df['label'].apply(lambda x: len(parse_multiple_labels(x)))
    
    # Calculate priority scores
    df['impact_score'] = df['share'] * (1 - df['pos'])  # High volume + low satisfaction = high impact
    df['urgency_score'] = (5 - df['stars_mean']) / 4  # Low stars = high urgency
    
    # Categorize topics
    df['priority'] = 'Low'
    df.loc[(df['impact_score'] > df['impact_score'].median()) | (df['urgency_score'] > 0.4), 'priority'] = 'Medium'
    df.loc[(df['impact_score'] > df['impact_score'].quantile(0.75)) & (df['urgency_score'] > 0.6), 'priority'] = 'High'
    
    # Color mapping
    color_map = {'High': '#ff4444', 'Medium': '#ffaa44', 'Low': '#44ff44'}
    
    fig = go.Figure()
    
    for priority in ['High', 'Medium', 'Low']:
        priority_data = df[df['priority'] == priority]
        if not priority_data.empty:
            hover_text = []
            for _, row in priority_data.iterrows():
                all_labels = parse_multiple_labels(row['label'])
                labels_preview = "<br>".join([f"• {label}" for label in all_labels])  # Show ALL labels
                
                hover_text.append(
                    f"<b>Multi-Aspect Topic</b><br>"
                    f"All {len(all_labels)} Aspects:<br>{labels_preview}<br><br>"
                    f"Priority: {priority}<br>"
                    f"Impact Score: {row['impact_score']:.3f}<br>"
                    f"Urgency Score: {row['urgency_score']:.3f}<br>"
                    f"Volume: {row['n']} reviews<br>"
                    f"Positive Rate: {row['pos']:.1%}<br>"
                )
            
            fig.add_trace(go.Scatter(
                x=priority_data['impact_score'],
                y=priority_data['urgency_score'],
                mode='markers',
                marker=dict(
                    size=priority_data['n'],
                    sizemode='diameter',
                    sizeref=df['n'].max()/25,
                    color=color_map[priority],
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                name=f'{priority} Priority',
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_text
            ))
            
            # Add text annotations using compact labels
            for _, row in priority_data.iterrows():
                fig.add_annotation(
                    x=row['impact_score'],
                    y=row['urgency_score'],
                    text=row['compact_label'],
                    showarrow=False,
                    font=dict(size=8, color='black'),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=1
                )
    
    fig.update_layout(
        title="Multi-Aspect Topic Priority Matrix",
        xaxis_title="Impact Score (Volume × Dissatisfaction)",
        yaxis_title="Urgency Score (Based on Star Rating)",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(r=120)
    )
    
    # Add quadrant lines
    fig.add_hline(y=df['urgency_score'].median(), line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=df['impact_score'].median(), line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=df['impact_score'].max()*0.8, y=df['urgency_score'].max()*0.9, 
                      text="High Impact<br>High Urgency", showarrow=False, 
                      font=dict(size=12, color="red"), bgcolor="rgba(255,255,255,0.8)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Action recommendations with full multi-label info
    high_priority = df[df['priority'] == 'High'].sort_values('impact_score', ascending=False)
    if not high_priority.empty:
        st.subheader("🚨 Immediate Action Required - Multi-Aspect Issues")
        for _, topic in high_priority.head(3).iterrows():
            all_labels = parse_multiple_labels(topic['label'])
            
            with st.expander(f"**{topic['compact_label']}** - Impact Score: {topic['impact_score']:.3f}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Volume", f"{topic['n']:,} reviews", f"{topic['share']:.1%} of total")
                with col2:
                    st.metric("Satisfaction", f"{topic['stars_mean']:.1f}/5.0", f"{topic['pos']:.1%} positive")
                with col3:
                    st.metric("Complexity", f"{len(all_labels)} aspects")
                
                # Show ALL related aspects without truncation
                st.subheader(f"All {len(all_labels)} Related Aspects:")
                for i, label in enumerate(all_labels, 1):
                    # Highlight main concept and show details
                    main_concept = label.split('(')[0].strip()
                    detail_part = f"({label.split('(', 1)[1]}" if '(' in label else ""
                    
                    st.markdown(f"{i}. **{main_concept}** {detail_part}")
    else:
        st.info("✅ No high-priority multi-aspect issues identified at this time")

def create_sentiment_timeline(sentences_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Create clean and readable sentiment timeline analysis"""
    
    if 'date' not in sentences_df.columns:
        st.warning("Date information not available for timeline analysis")
        return
    
    # Prepare timeline data
    timeline_df = sentences_df.copy()
    timeline_df['date'] = pd.to_datetime(timeline_df['date'], errors='coerce')
    timeline_df = timeline_df.dropna(subset=['date'])
    timeline_df['month'] = timeline_df['date'].dt.to_period('M').astype(str)
    
    # Get top topics and prepare clean labels
    df_with_labels = summary_df[summary_df['topic_id'] != -1].copy()
    df_with_labels['clean_label'] = df_with_labels['label'].apply(
        lambda x: parse_multiple_labels(x)[0].split('(')[0].strip() if parse_multiple_labels(x) else f"Topic {x}"
    )
    
    # User controls for better experience
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_topics = st.selectbox("Number of topics to show", [2, 3, 4, 5], index=1, key="timeline_topics")
    
    with col2:
        time_grouping = st.selectbox("Time grouping", ["Monthly", "Quarterly"], index=0, key="timeline_grouping")
    
    with col3:
        min_reviews = st.slider("Min reviews per period", 3, 20, 8, key="timeline_min_reviews")
    
    # Select top topics by volume
    top_topics = df_with_labels.nlargest(num_topics, 'n')['topic_id'].tolist()
    timeline_df = timeline_df[timeline_df['topic_id'].isin(top_topics)]
    
    # Create label mapping
    clean_label_mapping = dict(zip(df_with_labels['topic_id'], df_with_labels['clean_label']))
    
    # Group by time period
    if time_grouping == "Quarterly":
        timeline_df['period'] = pd.to_datetime(timeline_df['date']).dt.to_period('Q').astype(str)
    else:
        timeline_df['period'] = timeline_df['month']
    
    # Get all available periods for consistent timeline
    all_periods = sorted(timeline_df['period'].unique())
    
    # Calculate metrics for each topic with consistent periods
    monthly_metrics = []
    for topic_id in top_topics:
        topic_data = timeline_df[timeline_df['topic_id'] == topic_id]
        clean_label = clean_label_mapping.get(topic_id, f"Topic {topic_id}")
        
        if topic_data.empty:
            continue
            
        # Group by period and calculate metrics
        monthly = topic_data.groupby('period').agg({
            'sentiment': lambda x: (x == 1).mean() if 'sentiment' in timeline_df.columns else 0.5,
            'stars': 'mean',
            'topic_id': 'count'
        }).reset_index()
        
        # Filter periods with too few reviews
        monthly = monthly[monthly['topic_id'] >= min_reviews]
        
        # Only keep topics that have enough data points
        if len(monthly) < 3:  # Need at least 3 data points
            continue
            
        monthly['topic_label'] = clean_label
        monthly['topic_id_orig'] = topic_id
        monthly['review_count'] = monthly['topic_id']
        monthly_metrics.append(monthly)
    
    if not monthly_metrics:
        st.info(f"Not enough data for timeline analysis. Try reducing minimum reviews to {min_reviews//2} or fewer topics.")
        return
    
    combined_timeline = pd.concat(monthly_metrics, ignore_index=True)
    
    # Create clean visualization
    fig = go.Figure()
    
    # Clean color palette - high contrast
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#2D7D32']
    
    # Plot each topic separately to avoid connection issues
    for i, topic_id in enumerate(top_topics):
        topic_data = combined_timeline[combined_timeline['topic_id_orig'] == topic_id]
        if topic_data.empty:
            continue
            
        # Sort data properly by period
        topic_data = topic_data.sort_values('period')
        
        topic_label = topic_data['topic_label'].iloc[0]
        color = colors[i % len(colors)]
        
        # Add trace with clean settings
        fig.add_trace(go.Scatter(
            x=topic_data['period'],
            y=topic_data['sentiment'],
            mode='lines+markers',
            name=topic_label,
            line=dict(
                color=color, 
                width=3,
                shape='linear'  # Ensure linear interpolation
            ),
            marker=dict(
                size=8,
                color=color,
                line=dict(color='white', width=2)
            ),
            connectgaps=False,  # Don't connect gaps in data
            hovertemplate=(
                f"<b>{topic_label}</b><br>"
                "Period: %{x}<br>"
                "Positive Rate: %{y:.1%}<br>"
                "Reviews: %{customdata}<br>"
                "<extra></extra>"
            ),
            customdata=topic_data['review_count']
        ))
    
    # Clean layout
    fig.update_layout(
        title={
            'text': f"Sentiment Trends - Top {num_topics} Topics ({time_grouping})",
            'x': 0.5,
            'font': {'size': 16}
        },
        xaxis_title="Time Period",
        yaxis_title="Positive Sentiment Rate",
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=80, b=80, l=60, r=60),
        yaxis=dict(
            range=[0, 1],
            tickformat='.0%',
            gridcolor='lightgray',
            gridwidth=1
        ),
        xaxis=dict(
            tickangle=45,
            gridcolor='lightgray',
            gridwidth=1
        ),
        plot_bgcolor='white',
        showlegend=True
    )
    
    # Add clean reference line
    fig.add_hline(
        y=0.5, 
        line_dash="dot", 
        line_color="gray", 
        opacity=0.5, 
        annotation_text="Neutral (50%)", 
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_font_color="gray"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Clean trend analysis
    if len(combined_timeline['period'].unique()) >= 3:
        st.subheader("📊 Trend Analysis")
        
        trend_analysis = []
        for topic_id in top_topics:
            topic_timeline = combined_timeline[combined_timeline['topic_id_orig'] == topic_id].sort_values('period')
            if len(topic_timeline) >= 3:
                # Calculate trend more robustly
                recent_periods = min(3, len(topic_timeline) // 2)
                recent_sentiment = topic_timeline.tail(recent_periods)['sentiment'].mean()
                earlier_sentiment = topic_timeline.head(recent_periods)['sentiment'].mean()
                trend = recent_sentiment - earlier_sentiment
                
                trend_analysis.append({
                    'topic_id': topic_id,
                    'topic_label': topic_timeline['topic_label'].iloc[0],
                    'trend': trend,
                    'recent_sentiment': recent_sentiment,
                    'total_reviews': topic_timeline['review_count'].sum(),
                    'data_points': len(topic_timeline)
                })
        
        if trend_analysis:
            trend_df = pd.DataFrame(trend_analysis)
            
            # Clean trend display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Improving")
                improving = trend_df[trend_df['trend'] > 0.05].sort_values('trend', ascending=False)
                if not improving.empty:
                    for _, topic in improving.iterrows():
                        st.success(f"**{topic['topic_label']}** (+{topic['trend']:.1%})")
                        st.caption(f"Recent: {topic['recent_sentiment']:.1%} • {topic['total_reviews']} reviews • {topic['data_points']} periods")
                else:
                    st.info("No significantly improving trends")
            
            with col2:
                st.markdown("#### 📉 Declining")
                declining = trend_df[trend_df['trend'] < -0.05].sort_values('trend')
                if not declining.empty:
                    for _, topic in declining.iterrows():
                        st.error(f"**{topic['topic_label']}** ({topic['trend']:.1%})")
                        st.caption(f"Recent: {topic['recent_sentiment']:.1%} • {topic['total_reviews']} reviews • {topic['data_points']} periods")
                else:
                    st.info("No significantly declining trends")
        
        # Summary stats
        st.subheader("📋 Period Summary")
        total_periods = len(combined_timeline['period'].unique())
        total_reviews = combined_timeline['review_count'].sum()
        avg_sentiment = combined_timeline['sentiment'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time Periods", total_periods)
        with col2:
            st.metric("Total Reviews", f"{total_reviews:,}")
        with col3:
            st.metric("Average Sentiment", f"{avg_sentiment:.1%}")
    
    else:
        st.info("Insufficient data points for trend analysis (need at least 3 periods)")
    
    # Data quality info
    with st.expander("ℹ️ Data Quality Info"):
        st.write(f"**Analysis Settings:**")
        st.write(f"- Showing top {num_topics} topics by volume")
        st.write(f"- Time grouping: {time_grouping}")
        st.write(f"- Minimum {min_reviews} reviews per period")
        st.write(f"- Minimum 3 data points per topic")
        st.write(f"- Data points per topic:")
        
        for topic_id in top_topics:
            topic_data = combined_timeline[combined_timeline['topic_id_orig'] == topic_id]
            if not topic_data.empty:
                topic_label = topic_data['topic_label'].iloc[0]
                st.write(f"  • {topic_label}: {len(topic_data)} periods")
        
        st.write(f"- connectgaps=False (gaps in data are not connected)")
        st.write(f"- Total valid data points: {len(combined_timeline)} period-topic combinations")

def generate_enhanced_llm_insights(summary_df: pd.DataFrame, sentences_df: pd.DataFrame) -> str:
    """Enhanced LLM insights with rich context and examples"""
    
    API_KEY = os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
    if not API_KEY:
        return None
    
    # 기본 통계 수집
    total_reviews = sentences_df['review_id'].nunique() if 'review_id' in sentences_df.columns else len(sentences_df)
    avg_stars = summary_df['stars_mean'].mean()
    positive_rate = summary_df['pos'].mean()
    
    # 토픽 데이터 준비
    df = summary_df[summary_df['topic_id'] != -1].copy()
    df['display_label'] = df['label'].apply(lambda x: get_smart_display_label(x, 100))
    df['label_count'] = df['label'].apply(lambda x: len(parse_multiple_labels(x)))
    
    # 1. 각 토픽별 실제 리뷰 예시들
    topic_examples = {}
    for _, topic in df.nlargest(8, 'n').iterrows():  # 상위 8개 토픽
        topic_id = topic['topic_id']
        topic_sentences = sentences_df[sentences_df['topic_id'] == topic_id]
        
        # 긍정/부정 예시 각각 수집
        positive_examples = topic_sentences[topic_sentences['sentiment'] == 1]['sentence'].head(3).tolist()
        negative_examples = topic_sentences[topic_sentences['sentiment'] == 0]['sentence'].head(3).tolist()
        
        topic_examples[topic_id] = {
            'label': topic['display_label'],
            'raw_label': topic['label'],
            'positive_examples': positive_examples,
            'negative_examples': negative_examples,
            'metrics': {
                'volume': topic['n'],
                'positive_rate': topic['pos'],
                'avg_stars': topic['stars_mean'],
                'share': topic.get('share', 0)
            }
        }
    
    # 2. 시간대별 트렌드 분석
    trend_info = analyze_temporal_trends(sentences_df, df)
    
    # 3. 비즈니스별 비교 분석
    business_comparison = analyze_business_differences(sentences_df, df)
    
    # 4. 키워드 패턴 분석
    keyword_patterns = analyze_keyword_patterns(sentences_df, df)
    
    # 5. 감정 세밀도 분석
    sentiment_breakdown = analyze_detailed_sentiment(sentences_df, df)
    
    # 6. 토픽 간 연관성 분석
    topic_relationships = analyze_topic_relationships(df, sentences_df)
    
    # Enhanced prompt 구성
    enhanced_prompt = f"""
You are analyzing customer feedback using advanced zero-shot multi-label topic modeling. 

=== CRITICAL UNDERSTANDING GUIDELINES ===

**IMPORTANT - ZERO-SHOT MULTI-LABEL INTERPRETATION:**
- Each topic represents a HOLISTIC customer experience area with multiple interconnected aspects
- Don't treat the zero-shot labels as separate issues - they are facets of the same unified experience
- The descriptions in parentheses (e.g., "inaccurate", "slow") are EXAMPLES for context, not necessarily negative issues
- ALWAYS consider the actual performance scores (positive rate, stars) rather than just label text
- A topic labeled "order accuracy (missing items)" with 85% positive rate means customers are SATISFIED with order accuracy
- Focus on how improving one aspect creates positive ripple effects across related aspects within the same topic
- Use specific numbers and percentages from the data to support recommendations
- Provide practical, implementable recommendations for restaurant/business managers
- Emphasize the systemic nature of customer experience improvements

**ANALYSIS FRAMEWORK:**
- High volume + Low satisfaction = Immediate attention needed
- High volume + High satisfaction = Competitive advantage to leverage  
- Multiple aspects in one topic = Opportunity for holistic improvement
- Cross-topic patterns = Systemic operational insights

=== BUSINESS OVERVIEW ===
- Total Reviews Analyzed: {total_reviews:,}
- Average Rating: {avg_stars:.1f}/5.0 stars
- Overall Positive Sentiment: {positive_rate:.1%}
- Multi-Aspect Topics Identified: {len(df)}

=== DETAILED TOPIC ANALYSIS WITH REAL EXAMPLES ===
"""
    
    # 각 토픽별 상세 정보 추가
    for topic_id, info in topic_examples.items():
        performance_indicator = get_performance_indicator(info['metrics']['positive_rate'])
        
        enhanced_prompt += f"""

** {info['label']} ** {performance_indicator}
Volume: {info['metrics']['volume']} reviews ({info['metrics']['share']:.1%} of total) | Positive Rate: {info['metrics']['positive_rate']:.1%} | Avg Stars: {info['metrics']['avg_stars']:.1f}

INTERPRETATION GUIDANCE: {format_aspect_breakdown(info['raw_label'], info['metrics']['positive_rate'])}

POSITIVE Customer Voices:
{format_examples(info['positive_examples'])}

NEGATIVE Customer Voices:
{format_examples(info['negative_examples'])}
---
"""
    
    # 추가 컨텍스트들
    if trend_info:
        enhanced_prompt += f"\n=== TEMPORAL TRENDS ===\n{trend_info}\n"
    
    if business_comparison:
        enhanced_prompt += f"\n=== BUSINESS COMPARISON ===\n{business_comparison}\n"
    
    enhanced_prompt += f"\n=== KEYWORD PATTERNS ===\n{keyword_patterns}\n"
    enhanced_prompt += f"\n=== SENTIMENT BREAKDOWN ===\n{sentiment_breakdown}\n"
    enhanced_prompt += f"\n=== TOPIC RELATIONSHIPS ===\n{topic_relationships}\n"
    
    # 분석 요청
    enhanced_prompt += """

=== ANALYSIS REQUEST ===
Based on the holistic multi-aspect understanding above, provide:

1. **Executive Summary** (3-4 key findings emphasizing the multi-aspect nature and ACTUAL performance)

2. **Performance Reality Check**
   - Topics that LOOK concerning but actually perform well (high scores despite concerning labels)
   - Topics that LOOK fine but need attention (low scores despite neutral labels)
   - Use specific customer quotes as evidence

3. **Systemic Experience Insights** 
   - How do the multi-aspect topics interconnect?
   - Which single improvements would create the biggest ripple effects across multiple topics?
   - Cross-topic improvement opportunities

4. **Evidence-Based Action Plan**
   - Ranked by actual impact using the performance numbers
   - Specific customer quotes supporting each recommendation
   - Multi-aspect improvements over single-issue fixes
   - Immediate wins vs long-term strategies

5. **Competitive Positioning Insights**
   - Unique strengths revealed in the data (high-performing multi-aspect areas)
   - How to leverage interconnected positive aspects for marketing

6. **Customer Experience Journey Insights**
   - How different topics connect in the customer journey
   - Critical moments that impact multiple aspects simultaneously

REMEMBER: 
- A topic about "wait time (long delays)" with 80% positive rate means wait times are GOOD, not bad!
- Focus on the performance metrics and customer voices, not just the label text
- Look for systemic patterns across interconnected aspects
- Provide specific, implementable recommendations with supporting evidence
"""
    
    # OpenAI API 호출
    try:
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert business intelligence analyst specializing in customer experience and restaurant operations. You have deep understanding of zero-shot multi-label topic classification and how it reveals interconnected aspects of customer experience. You always base recommendations on actual performance data, not label assumptions."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.3,
            max_tokens=3000  # 더 긴 응답을 위해 증가
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def format_examples(examples):
    """Format example sentences nicely"""
    if not examples:
        return "- (No examples available)"
    
    formatted = []
    for i, example in enumerate(examples, 1):
        # 문장 길이 조절
        example = str(example).strip()
        if len(example) > 150:
            example = example[:147] + "..."
        formatted.append(f"- \"{example}\"")
    
    return "\n".join(formatted)

def get_performance_indicator(positive_rate):
    """성과 지표 이모지 반환"""
    if positive_rate > 0.75:
        return "🟢 STRENGTH"
    elif positive_rate > 0.6:
        return "🟡 GOOD"
    elif positive_rate > 0.4:
        return "🟠 MIXED"
    else:
        return "🔴 CONCERN"

def format_aspect_breakdown(raw_label, positive_rate):
    """각 라벨의 측면들과 성과 해석 가이드"""
    aspects = parse_multiple_labels(raw_label)
    
    interpretation = f"This topic covers {len(aspects)} interconnected aspects. With {positive_rate:.1%} positive rate, this indicates "
    if positive_rate > 0.7:
        interpretation += "STRONG performance across all these connected aspects:"
    elif positive_rate > 0.4:
        interpretation += "MIXED performance - some aspects working well, others need attention:"
    else:
        interpretation += "SIGNIFICANT opportunities for improvement across these connected areas:"
    
    breakdown = f"{interpretation}\n"
    for i, aspect in enumerate(aspects, 1):
        # 괄호 안 예시와 실제 성과를 구분해서 설명
        main_concept = aspect.split('(')[0].strip()
        example_part = f"({aspect.split('(', 1)[1]}" if '(' in aspect else ""
        
        breakdown += f"  • **{main_concept}** {example_part}\n"
    
    return breakdown

def analyze_temporal_trends(sentences_df, summary_df):
    """시간대별 트렌드 분석"""
    if 'date' not in sentences_df.columns:
        return "Date information not available for trend analysis."
    
    try:
        sentences_df_temp = sentences_df.copy()
        sentences_df_temp['date'] = pd.to_datetime(sentences_df_temp['date'], errors='coerce')
        sentences_df_temp = sentences_df_temp.dropna(subset=['date'])
        
        if len(sentences_df_temp) < 50:
            return "Insufficient temporal data for trend analysis."
        
        # 최근 50% vs 이전 50% 비교
        recent_cutoff = sentences_df_temp['date'].quantile(0.5)
        
        recent_data = sentences_df_temp[sentences_df_temp['date'] >= recent_cutoff]
        older_data = sentences_df_temp[sentences_df_temp['date'] < recent_cutoff]
        
        recent_sentiment = recent_data['sentiment'].mean()
        older_sentiment = older_data['sentiment'].mean()
        
        trend_direction = "IMPROVING" if recent_sentiment > older_sentiment else "DECLINING" if recent_sentiment < older_sentiment else "STABLE"
        change = abs(recent_sentiment - older_sentiment)
        
        # 토픽별 트렌드
        topic_trends = []
        for topic_id in summary_df['topic_id'].head(5):  # 상위 5개 토픽
            recent_topic = recent_data[recent_data['topic_id'] == topic_id]['sentiment'].mean()
            older_topic = older_data[older_data['topic_id'] == topic_id]['sentiment'].mean()
            
            if pd.notna(recent_topic) and pd.notna(older_topic):
                topic_change = recent_topic - older_topic
                topic_name = summary_df[summary_df['topic_id'] == topic_id]['display_label'].iloc[0] if len(summary_df[summary_df['topic_id'] == topic_id]) > 0 else f"Topic {topic_id}"
                topic_trends.append(f"  • {topic_name}: {topic_change:+.1%}")
        
        return f"""
Recent Period Sentiment: {recent_sentiment:.1%}
Earlier Period Sentiment: {older_sentiment:.1%}
Overall Trend: {trend_direction} ({change:.1%} change)

Key Topic Trends:
{chr(10).join(topic_trends) if topic_trends else "Insufficient data for topic-level trends"}

Time Range: {sentences_df_temp['date'].min().strftime('%Y-%m-%d')} to {sentences_df_temp['date'].max().strftime('%Y-%m-%d')}
"""
    except Exception as e:
        return f"Trend analysis error: {str(e)}"

def analyze_business_differences(sentences_df, summary_df):
    """여러 비즈니스 간 비교 분석"""
    if 'business_id' not in sentences_df.columns or sentences_df['business_id'].nunique() <= 1:
        return "Single business analysis - no comparison data available."
    
    try:
        business_stats = sentences_df.groupby('business_id').agg({
            'sentiment': 'mean',
            'stars': 'mean',
            'sentence': 'count'
        }).round(3)
        business_stats.columns = ['sentiment_rate', 'avg_stars', 'review_count']
        
        # 최소 10개 리뷰 이상인 비즈니스만
        business_stats = business_stats[business_stats['review_count'] >= 10]
        
        if len(business_stats) < 2:
            return "Insufficient data for business comparison."
        
        best_performer = business_stats['sentiment_rate'].idxmax()
        worst_performer = business_stats['sentiment_rate'].idxmin()
        
        # 상위/하위 3개 비즈니스
        top_businesses = business_stats.nlargest(3, 'sentiment_rate')
        bottom_businesses = business_stats.nsmallest(3, 'sentiment_rate')
        
        comparison_text = f"""
Best Performing Location: {best_performer} 
  - Positive Rate: {business_stats.loc[best_performer, 'sentiment_rate']:.1%}
  - Avg Stars: {business_stats.loc[best_performer, 'avg_stars']:.1f}
  - Reviews: {business_stats.loc[best_performer, 'review_count']}

Needs Most Attention: {worst_performer}
  - Positive Rate: {business_stats.loc[worst_performer, 'sentiment_rate']:.1%}
  - Avg Stars: {business_stats.loc[worst_performer, 'avg_stars']:.1f}
  - Reviews: {business_stats.loc[worst_performer, 'review_count']}

Performance Range: {business_stats['sentiment_rate'].min():.1%} to {business_stats['sentiment_rate'].max():.1%}
Standard Deviation: {business_stats['sentiment_rate'].std():.1%}

Top 3 Performers:
{format_business_list(top_businesses)}

Bottom 3 Performers:
{format_business_list(bottom_businesses)}
"""
        return comparison_text
    except Exception as e:
        return f"Business comparison error: {str(e)}"

def format_business_list(business_df):
    """비즈니스 목록 포맷팅"""
    result = []
    for idx, row in business_df.iterrows():
        result.append(f"  • {idx}: {row['sentiment_rate']:.1%} positive, {row['avg_stars']:.1f} stars ({row['review_count']} reviews)")
    return "\n".join(result)

def analyze_keyword_patterns(sentences_df, summary_df):
    """키워드 패턴 분석"""
    try:
        # 긍정/부정 리뷰 분리
        positive_reviews = sentences_df[sentences_df['sentiment'] == 1]['sentence'].fillna("").astype(str)
        negative_reviews = sentences_df[sentences_df['sentiment'] == 0]['sentence'].fillna("").astype(str)
        
        # 간단한 키워드 추출 (빈도 기반)
        def extract_keywords(texts, top_n=5):
            all_text = " ".join(texts).lower()
            # 간단한 단어 추출 (실제로는 더 정교한 NLP 필요)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
            # 불용어 제거
            stop_words = {'the', 'and', 'was', 'were', 'are', 'been', 'have', 'has', 'had', 'this', 'that', 'with', 'for', 'they', 'but', 'not', 'you', 'all', 'can', 'her', 'him', 'his', 'she', 'our', 'out', 'one', 'get', 'use', 'man', 'new', 'now', 'way', 'may', 'say'}
            words = [w for w in words if w not in stop_words and len(w) > 3]
            
            from collections import Counter
            return [word for word, count in Counter(words).most_common(top_n)]
        
        positive_keywords = extract_keywords(positive_reviews.tolist())
        negative_keywords = extract_keywords(negative_reviews.tolist())
        
        # 최근 vs 과거 키워드 (날짜가 있는 경우)
        emerging_text = ""
        if 'date' in sentences_df.columns:
            try:
                sentences_temp = sentences_df.copy()
                sentences_temp['date'] = pd.to_datetime(sentences_temp['date'], errors='coerce')
                recent_cutoff = sentences_temp['date'].quantile(0.7)  # 최근 30%
                recent_reviews = sentences_temp[sentences_temp['date'] >= recent_cutoff]['sentence'].fillna("").astype(str)
                recent_keywords = extract_keywords(recent_reviews.tolist(), 3)
                emerging_text = f"\nEmerging Keywords (recent): {', '.join(recent_keywords)}"
            except:
                pass
        
        return f"""
Most Mentioned in POSITIVE reviews: {', '.join(positive_keywords)}
Most Mentioned in NEGATIVE reviews: {', '.join(negative_keywords)}{emerging_text}

Keyword Distribution:
- Positive sentiment: {len(positive_reviews)} reviews
- Negative sentiment: {len(negative_reviews)} reviews
- Positive/Negative ratio: {len(positive_reviews)/(len(negative_reviews)+1):.1f}:1
"""
    except Exception as e:
        return f"Keyword analysis: Basic sentiment distribution available. {len(sentences_df)} total sentences analyzed."

def analyze_detailed_sentiment(sentences_df, summary_df):
    """감정 세밀도 분석"""
    try:
        # 감정 분포
        sentiment_dist = sentences_df['sentiment'].value_counts(normalize=True)
        
        # 별점별 감정 분포 (별점이 있는 경우)
        star_sentiment_text = ""
        if 'stars' in sentences_df.columns:
            star_sentiment = sentences_df.groupby('stars')['sentiment'].agg(['mean', 'count']).round(3)
            star_analysis = []
            for star, row in star_sentiment.iterrows():
                star_analysis.append(f"  {star}★: {row['mean']:.1%} positive ({row['count']} reviews)")
            star_sentiment_text = f"\nStar Rating vs Sentiment Alignment:\n" + "\n".join(star_analysis)
        
        # 감정-별점 불일치 분석
        mismatch_text = ""
        if 'stars' in sentences_df.columns:
            # 높은 별점이지만 부정적 감정
            high_star_negative = sentences_df[(sentences_df['stars'] >= 4) & (sentences_df['sentiment'] == 0)]
            # 낮은 별점이지만 긍정적 감정  
            low_star_positive = sentences_df[(sentences_df['stars'] <= 2) & (sentences_df['sentiment'] == 1)]
            
            mismatch_text = f"\nSentiment-Rating Mismatches:\n"
            mismatch_text += f"  High stars (4-5) but negative sentiment: {len(high_star_negative)} cases\n"
            mismatch_text += f"  Low stars (1-2) but positive sentiment: {len(low_star_positive)} cases"
        
        # 신뢰도 분석 (confidence가 있는 경우)
        confidence_text = ""
        if 'sentiment_conf' in sentences_df.columns:
            avg_conf = sentences_df['sentiment_conf'].mean()
            low_conf_count = len(sentences_df[sentences_df['sentiment_conf'] < 0.7])
            confidence_text = f"\nSentiment Confidence:\n"
            confidence_text += f"  Average confidence: {avg_conf:.1%}\n"
            confidence_text += f"  Low confidence predictions: {low_conf_count} ({low_conf_count/len(sentences_df):.1%})"
        
        return f"""
Sentiment Distribution:
- Positive: {sentiment_dist.get(1, 0):.1%} ({sentiment_dist.get(1, 0) * len(sentences_df):.0f} reviews)
- Negative: {sentiment_dist.get(0, 0):.1%} ({sentiment_dist.get(0, 0) * len(sentences_df):.0f} reviews){star_sentiment_text}{mismatch_text}{confidence_text}

Total Reviews Analyzed: {len(sentences_df)}
"""
    except Exception as e:
        return f"Sentiment analysis: {len(sentences_df)} reviews analyzed with basic distribution available."

def analyze_topic_relationships(summary_df, sentences_df):
    """토픽 간 연관성 분석"""
    try:
        df = summary_df[summary_df['topic_id'] != -1].copy()
        
        # 볼륨과 성과의 관계
        volume_performance = []
        for _, topic in df.iterrows():
            category = "High Volume" if topic['n'] > df['n'].median() else "Low Volume"
            performance = "High Performance" if topic['pos'] > 0.7 else "Medium Performance" if topic['pos'] > 0.4 else "Low Performance"
            volume_performance.append(f"  • {category} + {performance}: {topic['display_label'][:50]}...")
        
        # 복잡성 분석 (멀티라벨 개수)
        df['label_count'] = df['label'].apply(lambda x: len(parse_multiple_labels(x)))
        complexity_analysis = []
        high_complexity = df[df['label_count'] > 3].nlargest(3, 'pos')
        for _, topic in high_complexity.iterrows():
            complexity_analysis.append(f"  • {topic['label_count']} aspects, {topic['pos']:.1%} positive: {topic['display_label'][:50]}...")
        
        # 성과 클러스터링
        excellent_topics = len(df[df['pos'] > 0.8])
        good_topics = len(df[(df['pos'] > 0.6) & (df['pos'] <= 0.8)])
        concerning_topics = len(df[df['pos'] <= 0.4])
        
        return f"""
Volume vs Performance Patterns:
{chr(10).join(volume_performance[:6])}

High-Complexity Multi-Aspect Topics (3+ aspects):
{chr(10).join(complexity_analysis) if complexity_analysis else "No highly complex topics found"}

Performance Clustering:
- Excellent (>80% positive): {excellent_topics} topics
- Good (60-80% positive): {good_topics} topics  
- Concerning (<40% positive): {concerning_topics} topics

Strategic Insight: {"High complexity topics performing well indicate strong systemic operations" if len(high_complexity) > 0 and high_complexity['pos'].mean() > 0.7 else "Focus on simplifying operations for better performance"}
"""
    except Exception as e:
        return f"Topic relationship analysis: {len(summary_df)} topics analyzed with basic categorization available."

def generate_executive_summary(summary_df: pd.DataFrame, sentences_df: pd.DataFrame) -> str:
    """Generate executive summary with full multi-label insights"""
    
    total_reviews = sentences_df['review_id'].nunique() if 'review_id' in sentences_df.columns else len(sentences_df)
    avg_stars = summary_df['stars_mean'].mean()
    positive_rate = summary_df['pos'].mean()
    
    # Find key issues and strengths - NO primary label selection
    df = summary_df[summary_df['topic_id'] != -1].copy()
    df['display_label'] = df['label'].apply(lambda x: get_smart_display_label(x, 80))
    df['label_count'] = df['label'].apply(lambda x: len(parse_multiple_labels(x)))
    
    worst_topics = df.nsmallest(3, 'pos')
    best_topics = df.nlargest(3, 'pos')
    high_volume_topics = df.nlargest(5, 'n')
    
    summary = f"""
    ## Executive Summary
    
    **Overall Performance:**
    - Analyzed {total_reviews:,} customer reviews across {len(df)} multi-aspect topics
    - Average satisfaction: {avg_stars:.1f}/5.0 stars
    - Positive sentiment rate: {positive_rate:.1%}
    
    **Key Strengths (Multi-Aspect Topics):**
    """
    
    for _, topic in best_topics.iterrows():
        complexity_note = f" ({topic['label_count']} aspects)" if topic['label_count'] > 1 else ""
        summary += f"\n- **{topic['display_label']}**{complexity_note}: {topic['pos']:.1%} positive, {topic['stars_mean']:.1f} stars"
    
    summary += "\n\n**Critical Multi-Aspect Issues Requiring Attention:**"
    
    for _, topic in worst_topics.iterrows():
        complexity_note = f" ({topic['label_count']} aspects)" if topic['label_count'] > 1 else ""
        summary += f"\n- **{topic['display_label']}**{complexity_note}: Only {topic['pos']:.1%} positive, {topic['stars_mean']:.1f} stars"
    
    summary += "\n\n**Most Discussed Multi-Aspect Topics:**"
    
    for _, topic in high_volume_topics.iterrows():
        complexity_note = f" ({topic['label_count']} aspects)" if topic['label_count'] > 1 else ""
        summary += f"\n- **{topic['display_label']}**{complexity_note}: {topic['n']} reviews ({topic['share']:.1%} of total)"
    
    return summary

def create_llm_insights_section(summary_df: pd.DataFrame, sentences_df: pd.DataFrame) -> None:
    """Create LLM-powered insights section"""
    
    st.markdown('<h2 class="section-header">🤖 AI-Powered Business Insights</h2>', unsafe_allow_html=True)
    
    # Check API key
    API_KEY = os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
    
    if not API_KEY:
        st.warning("🔑 OpenAI API key not configured. Please set OPENAI_API_KEY environment variable or in Streamlit secrets to enable AI insights.")
        with st.expander("How to configure OpenAI API key"):
            st.code("""
            # Method 1: Environment variable
            export OPENAI_API_KEY="your-api-key-here"
            
            # Method 2: Streamlit secrets
            # Create .streamlit/secrets.toml:
            OPENAI_API_KEY = "your-api-key-here"
            """)
        return
    
    # Generate insights button
    if st.button("🚀 Generate AI Insights", type="primary"):
        with st.spinner("🧠 AI is analyzing your customer feedback data..."):
            insights = generate_enhanced_llm_insights(summary_df, sentences_df)
            
            if insights and not insights.startswith("Error"):
                st.session_state['llm_insights'] = insights
                st.success("✅ AI insights generated successfully!")
            else:
                st.error(f"❌ Failed to generate insights: {insights}")
    
    # Display cached insights
    if 'llm_insights' in st.session_state:
        st.markdown("### 📋 AI Analysis Report")
        
        # Create download button
        insights_text = st.session_state['llm_insights']
        st.download_button(
            label="📥 Download Insights Report",
            data=insights_text,
            file_name=f"ai_insights_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )
        
        # Display insights in a nice format
        with st.container():
            st.markdown(insights_text)
    
    # Optional: Real-time insights toggle
    with st.expander("⚙️ Advanced AI Settings"):
        auto_insights = st.checkbox("🔄 Auto-generate insights when data loads", value=False)
        model_choice = st.selectbox("🤖 AI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
        temperature = st.slider("🌡️ Creativity Level", 0.0, 1.0, 0.3, 0.1)
        
        if auto_insights and st.session_state.get('data_loaded') and 'llm_insights' not in st.session_state:
            st.info("🔄 Auto-generating insights...")
            insights = generate_enhanced_llm_insights(summary_df, sentences_df)
            if insights and not insights.startswith("Error"):
                st.session_state['llm_insights'] = insights
                st.rerun()

# ========= Enhanced Streamlit App =========

st.set_page_config(page_title="Business Intelligence Dashboard", page_icon="📊", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎯 Business Intelligence Dashboard</h1>', unsafe_allow_html=True)

# Enhanced sidebar for pipeline and data loading
st.sidebar.header("📂 Data Source")
data_source = st.sidebar.selectbox(
    "Choose Data Source", 
    ["Load from existing outputs", "Run new pipeline"], 
    index=0
)

if data_source == "Run new pipeline":
    st.sidebar.subheader("🔍 Business Selection")
    
    # Business filtering options
    col1, col2 = st.sidebar.columns(2)
    with col1:
        name_substr = st.text_input("Name contains", value="", placeholder="e.g., Pizza")
        city = st.text_input("City", value="", placeholder="e.g., Las Vegas")
    with col2:
        category = st.text_input("Category", value="Restaurants", placeholder="e.g., Restaurants")
        state = st.text_input("State", value="", placeholder="e.g., NV")
    
    limit = st.sidebar.number_input("Max businesses to analyze", min_value=1, max_value=100, value=20, step=1)
    
    # Manual business ID input
    biz_ids_manual = st.sidebar.text_area(
        "Or specify Business IDs directly (one per line)", 
        value="", 
        placeholder="business_id_1\nbusiness_id_2\n...",
        help="Leave empty to use search filters above"
    )
    
    st.sidebar.subheader("🎯 Topic Analysis Settings")
    
    topic_scope = st.sidebar.selectbox(
        "Analysis Scope", 
        ["per-store", "pooled", "both"], 
        index=0,
        help="per-store: Individual analysis per business | pooled: Combined analysis | both: Both approaches"
    )
    
    # Threshold settings
    col1, col2 = st.sidebar.columns(2)
    with col1:
        per_store_min_sentences = st.number_input("Min sentences per store", min_value=20, max_value=500, value=80, step=10)
    with col2:
        per_store_min_reviews = st.number_input("Min reviews per store (0=ignore)", min_value=0, max_value=100, value=0, step=5)
    
    # Topic model parameters
    with st.sidebar.expander("🔧 Advanced Topic Settings"):
        min_topic_size = st.number_input("Min topic size (0=auto)", min_value=0, max_value=50, value=0, step=1)
        nr_topics = st.number_input("Max topics (0=unlimited)", min_value=0, max_value=100, value=0, step=5)
        
        # Zero-shot labeling
        enable_zeroshot = st.checkbox("Enable zero-shot topic labeling", value=True)
        if enable_zeroshot:
            zeroshot_min_prob = st.slider("Zero-shot confidence threshold", 0.1, 0.9, 0.4, 0.05)
            zeroshot_online = st.checkbox("Allow online model download", value=False)
            label_mode = st.selectbox("Label mode", ["core", "auto", "full"], index=1)
    
    # Output settings
    with st.sidebar.expander("📁 Output Settings"):
        out_dir = st.text_input("Output directory", value="./outputs")
        filename_template = st.text_input(
            "Filename template", 
            value="with_topics__{scope}__{addr_slug}.csv",
            help="Available placeholders: {scope}, {addr_slug}"
        )
        write_index = st.checkbox("Generate run index file", value=True)
    
    # Run pipeline button
    if st.sidebar.button("🚀 Run Analysis Pipeline", type="primary"):
        with st.status("Running analysis pipeline...", expanded=True) as status:
            try:
                # Prepare business IDs
                if biz_ids_manual.strip():
                    biz_ids = [line.strip() for line in biz_ids_manual.strip().splitlines() if line.strip()]
                    st.write(f"Using manually specified business IDs: {len(biz_ids)} businesses")
                else:
                    # Use search filters
                    biz_ids = find_business_ids(
                        business_json_path=str(PATH_BUSINESS),
                        name_substring=name_substr if name_substr else None,
                        category_keyword=category if category else "Restaurants",
                        city=city if city else None,
                        state=state if state else None,
                        limit=int(limit)
                    )
                    st.write(f"Found {len(biz_ids)} businesses matching search criteria")
                
                if not biz_ids:
                    st.error("❌ No businesses found matching your criteria. Please adjust your filters.")
                    st.stop()
                
                # Load business metadata
                id2meta = load_business_meta(str(PATH_BUSINESS), biz_ids)
                
                # Show selected businesses
                st.write("**Selected Businesses:**")
                for bid in biz_ids[:10]:  # Show first 10
                    meta = id2meta.get(bid, {}) or {}
                    name = meta.get("name", "Unknown")
                    address = meta.get("address", "Unknown")
                    st.write(f"• {name} - {address}")
                if len(biz_ids) > 10:
                    st.write(f"... and {len(biz_ids) - 10} more")
                
                # Step 1: Filter reviews
                st.write("📝 Filtering reviews...")
                kept = filter_reviews_by_business_ids(
                    review_json_path=str(PATH_REVIEW),
                    target_business_ids=biz_ids,
                    out_jsonl_path=str(PATH_REVIEWS_FILTERED),
                )
                st.write(f"✅ Filtered {kept:,} reviews")
                
                # Step 2: Sentiment analysis
                st.write("🎭 Analyzing sentiment...")
                run_sentence_sentiment(
                    filtered_jsonl_path=str(PATH_REVIEWS_FILTERED),
                    out_csv_path=str(PATH_SENT_WITH_SENT),
                    sentiment_model=SENTIMENT_MODEL,
                    max_chars=256,
                    batch_size=64
                )
                st.write("✅ Sentiment analysis complete")
                
                # Step 3: Topic modeling setup
                st.write("📊 Preparing topic analysis...")
                sent_df = pd.read_csv(PATH_SENT_WITH_SENT)
                sent_counts = sent_df.groupby("business_id").size().to_dict()
                
                # Review count filtering if needed
                review_counts = None
                if per_store_min_reviews > 0 and "review_id" in sent_df.columns:
                    review_counts = (
                        sent_df.drop_duplicates(["business_id", "review_id"])
                              .groupby("business_id").size().to_dict()
                    )
                
                # Output directory setup
                out_path = Path(out_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                
                # Address slug generation (with collision handling)
                def slugify(s: str, max_len: int = 60) -> str:
                    s = (s or "").lower()
                    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
                    if len(s) > max_len:
                        s = s[:max_len].rstrip("-")
                    return s or "store"
                
                seen_addr = {}
                def addr_slug_of(bid: str) -> str:
                    meta = id2meta.get(bid, {}) or {}
                    base = meta.get("address") or meta.get("name") or "store"
                    s = slugify(base, max_len=60)
                    cnt = seen_addr.get(s, 0) + 1
                    seen_addr[s] = cnt
                    return s if cnt == 1 else f"{s}-{cnt}"
                
                # Topic modeling runner function
                def run_topic_analysis(scope: str, business_id, out_base_path: Path):
                    apply_bertopic_for_business(
                        sentences_csv_path=str(PATH_SENT_WITH_SENT),
                        out_csv_path=str(out_base_path),
                        business_id=business_id,
                        embedding_model=EMBEDDING_MODEL,
                        min_topic_size=(None if min_topic_size == 0 else min_topic_size),
                        nr_topics=(None if nr_topics == 0 else nr_topics),
                        enable_zeroshot=enable_zeroshot,
                        zeroshot_online=zeroshot_online if enable_zeroshot else False,
                        zeroshot_min_prob=zeroshot_min_prob if enable_zeroshot else 0.4,
                        label_mode=label_mode if enable_zeroshot else "auto",
                    )
                    return str(out_base_path)
                
                run_index = []
                
                # Step 4: Topic modeling
                if topic_scope in ("pooled", "both"):
                    st.write("🔄 Running pooled topic analysis...")
                    pooled_name = filename_template.format(scope="pooled", addr_slug="all-stores")
                    pooled_path = out_path / pooled_name
                    
                    try:
                        base = run_topic_analysis("pooled", None, pooled_path)
                        
                        # Verify files were created
                        expected_files = [
                            pooled_path,
                            Path(str(pooled_path).replace('.csv', '_topic_summary_with_labels.csv'))
                        ]
                        
                        for expected_file in expected_files:
                            if expected_file.exists():
                                st.write(f"  ✅ Created: {expected_file.name}")
                            else:
                                st.write(f"  ❌ Missing: {expected_file.name}")
                        
                        run_index.append({
                            "scope": "pooled",
                            "business_id": "ALL",
                            "business_address": "ALL",
                            "n_sentences": len(sent_df),
                            "n_reviews": sent_df["review_id"].nunique() if "review_id" in sent_df.columns else None,
                            "status": "modeled",
                            "output_base": base
                        })
                        st.write("✅ Pooled analysis complete")
                        
                    except Exception as e:
                        st.error(f"❌ Pooled analysis failed: {str(e)}")
                        st.exception(e)
                
                if topic_scope in ("per-store", "both"):
                    st.write("🔄 Running per-store topic analysis...")
                    modeled_count = 0
                    skipped_count = 0
                    
                    for i, bid in enumerate(biz_ids):
                        n_s = sent_counts.get(bid, 0)
                        n_r = review_counts.get(bid, 0) if review_counts else None
                        
                        progress = (i + 1) / len(biz_ids)
                        st.progress(progress, text=f"Processing store {i+1}/{len(biz_ids)}: {bid}")
                        
                        # Check thresholds
                        if n_s < per_store_min_sentences:
                            st.write(f"⏭️ Skipping {bid}: only {n_s} sentences (< {per_store_min_sentences})")
                            run_index.append({
                                "scope": "per-store", "business_id": bid,
                                "business_address": (id2meta.get(bid, {}) or {}).get("address", ""),
                                "n_sentences": n_s, "n_reviews": n_r,
                                "status": "skipped_low_sentences", "output_base": None
                            })
                            skipped_count += 1
                            continue
                        
                        if review_counts and n_r < per_store_min_reviews:
                            st.write(f"⏭️ Skipping {bid}: only {n_r} reviews (< {per_store_min_reviews})")
                            run_index.append({
                                "scope": "per-store", "business_id": bid,
                                "business_address": (id2meta.get(bid, {}) or {}).get("address", ""),
                                "n_sentences": n_s, "n_reviews": n_r,
                                "status": "skipped_low_reviews", "output_base": None
                            })
                            skipped_count += 1
                            continue
                        
                        # Run analysis for this store
                        try:
                            addr_slug = addr_slug_of(bid)
                            fname = filename_template.format(scope="perstore", addr_slug=addr_slug)
                            store_path = out_path / fname
                            
                            st.write(f"  🏪 Analyzing {bid} → {fname}")
                            base = run_topic_analysis("per-store", bid, store_path)
                            
                            # Verify files were created
                            expected_files = [
                                store_path,
                                Path(str(store_path).replace('.csv', '_topic_summary_with_labels.csv'))
                            ]
                            
                            created_files = 0
                            for expected_file in expected_files:
                                if expected_file.exists():
                                    created_files += 1
                                    st.write(f"    ✅ Created: {expected_file.name}")
                                else:
                                    st.write(f"    ❌ Missing: {expected_file.name}")
                            
                            if created_files > 0:
                                run_index.append({
                                    "scope": "per-store", "business_id": bid,
                                    "business_address": (id2meta.get(bid, {}) or {}).get("address", ""),
                                    "n_sentences": n_s, "n_reviews": n_r,
                                    "status": "modeled", "output_base": base
                                })
                                modeled_count += 1
                            else:
                                st.error(f"❌ Failed to create files for {bid}")
                                
                        except Exception as e:
                            st.error(f"❌ Analysis failed for {bid}: {str(e)}")
                            st.exception(e)
                            skipped_count += 1
                    
                    st.write(f"✅ Per-store analysis complete: {modeled_count} modeled, {skipped_count} skipped")
                
                # Step 5: Save run index and final verification
                if write_index and run_index:
                    idx_path = out_path / "topic_run_index.csv"
                    pd.DataFrame(run_index).to_csv(idx_path, index=False, encoding="utf-8")
                    st.write(f"📋 Run index saved: {idx_path}")
                
                # Final status and file verification
                st.write("📁 **Final Output Directory Scan:**")
                all_files = list(out_path.glob("*"))
                if all_files:
                    st.write(f"  📂 Total files created: {len(all_files)}")
                    
                    # Group files by type
                    csv_files = [f for f in all_files if f.suffix == '.csv']
                    summary_files = [f for f in csv_files if 'topic_summary_with_labels' in f.name]
                    sentence_files = [f for f in csv_files if f.name.startswith('with_topics_') and 'topic_summary' not in f.name]
                    
                    st.write(f"  📊 CSV files: {len(csv_files)}")
                    st.write(f"  📋 Summary files: {len(summary_files)}")
                    st.write(f"  📝 Sentence files: {len(sentence_files)}")
                    
                    # Show file sizes
                    with st.expander("📋 File Details"):
                        for f in sorted(all_files):
                            size = f.stat().st_size
                            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                            st.write(f"  • {f.name}: {size_str}")
                else:
                    st.error(f"❌ No files found in output directory: {out_path}")
                    st.write("🔍 **Troubleshooting Tips:**")
                    st.write("  • Check if the directory is writable")
                    st.write("  • Verify all pipeline steps completed without errors")
                    st.write("  • Look for error messages in the logs above")
                
                status.update(label="✅ Pipeline completed successfully!", state="complete")
                
                # Verify and list generated files
                st.write("📁 **Generated Files:**")
                summary_files = list(out_path.glob("*topic_summary_with_labels.csv"))
                sentence_files = list(out_path.glob("with_topics_*.csv"))
                sentence_files = [f for f in sentence_files if not any(suffix in f.name for suffix in 
                                ['_topic_summary', '_topic_trend', '_topic_examples', '_with_labels'])]
                
                if summary_files:
                    st.write("✅ **Summary files:**")
                    for f in summary_files:
                        st.write(f"  • {f.name}")
                else:
                    st.error("❌ No summary files generated!")
                
                if sentence_files:
                    st.write("✅ **Sentence files:**")
                    for f in sentence_files:
                        st.write(f"  • {f.name}")
                else:
                    st.error("❌ No sentence files generated!")
                
                # Auto-load results if files exist
                if summary_files and sentence_files:
                    st.write("🔄 **Auto-loading results...**")
                    
                    try:
                        # Load and combine files
                        summary_dfs = []
                        sentence_dfs = []
                        
                        for f in summary_files:
                            df = pd.read_csv(f)
                            if 'source_file' not in df.columns:
                                df['source_file'] = f.name
                            summary_dfs.append(df)
                        
                        for f in sentence_files:
                            df = pd.read_csv(f)
                            if 'source_file' not in df.columns:
                                df['source_file'] = f.name
                            sentence_dfs.append(df)
                        
                        summary_df = pd.concat(summary_dfs, ignore_index=True)
                        sentences_df = pd.concat(sentence_dfs, ignore_index=True)
                        
                        # Store in session state
                        st.session_state['summary_df'] = summary_df
                        st.session_state['sentences_df'] = sentences_df
                        st.session_state['data_loaded'] = True
                        st.session_state['load_info'] = {
                            'method': 'Pipeline auto-load',
                            'directory': str(out_path),
                            'summary_rows': len(summary_df),
                            'sentence_rows': len(sentences_df),
                            'topics_count': len(summary_df[summary_df['topic_id'] != -1]) if 'topic_id' in summary_df.columns else 0,
                            'businesses_count': sentences_df['business_id'].nunique() if 'business_id' in sentences_df.columns else 0
                        }
                        
                        st.balloons()
                        st.success(f"🎉 Pipeline complete! Auto-loaded {len(summary_df)} topics, {len(sentences_df)} sentences")
                        st.info("👇 Scroll down to see the analysis results!")
                        
                        # Add a clear navigation button
                        if st.button("🔽 Jump to Analysis Results", type="primary"):
                            st.rerun()
                        
                        # Force refresh to show results
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Auto-loading failed: {str(e)}")
                        st.exception(e)
                        
                        # Provide manual loading instructions
                        st.markdown("### 🔧 Manual Loading Instructions")
                        st.info(f"""
                        **Auto-loading failed, but your files were generated successfully!**
                        
                        **To view your results:**
                        1. Go to the "Load Existing Results" section in the sidebar
                        2. Set Results Directory to: `{out_path}`
                        3. Click "🔍 Discover Result Files"
                        4. Click "📊 Load Analysis Results"
                        """)
                        
                        st.session_state['scan_dir_auto'] = str(out_path)
                
                else:
                    st.warning("⚠️ Pipeline completed but some required files are missing.")
                    st.markdown("### 🔍 What to check:")
                    st.write("1. Look for error messages in the pipeline logs above")
                    st.write("2. Ensure your input data has sufficient content")
                    st.write("3. Check if the output directory is writable")
                    st.write(f"4. Manually inspect the output directory: `{out_path}`")
                    
                    # Still provide the option to try manual loading
                    st.info("💡 You can still try manual loading if some files were created")
                    st.session_state['scan_dir_auto'] = str(out_path)
                
            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
                st.exception(e)
else:
    # Data loading section
    st.sidebar.subheader("📁 Load Existing Results")
    
    # Auto-populate scan directory if pipeline just ran
    default_scan_dir = st.session_state.get('scan_dir_auto', './outputs')
    scan_dir = st.sidebar.text_input("Results Directory", value=default_scan_dir)
    
    # File discovery
    if st.sidebar.button("🔍 Discover Result Files"):
        scan_path = Path(scan_dir)
        if scan_path.exists():
            # Look for result files
            summary_files = list(scan_path.glob("*topic_summary_with_labels.csv"))
            sentence_files = list(scan_path.glob("with_topics_*.csv"))
            sentence_files = [f for f in sentence_files if not f.name.endswith('_topic_summary_with_labels.csv') 
                            and not f.name.endswith('_topic_summary.csv')
                            and not f.name.endswith('_topic_trend_by_month.csv')
                            and not f.name.endswith('_topic_examples.csv')
                            and not f.name.endswith('_with_labels.csv')]
            
            if summary_files or sentence_files:
                st.sidebar.success(f"✅ Found {len(summary_files)} summary files, {len(sentence_files)} sentence files")
                
                # Show available files
                with st.sidebar.expander("📋 Available Files"):
                    st.write("**Summary Files:**")
                    for f in summary_files[:5]:
                        st.write(f"• {f.name}")
                    if len(summary_files) > 5:
                        st.write(f"... and {len(summary_files) - 5} more")
                    
                    st.write("**Sentence Files:**")
                    for f in sentence_files[:5]:
                        st.write(f"• {f.name}")
                    if len(sentence_files) > 5:
                        st.write(f"... and {len(sentence_files) - 5} more")
            else:
                st.sidebar.warning("⚠️ No compatible result files found")
        else:
            st.sidebar.error("❌ Directory does not exist")
    
    # Load specific files or auto-detect
    load_method = st.sidebar.radio(
        "Loading Method",
        ["Auto-detect and combine", "Select specific files"],
        index=0
    )
    
    if load_method == "Select specific files":
        scan_path = Path(scan_dir)
        if scan_path.exists():
            summary_files = list(scan_path.glob("*topic_summary_with_labels.csv"))
            sentence_files = list(scan_path.glob("with_topics_*.csv"))
            sentence_files = [f for f in sentence_files if not any(suffix in f.name for suffix in 
                            ['_topic_summary', '_topic_trend', '_topic_examples', '_with_labels'])]
            
            selected_summary = st.sidebar.selectbox(
                "Summary file",
                [""] + [f.name for f in summary_files],
                index=0
            )
            selected_sentences = st.sidebar.selectbox(
                "Sentence file", 
                [""] + [f.name for f in sentence_files],
                index=0
            )
    
    if st.sidebar.button("📊 Load Analysis Results", type="primary"):
        try:
            scan_path = Path(scan_dir)
            
            if load_method == "Auto-detect and combine":
                # Auto-discover and combine all compatible files
                summary_files = list(scan_path.glob("*topic_summary_with_labels.csv"))
                sentence_files = list(scan_path.glob("with_topics_*.csv"))
                sentence_files = [f for f in sentence_files if not any(suffix in f.name for suffix in 
                                ['_topic_summary', '_topic_trend', '_topic_examples', '_with_labels'])]
                
                if not summary_files:
                    st.sidebar.error("❌ No topic summary files found")
                    st.stop()
                
                if not sentence_files:
                    st.sidebar.error("❌ No sentence files found")
                    st.stop()
                
                # Load and combine files
                summary_dfs = []
                sentence_dfs = []
                
                for f in summary_files:
                    df = pd.read_csv(f)
                    if 'source_file' not in df.columns:
                        df['source_file'] = f.name
                    summary_dfs.append(df)
                
                for f in sentence_files:
                    df = pd.read_csv(f)
                    if 'source_file' not in df.columns:
                        df['source_file'] = f.name
                    sentence_dfs.append(df)
                
                summary_df = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()
                sentences_df = pd.concat(sentence_dfs, ignore_index=True) if sentence_dfs else pd.DataFrame()
                
                st.sidebar.success(f"✅ Combined {len(summary_files)} summary + {len(sentence_files)} sentence files")
                
            else:
                # Load specific selected files
                if not selected_summary or not selected_sentences:
                    st.sidebar.error("❌ Please select both summary and sentence files")
                    st.stop()
                
                summary_df = pd.read_csv(scan_path / selected_summary)
                sentences_df = pd.read_csv(scan_path / selected_sentences)
                
                st.sidebar.success(f"✅ Loaded {selected_summary} + {selected_sentences}")
            
            # Store in session state
            st.session_state['summary_df'] = summary_df
            st.session_state['sentences_df'] = sentences_df
            st.session_state['data_loaded'] = True
            st.session_state['load_info'] = {
                'method': load_method,
                'directory': scan_dir,
                'summary_rows': len(summary_df),
                'sentence_rows': len(sentences_df),
                'topics_count': len(summary_df[summary_df['topic_id'] != -1]) if 'topic_id' in summary_df.columns else 0,
                'businesses_count': sentences_df['business_id'].nunique() if 'business_id' in sentences_df.columns else 0
            }
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")
            st.sidebar.exception(e)

# Main dashboard - ALWAYS render if data exists
if st.session_state.get('data_loaded', False):
    summary_df = st.session_state['summary_df']
    sentences_df = st.session_state['sentences_df']
    load_info = st.session_state.get('load_info', {})
    
    # Clear the pipeline success messages to focus on results
    st.markdown("---")
    st.markdown('<h1 style="color: #1f77b4; text-align: center;">📊 Analysis Results Dashboard</h1>', unsafe_allow_html=True)
    
    # Data loading summary
    if load_info:
        st.markdown(f"""
        <div style='padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center;'>
            <h3 style='margin: 0; color: white;'>📊 Analysis Complete!</h3>
            <p style='margin: 0.5rem 0; font-size: 1.1rem;'>
                <strong>{load_info.get('topics_count', 0)} topics</strong> found in 
                <strong>{load_info.get('sentence_rows', 0):,} sentences</strong> from 
                <strong>{load_info.get('businesses_count', 0)} businesses</strong>
            </p>
            <small>Method: {load_info.get('method', 'Unknown')} | Source: {load_info.get('directory', 'Unknown')}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Performance Indicators
    st.markdown('<h2 class="section-header">📈 Key Performance Indicators</h2>', unsafe_allow_html=True)
    create_kpi_cards(summary_df, sentences_df)
    
    # Executive Summary
    st.markdown('<h2 class="section-header">📋 Executive Summary</h2>', unsafe_allow_html=True)
    with st.expander("View Detailed Summary", expanded=True):
        summary_text = generate_executive_summary(summary_df, sentences_df)
        st.markdown(summary_text)
    
    # AI-Powered Insights Section
    create_llm_insights_section(summary_df, sentences_df)
    
    # Main visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">🎯 Topic Performance Analysis</h2>', unsafe_allow_html=True)
        create_topic_overview_chart(summary_df)
    
    with col2:
        st.markdown('<h2 class="section-header">⚡ Quick Stats</h2>', unsafe_allow_html=True)
        
        # Quick stats
        df_for_stats = summary_df[summary_df['topic_id'] != -1]
        total_topics = len(df_for_stats)
        problematic_topics = len(df_for_stats[(df_for_stats['pos'] < 0.4)])
        excellent_topics = len(df_for_stats[(df_for_stats['pos'] > 0.8)])
        
        st.metric("Total Topics", total_topics)
        st.metric("Problematic Topics", problematic_topics, delta=f"{problematic_topics/total_topics:.1%}" if total_topics > 0 else "0%")
        st.metric("Excellent Topics", excellent_topics, delta=f"{excellent_topics/total_topics:.1%}" if total_topics > 0 else "0%")
        
        # Top keywords
        st.subheader("🔍 Top Aspects")
        
        # Extract individual aspects from zero-shot labels
        all_aspects = []
        for _, row in df_for_stats.iterrows():
            aspects = parse_multiple_labels(row['label'])
            for aspect in aspects:
                # Extract main concept (text before parentheses)
                main_concept = aspect.split('(')[0].strip() if '(' in aspect else aspect
                all_aspects.append((main_concept, row['n']))  # Weight by review count
        
        if all_aspects:
            aspect_weights = {}
            for aspect, weight in all_aspects:
                aspect_weights[aspect] = aspect_weights.get(aspect, 0) + weight
            
            top_aspects = sorted(aspect_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            for aspect, weight in top_aspects:
                st.write(f"• **{aspect}** ({weight:,})")
        else:
            st.info("No aspects found")
    
    # Priority Matrix
    st.markdown('<h2 class="section-header">🎯 Action Priority Matrix</h2>', unsafe_allow_html=True)
    create_priority_matrix(summary_df)
    
    # Timeline Analysis (if date data available)
    st.markdown('<h2 class="section-header">📅 Sentiment Timeline</h2>', unsafe_allow_html=True)
    create_sentiment_timeline(sentences_df, summary_df)
    
    # Detailed Tables (collapsible)
    with st.expander("📊 Detailed Multi-Aspect Topic Analysis", expanded=False):
        st.subheader("Complete Zero-Shot Multi-Label Topic Summary")
        
        # Prepare enhanced summary table - show ALL aspects
        display_df = summary_df[summary_df['topic_id'] != -1].copy()
        display_df['display_label'] = display_df['label'].apply(lambda x: get_smart_display_label(x, 100))
        display_df['compact_label'] = display_df['label'].apply(lambda x: get_compact_display_label(x, 80))
        display_df['all_aspects_preview'] = display_df['label'].apply(
            lambda x: ' | '.join(parse_multiple_labels(x)[:3]) + (f' | +{len(parse_multiple_labels(x))-3} more' if len(parse_multiple_labels(x)) > 3 else '')
        )
        display_df['label_count'] = display_df['label'].apply(lambda x: len(parse_multiple_labels(x)))
        
        # Reorder columns for better display
        display_cols = ['topic_id', 'display_label', 'label_count', 'n', 'pos', 'stars_mean', 'share', 'all_aspects_preview']
        display_cols = [col for col in display_cols if col in display_df.columns]
        
        st.dataframe(
            display_df[display_cols].sort_values('n', ascending=False), 
            use_container_width=True,
            column_config={
                "topic_id": st.column_config.NumberColumn("Topic ID", width="small"),
                "display_label": st.column_config.TextColumn("Multi-Aspect Topic", width="large"),
                "label_count": st.column_config.NumberColumn("# Aspects", width="small"),
                "n": st.column_config.NumberColumn("Reviews", width="small"),
                "pos": st.column_config.NumberColumn("Positive Rate", format="%.1%", width="small"),
                "stars_mean": st.column_config.NumberColumn("Avg Stars", format="%.1f", width="small"),
                "share": st.column_config.NumberColumn("Share", format="%.1%", width="small"),
                "all_aspects_preview": st.column_config.TextColumn("Aspect Preview", width="large")
            },
            height=400
        )
        
        # Show full label details for selected topics
        st.subheader("Complete Zero-Shot Aspect Breakdown")
        selected_topics = st.multiselect(
            "Select topics to see all zero-shot aspects (no truncation):",
            options=display_df['topic_id'].tolist(),
            default=display_df.nlargest(3, 'n')['topic_id'].tolist(),
            format_func=lambda x: f"Topic {x}: {display_df[display_df['topic_id']==x]['compact_label'].iloc[0]}"
        )
        
        for topic_id in selected_topics:
            topic_row = display_df[display_df['topic_id'] == topic_id].iloc[0]
            all_aspects = parse_multiple_labels(topic_row['label'])
            
            with st.container():
                st.markdown(f"### Topic {topic_id}: Complete Multi-Aspect Analysis")
                st.markdown(f"**{topic_row['display_label']}**")
                st.markdown(f"*Zero-shot classification identified {len(all_aspects)} interconnected aspects*")
                
                # Display ALL aspects in a comprehensive format
                st.markdown("**All Zero-Shot Aspects (Complete List):**")
                for i, aspect in enumerate(all_aspects, 1):
                    # Highlight main concept and show full details
                    main_concept = aspect.split('(')[0].strip()
                    detail_part = f"({aspect.split('(', 1)[1]}" if '(' in aspect else ""
                    
                    if detail_part:
                        st.markdown(f"{i}. **{main_concept}** {detail_part}")
                    else:
                        st.markdown(f"{i}. **{aspect}**")
                
                # Show metrics in a comprehensive layout
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Reviews", f"{topic_row['n']:,}")
                with col2:
                    st.metric("Positive Rate", f"{topic_row['pos']:.1%}")
                with col3:
                    st.metric("Avg Stars", f"{topic_row['stars_mean']:.1f}")
                with col4:
                    st.metric("Share", f"{topic_row['share']:.1%}")
                with col5:
                    st.metric("Complexity", f"{len(all_aspects)} aspects")
                
                st.markdown("---")
        
        st.subheader("Sample Sentences by Multi-Aspect Topic")
        
        # Group by display label for better organization
        display_df_sorted = display_df.sort_values('n', ascending=False)
        
        for _, topic_row in display_df_sorted.head(5).iterrows():
            topic_id = topic_row['topic_id']
            display_label = topic_row['display_label']
            all_aspects = parse_multiple_labels(topic_row['label'])
            
            topic_sentences = sentences_df[sentences_df['topic_id'] == topic_id].head(3)
            
            with st.container():
                # Show full topic information
                st.markdown(f"**{display_label}** (Topic {topic_id})")
                st.caption(f"Encompasses {len(all_aspects)} aspects: {', '.join([a.split('(')[0].strip() for a in all_aspects[:3]])}{'...' if len(all_aspects) > 3 else ''}")
                
                # Show sample sentences
                for i, (_, sentence) in enumerate(topic_sentences.iterrows()):
                    sentiment_emoji = "😊" if sentence.get('sentiment', 0) == 1 else "😔" if sentence.get('sentiment', 0) == 0 else "😐"
                    sentence_text = sentence.get('sentence', 'No sentence data')[:300]  # Increased length
                    if len(sentence.get('sentence', '')) > 300:
                        sentence_text += "..."
                    
                    with st.container():
                        st.write(f"{sentiment_emoji} {sentence_text}")
                        if i < len(topic_sentences) - 1:
                            st.divider()
                
                # Quick aspect analysis for this topic
                with st.expander(f"View all {len(all_aspects)} aspects for this topic"):
                    st.markdown("**Complete Zero-Shot Aspect List:**")
                    for i, aspect in enumerate(all_aspects, 1):
                        st.write(f"{i}. {aspect}")
                
                st.markdown("---")
    
    # Download Options
    st.markdown('<h2 class="section-header">📥 Export Results</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download summary
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Topic Summary",
            data=summary_csv,
            file_name=f"topic_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download sentences (sample)
        sentences_sample = sentences_df.head(1000)  # Limit size
        sentences_csv = sentences_sample.to_csv(index=False)
        st.download_button(
            label="📝 Download Sentences (Sample)",
            data=sentences_csv,
            file_name=f"sentences_sample_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Download executive summary
        if 'llm_insights' in st.session_state:
            st.download_button(
                label="🤖 Download AI Insights",
                data=st.session_state['llm_insights'],
                file_name=f"ai_insights_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
        else:
            st.info("Generate AI insights first")

elif data_source == "Run new pipeline":
    # Show pipeline status if not completed
    if st.session_state.get('pipeline_running', False):
        st.info("🔄 Pipeline is running... Results will appear here when complete.")
    else:
        st.info("👆 Configure and run the pipeline above to see analysis results.")

else:
    # Welcome screen (only show if no data loaded)
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 2rem 0;'>
        <h2>🚀 Welcome to Business Intelligence Dashboard</h2>
        <p style='font-size: 1.2rem; margin: 1rem 0;'>
            Analyze customer sentiment and discover actionable insights from your business reviews
        </p>
        <p>👈 Use the sidebar to load your analysis results or run a new pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Smart Analytics
        - Automatic topic discovery
        - Sentiment analysis
        - Performance tracking
        - Trend identification
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Action Insights
        - Priority matrix
        - Problem identification
        - Improvement recommendations
        - KPI monitoring
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Visual Reports
        - Interactive charts
        - Executive summaries
        - Timeline analysis
        - Comparative metrics
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Business Intelligence Dashboard v2.0 | "
    "Powered by BERTopic & Streamlit</div>", 
    unsafe_allow_html=True
)