import os
import io
import re
import time
import warnings
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

# Suppress warnings
warnings.filterwarnings('ignore')

# --- optional: .env 지원 ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===== Project modules =====
from config import (
    PATH_BUSINESS, PATH_REVIEW,
    PATH_REVIEWS_FILTERED, PATH_SENT_WITH_SENT, PATH_WITH_TOPICS,
    SENTIMENT_MODEL, EMBEDDING_MODEL
)
from modules.find_business_ids import find_business_ids
from modules.filter_reviews import filter_reviews_by_business_ids
from modules.sentence_sentiment import run_sentence_sentiment
from modules.nli_multilabel_final import apply_nli_aspect_analysis, RESTAURANT_ASPECTS  # 새로운 모듈
from modules.business_meta import load_business_meta

# ========= Aspect Analysis Dashboard Functions =========

def create_aspect_kpi_cards(summary_df: pd.DataFrame, sentences_df: pd.DataFrame) -> None:
    """36개 레스토랑 aspect 분석에 특화된 KPI 카드"""
    
    # 기본 통계
    total_reviews = sentences_df['review_id'].nunique() if 'review_id' in sentences_df.columns else len(sentences_df)
    total_aspects_found = len(summary_df) if 'aspect' in summary_df.columns else 0
    avg_satisfaction = summary_df['avg_stars'].mean() if 'avg_stars' in summary_df.columns else 0
    overall_positive_rate = summary_df['positive_ratio'].mean() if 'positive_ratio' in summary_df.columns else 0
    
    # 가장 문제되는/잘하는 aspect
    if not summary_df.empty and 'positive_ratio' in summary_df.columns:
        worst_aspect = summary_df.loc[summary_df['positive_ratio'].idxmin()]
        best_aspect = summary_df.loc[summary_df['positive_ratio'].idxmax()]
    else:
        worst_aspect = best_aspect = None
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Review Sentences Analyzed", 
            value=f"{total_reviews:,}",
            help="Total number of reviews processed for aspect analysis"
        )
    
    with col2:
        st.metric(
            label="🏷️ Aspects Identified", 
            value=f"{total_aspects_found}",
            help=f"Restaurant aspects found out of 36 predefined categories"
        )
    
    with col3:
        st.metric(
            label="⭐ Average Satisfaction", 
            value=f"{avg_satisfaction:.1f}/5.0" if avg_satisfaction > 0 else "N/A",
            delta=f"{(avg_satisfaction - 3.0):.1f} vs neutral" if avg_satisfaction > 0 else None,
            help="Average star rating across all aspects"
        )
    
    with col4:
        st.metric(
            label="😊 Positive Sentiment Rate", 
            value=f"{overall_positive_rate:.1%}" if overall_positive_rate > 0 else "N/A",
            delta=f"{(overall_positive_rate - 0.5):.1%} vs 50%" if overall_positive_rate > 0 else None,
            help="Overall positive sentiment rate across aspects"
        )
    
    # Alert cards for critical insights
    if worst_aspect is not None and best_aspect is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style='padding: 1rem; background: linear-gradient(135deg, #fff2f2 0%, #ffe6e6 100%); border-left: 4px solid #ff4444; border-radius: 8px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4 style='color: #cc0000; margin: 0; font-size: 1.1rem;'>🚨 Most Critical Aspect</h4>
                <p style='margin: 0.5rem 0; font-weight: bold; color: #333; line-height: 1.3;'>{worst_aspect['aspect']}</p>
                <p style='margin: 0; color: #666; font-size: 0.9rem;'>Positive Rate: {worst_aspect['positive_ratio']:.1%} | Mentions: {worst_aspect['n_sentences']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='padding: 1rem; background: linear-gradient(135deg, #f2fff2 0%, #e6ffe6 100%); border-left: 4px solid #44ff44; border-radius: 8px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4 style='color: #006600; margin: 0; font-size: 1.1rem;'>✨ Strongest Performance</h4>
                <p style='margin: 0.5rem 0; font-weight: bold; color: #333; line-height: 1.3;'>{best_aspect['aspect']}</p>
                <p style='margin: 0; color: #666; font-size: 0.9rem;'>Positive Rate: {best_aspect['positive_ratio']:.1%} | Mentions: {best_aspect['n_sentences']}</p>
            </div>
            """, unsafe_allow_html=True)

def create_aspect_performance_overview(summary_df: pd.DataFrame) -> None:
    """36개 aspect의 성과 개요 시각화"""
    
    if summary_df.empty or 'aspect' not in summary_df.columns:
        st.warning("No aspect data available")
        return
    
    # 데이터 준비
    df = summary_df.copy()
    df = df.sort_values('n_sentences', ascending=True)  # 가독성을 위해 오름차순
    
    # Aspect 카테고리 매핑 (36개 aspect를 그룹화)
    def categorize_aspect(aspect_name):
        aspect_lower = aspect_name.lower()
        if any(word in aspect_lower for word in ['service', 'server', 'host', 'wait', 'order', 'bill', 'manager']):
            return 'Service & Operations'
        elif any(word in aspect_lower for word in ['clean', 'pest', 'safety', 'tableware', 'restroom']):
            return 'Cleanliness & Safety'
        elif any(word in aspect_lower for word in ['noise', 'music', 'lighting', 'temperature', 'smell', 'decor', 'seating']):
            return 'Environment & Ambience'
        elif any(word in aspect_lower for word in ['parking', 'location', 'accessibility', 'family', 'kids']):
            return 'Accessibility & Family'
        elif any(word in aspect_lower for word in ['menu', 'value', 'price', 'portion', 'allergen', 'taste', 'ingredient', 'food', 'freshness']):
            return 'Food & Menu'
        elif any(word in aspect_lower for word in ['delivery', 'takeout', 'online', 'packaging']):
            return 'Delivery & Takeout'
        else:
            return 'Other'
    
    df['category'] = df['aspect'].apply(categorize_aspect)
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Aspect Performance Matrix', 'Volume Distribution', 
                       'Performance by Category', 'Top/Bottom Performers'),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Bubble chart: 성과 매트릭스
    hover_text = []
    for _, row in df.iterrows():
        share_text = f"Share: {row['share']:.1%}" if 'share' in df.columns and pd.notna(row.get('share')) else ""
        hover_text.append(
            f"<b>{row['aspect']}</b><br>"
            f"Category: {row['category']}<br>"
            f"Avg Stars: {row['avg_stars']:.1f}<br>"
            f"Positive Rate: {row['positive_ratio']:.1%}<br>"
            f"Mentions: {row['n_sentences']}<br>"
            f"{share_text}"
        )
    
    # 버블 크기 조정 (로그 스케일 + 정규화로 편차 줄이기)
    min_mentions = df['n_sentences'].min()
    max_mentions = df['n_sentences'].max()
    
    # 로그 스케일로 변환하여 편차 줄이기
    log_mentions = np.log1p(df['n_sentences'])  # log1p = log(1+x)로 0값 처리
    min_log = log_mentions.min()
    max_log = log_mentions.max()
    
    # 10~35 사이로 정규화
    if max_log > min_log:
        normalized_sizes = 10 + (log_mentions - min_log) / (max_log - min_log) * 25
    else:
        normalized_sizes = 20  # 모든 값이 같으면 중간값
    
    fig.add_trace(
        go.Scatter(
            x=df['avg_stars'],
            y=df['positive_ratio'],
            mode='markers',
            marker=dict(
                size=normalized_sizes,
                color=df['n_sentences'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(
                    title="★ Rating",
                    x=0.46,y=0.85,
                    len=0.4,        # 길이 단축
                    thickness=15,   # 두께 조정
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=["1★", "2★", "3★", "4★", "5★"]
                    # titleside 제거됨
                ),
                line=dict(width=1, color='white'),
                opacity=0.8,
                sizemin=8  # 최소 크기 보장
            ),
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_text,
            name="Aspects",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Pie chart: 볼륨 분포 (상위 10개)
    top_aspects = df.nlargest(10, 'n_sentences')
    pie_labels = [label[:15] + "..." if len(label) > 15 else label for label in top_aspects['aspect']]
    
    fig.add_trace(
        go.Pie(
            labels=pie_labels,
            values=top_aspects['n_sentences'],
            hovertemplate="<b>%{text}</b><br>Mentions: %{value}<br>Share: %{percent}<extra></extra>",
            hovertext=top_aspects['aspect'],
            name="Volume",
            textinfo='percent',
            textposition='auto',
            textfont=dict(size=9),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. 카테고리별 성과
    category_stats = df.groupby('category').agg({
        'positive_ratio': 'mean',
        'n_sentences': 'sum'
    }).sort_values('positive_ratio', ascending=True)
    
    fig.add_trace(
        go.Bar(
            x=category_stats['positive_ratio'],
            y=category_stats.index,
            orientation='h',
            marker_color='#3498db',
            name="Category Performance",
            hovertemplate="<b>%{y}</b><br>Positive Rate: %{x:.1%}<extra></extra>",
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Top/Bottom performers
    top_5 = df.nlargest(5, 'positive_ratio')
    bottom_5 = df.nsmallest(5, 'positive_ratio')
    performers = pd.concat([bottom_5, top_5])
    colors = ['#e74c3c'] * len(bottom_5) + ['#27ae60'] * len(top_5)
    
    # Y축 라벨을 적절한 길이로 자르기
    y_labels = [label[:25] + "..." if len(label) > 25 else label for label in performers['aspect']]
    
    fig.add_trace(
        go.Bar(
            x=performers['positive_ratio'],
            y=y_labels,
            orientation='h',
            marker_color=colors,
            name="Performers",
            hovertemplate="<b>%{text}</b><br>Positive Rate: %{x:.1%}<extra></extra>",
            text=performers['aspect'],
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 레이아웃 업데이트
    fig.update_xaxes(title_text="Average Star Rating", row=1, col=1, range=[0.5, 5.5])
    fig.update_yaxes(title_text="Positive Sentiment Rate", row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Positive Rate", row=2, col=1)
    fig.update_xaxes(title_text="Positive Rate", row=2, col=2)
    
    fig.update_layout(
        height=900,
        showlegend=False,  # 전체 legend 비활성화
        title_text="ReviewToRevenue: Restaurant Review Analysis Dashboard",
        title_x=0.5,
        title_font_size=16,
        margin=dict(l=50, r=50, t=100, b=100)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Summarization functions from app_new.py
@st.cache_resource
def load_summarizer_model(model_name="facebook/bart-large-cnn"):
    """BART 요약 모델 로드 (캐싱 적용) - max_length 설정 개선"""
    try:
        from transformers import pipeline, AutoTokenizer
        
        # 토크나이저와 파이프라인 모두 max_length 설정
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return pipeline(
            "summarization", 
            model=model_name, 
            tokenizer=tokenizer,
            device=-1,  # CPU 사용
            model_kwargs={'max_length': 1024},
            # 파이프라인 기본 설정
            max_length=130,
            min_length=30,
            truncation=True
        )
    except ImportError:
        st.error("Transformers library not available. Summarization feature disabled.")
        return None
    except Exception as e:
        st.error(f"Error loading summarization model: {str(e)}")
        return None

@st.cache_data(show_spinner="Generating aspect summary...")
def get_aspect_summary(sentences_list, aspect_name, sentiment_type="all"):
    """특정 aspect에 대한 리뷰 요약 생성"""
    
    if not sentences_list:
        return f"No sentences available for '{aspect_name}' summary."

    # 문장 수가 많을 경우 처리 (상위 15개 선택)
    if len(sentences_list) > 15:
        sentences_list = sentences_list[:15]

    combined_text = " ".join([str(s) for s in sentences_list if s and len(str(s).strip()) > 10])
    
    if not combined_text.strip():
        return f"No sufficient content for '{aspect_name}' summary."
    
    try:
        from transformers import AutoTokenizer
        
        # max_length 명시적으로 설정
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        # 토큰 수에 따라 동적으로 요약 길이 조정 - max_length 명시
        encoded = tokenizer.encode(combined_text, truncation=True, max_length=1024, add_special_tokens=True)
        input_token_count = len(encoded)
        dynamic_max_length = max(40, min(130, int(input_token_count * 0.4)))
        min_len = max(15, int(dynamic_max_length * 0.4))
        
        summarizer = load_summarizer_model()
        if not summarizer:
            return "Summarization model not available."
        
        summary_result = summarizer(
            combined_text,
            max_length=dynamic_max_length,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )
        return summary_result[0]['summary_text']

    except Exception as e:
        return f"Unable to generate summary for '{aspect_name}': {str(e)[:100]}..."


def create_aspect_category_analysis(summary_df: pd.DataFrame, sentences_df: pd.DataFrame = None) -> None:
    """Aspect 카테고리별 상세 분석"""
    
    if summary_df.empty:
        st.warning("No data available for category analysis")
        return
    
    st.markdown('<h2 class="section-header">🏗️ Analysis by Aspect Category</h2>', unsafe_allow_html=True)
    
    # 카테고리 정의 (더 상세하게)
    ASPECT_CATEGORIES = {
        'Service & Operations': [
            'wait time and queue management', 'host and seating process', 
            'server friendliness and politeness', 'server attentiveness and follow-ups',
            'order accuracy and missing items', 'kitchen speed and ticket time',
            'bill handling and split checks', 'payment methods and checkout',
            'manager response and recovery'
        ],
        'Cleanliness & Safety': [
            'tableware and utensils cleanliness', 'dining area cleanliness',
            'restroom cleanliness and supplies'
        ],
        'Environment & Ambience': [
            'noise level and crowding', 'music volume and selection',
            'lighting and visibility', 'temperature and ventilation',
            'smell and odors', 'decor and interior design',
            'seating comfort and space', 'outdoor seating and patio'
        ],
        'Accessibility & Family': [
            'parking convenience and options', 'location and transit accessibility',
            'ada accessibility and ramps', 'family friendly and kids options'
        ],
        'Food & Menu': [
            'menu variety and seasonal specials', 'menu clarity and descriptions',
            'value for money and price fairness', 'portion size and fullness',
            'allergen handling and cross contamination', 'overall taste and seasoning balance',
            'ingredient freshness and quality', 'texture and doneness accuracy',
            'temperature of dishes at serving'
        ],
        'Delivery & Takeout': [
            'online ordering usability', 'delivery time and temperature',
            'takeout packaging and spill protection'
        ]
    }
    
    # 역방향 매핑
    aspect_to_category = {}
    for category, aspects in ASPECT_CATEGORIES.items():
        for aspect in aspects:
            aspect_to_category[aspect] = category
    
    # 데이터 준비
    df = summary_df.copy()
    df['category'] = df['aspect'].map(aspect_to_category).fillna('Other')
    
    # 카테고리별 탭 생성 (한 번만)
    categories = list(ASPECT_CATEGORIES.keys())
    tabs = st.tabs([f"📂 {cat}" for cat in categories])
    
    for i, (category, tab) in enumerate(zip(categories, tabs)):
        with tab:
            category_data = df[df['category'] == category]
            
            if category_data.empty:
                st.info(f"No data available for {category} aspects")
                continue
            
            # 카테고리 요약
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Aspects Found", len(category_data))
            with col2:
                st.metric("Total Mentions", f"{category_data['n_sentences'].sum():,}")
            with col3:
                st.metric("Avg Performance", f"{category_data['positive_ratio'].mean():.1%}")
            with col4:
                st.metric("Avg Stars", f"{category_data['avg_stars'].mean():.1f}")
            
            # 서브 탭: 차트 vs 요약 (sentences_df가 있을 때만)
            if sentences_df is not None and not sentences_df.empty:
                subtab1, subtab2 = st.tabs(["📊 Performance Charts", "📝 AI Summaries"])
                
                with subtab1:
                    create_category_charts(category, category_data)
                
                with subtab2:
                    create_category_summaries(category, category_data, sentences_df)
            else:
                # 요약 데이터가 없으면 차트만
                create_category_charts(category, category_data)


def create_category_charts(category: str, category_data: pd.DataFrame):
    """카테고리별 성과 차트 생성"""
    # 상세 차트
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{category} - Performance', f'{category} - Volume'),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.15
    )
    
    # 성과 차트
    sorted_data = category_data.sort_values('positive_ratio', ascending=True)
    colors = ['#ff4444' if x < 0.5 else '#ffaa44' if x < 0.7 else '#44ff44' for x in sorted_data['positive_ratio']]
    
    # Y축 라벨 길이 조정
    y_labels_perf = [label[:25] + "..." if len(label) > 25 else label for label in sorted_data['aspect']]
    
    fig.add_trace(
        go.Bar(
            x=sorted_data['positive_ratio'],
            y=y_labels_perf,
            orientation='h',
            marker_color=colors,
            hovertemplate="<b>%{text}</b><br>Positive Rate: %{x:.1%}<extra></extra>",
            text=sorted_data['aspect'],  # 전체 이름은 hover에 표시
            showlegend=False,  # legend 숨김
            name="Performance"
        ),
        row=1, col=1
    )
    
    # 볼륨 차트
    volume_sorted = category_data.sort_values('n_sentences', ascending=True)
    y_labels_vol = [label[:25] + "..." if len(label) > 25 else label for label in volume_sorted['aspect']]
    
    fig.add_trace(
        go.Bar(
            x=volume_sorted['n_sentences'],
            y=y_labels_vol,
            orientation='h',
            marker_color='#3498db',
            hovertemplate="<b>%{text}</b><br>Mentions: %{x}<extra></extra>",
            text=volume_sorted['aspect'],  # 전체 이름은 hover에 표시
            showlegend=False,  # legend 숨김
            name="Volume"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=max(400, len(category_data) * 30),  # 동적 높이 조정
        showlegend=False,  # 전체 legend 비활성화
        margin=dict(l=200, r=50, t=80, b=50),  # 마진 조정
        title_text=None  # 제목 제거 (subplot_titles 사용)
    )
    
    # 축 레이블 설정
    fig.update_xaxes(title_text="Positive Rate", row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Number of Mentions", row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=10))  # Y축 폰트 크기 조정
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 상세 테이블
    st.subheader("📋 Detailed Breakdown")
    display_cols = ['aspect', 'n_sentences', 'positive_ratio', 'avg_stars']
    if 'share' in category_data.columns and not category_data['share'].isnull().all():
        display_cols.append('share')
    display_cols = [col for col in display_cols if col in category_data.columns]
    
    st.dataframe(
        category_data[display_cols].sort_values('n_sentences', ascending=False),
        use_container_width=True,
        column_config={
            "aspect": st.column_config.TextColumn("Aspect", width="large"),
            "n_sentences": st.column_config.NumberColumn("Mentions", width="small"),
            "positive_ratio": st.column_config.NumberColumn("Positive Rate", format="%.1%", width="small"),
            "avg_stars": st.column_config.NumberColumn("Avg Stars", format="%.1f", width="small"),
            "share": st.column_config.NumberColumn("Share", format="%.1%", width="small") if "share" in display_cols else None
        },
        hide_index=True
    )


def create_category_summaries(category: str, category_data: pd.DataFrame, sentences_df: pd.DataFrame):
    """카테고리별 AI 요약 생성"""
    
    st.subheader(f"🤖 {category} - AI-Generated Summaries")
    
    # 요약 모델 사용 가능성 확인
    summarizer = load_summarizer_model()
    if not summarizer:
        st.warning("⚠️ AI summarization is not available. Please install the transformers library.")
        return
    
    # 카테고리 내 top aspects 선택
    top_aspects = category_data.nlargest(5, 'n_sentences')
    
    if top_aspects.empty:
        st.info("No aspects available for summary in this category")
        return
    
    # 각 aspect별 요약
    for _, aspect_row in top_aspects.iterrows():
        aspect_name = aspect_row['aspect']
        
        # 해당 aspect의 문장들 가져오기
        if 'assigned_aspects' in sentences_df.columns:
            # assigned_aspects에서 해당 aspect 포함 문장 찾기
            aspect_sentences = sentences_df[
                sentences_df['assigned_aspects'].apply(
                    lambda x: aspect_name in (eval(x) if isinstance(x, str) and x.startswith('[') else [x]) if pd.notna(x) else False
                )
            ]
        else:
            # primary_aspect 사용 (fallback)
            aspect_sentences = sentences_df[sentences_df.get('primary_aspect', '') == aspect_name]
        
        if aspect_sentences.empty:
            continue
        
        # Expander로 각 aspect 요약 표시
        with st.expander(f"📋 {aspect_name} Summary ({len(aspect_sentences)} mentions)"):
            
            # 메트릭 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                positive_rate = (aspect_sentences.get('sentiment', 0) == 1).mean()
                st.metric("Positive Rate", f"{positive_rate:.1%}")
            with col2:
                avg_stars = aspect_sentences.get('stars', 0).mean()
                st.metric("Avg Stars", f"{avg_stars:.1f}")
            with col3:
                st.metric("Total Mentions", f"{len(aspect_sentences):,}")
            
            # 긍정/부정 분리
            if 'sentiment' in aspect_sentences.columns:
                positive_sentences = aspect_sentences[aspect_sentences['sentiment'] == 1]['sentence'].tolist()
                negative_sentences = aspect_sentences[aspect_sentences['sentiment'] == 0]['sentence'].tolist()
            else:
                # Fallback: 별점 기준
                positive_sentences = aspect_sentences[aspect_sentences.get('stars', 0) >= 4]['sentence'].tolist()
                negative_sentences = aspect_sentences[aspect_sentences.get('stars', 0) <= 2]['sentence'].tolist()
            
            # 요약 생성 및 표시
            col_pos, col_neg = st.columns(2)
            
            with col_pos:
                st.markdown("##### 👍 Positive Feedback Summary")
                if positive_sentences:
                    with st.spinner("Generating positive summary..."):
                        pos_summary = get_aspect_summary(positive_sentences, aspect_name, "positive")
                    st.success(pos_summary)
                else:
                    st.info("No positive reviews available")
            
            with col_neg:
                st.markdown("##### 👎 Negative Feedback Summary")
                if negative_sentences:
                    with st.spinner("Generating negative summary..."):
                        neg_summary = get_aspect_summary(negative_sentences, aspect_name, "negative")
                    st.error(neg_summary)
                else:
                    st.info("No negative reviews available")
            
            # 샘플 문장들 표시
            with st.expander("📝 Sample Sentences", expanded=False):
                if positive_sentences:
                    st.markdown("**Positive Examples:**")
                    for i, sentence in enumerate(positive_sentences[:3], 1):
                        st.write(f"{i}. {sentence[:150]}{'...' if len(sentence) > 150 else ''}")
                
                if negative_sentences:
                    st.markdown("**Negative Examples:**")
                    for i, sentence in enumerate(negative_sentences[:3], 1):
                        st.write(f"{i}. {sentence[:150]}{'...' if len(sentence) > 150 else ''}")
    
    # 카테고리 전체 요약
    st.markdown("---")
    st.subheader(f"🎯 Overall {category} Category Summary")
    
    if st.button(f"Generate {category} Category Summary", key=f"category_summary_{category}"):
        # 카테고리 내 모든 문장 수집
        all_sentences = []
        for _, aspect_row in category_data.iterrows():
            aspect_name = aspect_row['aspect']
            if 'assigned_aspects' in sentences_df.columns:
                aspect_sentences = sentences_df[
                    sentences_df['assigned_aspects'].apply(
                        lambda x: aspect_name in (eval(x) if isinstance(x, str) and x.startswith('[') else [x]) if pd.notna(x) else False
                    )
                ]['sentence'].tolist()
            else:
                aspect_sentences = sentences_df[sentences_df.get('primary_aspect', '') == aspect_name]['sentence'].tolist()
            all_sentences.extend(aspect_sentences)
        
        if all_sentences:
            with st.spinner(f"Generating {category} category overview..."):
                category_summary = get_aspect_summary(all_sentences[:20], category, "overall")  # 상위 20개만
            st.info(f"**{category} Overall Insights:** {category_summary}")
        else:
            st.warning(f"No sentences found for {category} category summary")

def create_aspect_priority_matrix(summary_df: pd.DataFrame) -> None:
    """Aspect별 우선순위 매트릭스 - 개선된 버전"""
    
    st.markdown('<h2 class="section-header">🎯 Aspect Priority Matrix</h2>', unsafe_allow_html=True)
    
    if summary_df.empty or len(summary_df) < 2:
        st.warning("Insufficient data for priority analysis. Need at least 2 aspects.")
        return
    
    try:
        df = summary_df.copy()
        
        # 필수 컬럼 확인
        required_cols = ['aspect', 'n_sentences']
        optional_cols = ['positive_ratio', 'avg_stars', 'share']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return
        
        # 데이터 검증 및 기본값 처리
        if 'positive_ratio' in df.columns:
            df['positive_ratio'] = pd.to_numeric(df['positive_ratio'], errors='coerce').fillna(0.5)
        else:
            df['positive_ratio'] = 0.5  # 기본값
            
        if 'avg_stars' in df.columns:
            df['avg_stars'] = pd.to_numeric(df['avg_stars'], errors='coerce').fillna(3.0)
        else:
            df['avg_stars'] = 3.0  # 기본값
            
        df['n_sentences'] = pd.to_numeric(df['n_sentences'], errors='coerce').fillna(1)
        
        # share 컬럼이 없으면 계산
        if 'share' not in df.columns or df['share'].isnull().all():
            total_sentences = df['n_sentences'].sum()
            if total_sentences > 0:
                df['share'] = df['n_sentences'] / total_sentences
            else:
                df['share'] = 1 / len(df)  # 균등 배분
        else:
            df['share'] = pd.to_numeric(df['share'], errors='coerce')
            # share가 모두 NaN이면 재계산
            if df['share'].isnull().all():
                total_sentences = df['n_sentences'].sum()
                df['share'] = df['n_sentences'] / total_sentences if total_sentences > 0 else 1 / len(df)
            else:
                df['share'] = df['share'].fillna(df['n_sentences'] / df['n_sentences'].sum())
        
        # 우선순위 점수 계산
        df['impact_score'] = df['share'] * (1 - df['positive_ratio'])  # 높은 볼륨 + 낮은 만족도
        df['urgency_score'] = (5 - df['avg_stars']) / 4  # 낮은 별점 = 높은 긴급도
        
        # 극값 처리 (outlier 방지)
        df['impact_score'] = np.clip(df['impact_score'], 0, 1)
        df['urgency_score'] = np.clip(df['urgency_score'], 0, 1)
        
        # NaN 값 처리
        df['impact_score'] = df['impact_score'].fillna(0)
        df['urgency_score'] = df['urgency_score'].fillna(0)
        
        # 우선순위 분류 (더 관대한 기준)
        if len(df) >= 3:
            impact_threshold_low = df['impact_score'].quantile(0.33)
            impact_threshold_high = df['impact_score'].quantile(0.67)
            urgency_threshold_low = df['urgency_score'].quantile(0.33)
            urgency_threshold_high = df['urgency_score'].quantile(0.67)
        else:
            # 데이터가 적을 때는 중앙값 사용
            impact_threshold_low = impact_threshold_high = df['impact_score'].median()
            urgency_threshold_low = urgency_threshold_high = df['urgency_score'].median()
        
        # 중앙값도 계산 (사분면 라인용)
        impact_median = df['impact_score'].median()
        urgency_median = df['urgency_score'].median()
        
        df['priority'] = 'Low'  # 기본값
        df.loc[(df['impact_score'] > impact_threshold_low) | (df['urgency_score'] > urgency_threshold_low), 'priority'] = 'Medium'
        df.loc[(df['impact_score'] > impact_threshold_high) & (df['urgency_score'] > urgency_threshold_high), 'priority'] = 'High'
        
        # 적어도 각 우선순위에 하나씩은 할당되도록 보장
        if len(df[df['priority'] == 'High']) == 0 and len(df) > 2:
            # 가장 높은 impact + urgency 조합을 High로
            max_idx = (df['impact_score'] + df['urgency_score']).idxmax()
            df.loc[max_idx, 'priority'] = 'High'
        
        if len(df[df['priority'] == 'Medium']) == 0 and len(df) > 1:
            # 중간값들을 Medium으로
            remaining = df[df['priority'] == 'Low']
            if len(remaining) > 1:
                mid_scores = (remaining['impact_score'] + remaining['urgency_score']).nlargest(len(remaining)//2)
                df.loc[mid_scores.index, 'priority'] = 'Medium'
        
        # 짧은 라벨 생성
                # 더 짧은 라벨 생성 (가독성 개선)
        df['short_label'] = df['aspect'].apply(lambda x: str(x)[:12] + "..." if len(str(x)) > 12 else str(x))
        
        # 라벨 표시할 것들만 선별 (High Priority + 상위 볼륨 몇 개)
        df['show_label'] = False
        high_priority_indices = df[df['priority'] == 'High'].index
        top_volume_indices = df.nlargest(5, 'n_sentences').index
        label_indices = set(high_priority_indices.tolist() + top_volume_indices.tolist())
        df.loc[list(label_indices), 'show_label'] = True
        # 시각화
        color_map = {'High': '#ff4444', 'Medium': '#ffaa44', 'Low': '#44ff44'}
        
        fig = go.Figure()
        
        for priority in ['High', 'Medium', 'Low']:
            priority_data = df[df['priority'] == priority]
            if priority_data.empty:
                continue
                
            hover_text = []
            for _, row in priority_data.iterrows():
                hover_text.append(
                    f"<b>{row['aspect']}</b><br>"
                    f"Priority: {priority}<br>"
                    f"Impact Score: {row['impact_score']:.3f}<br>"
                    f"Urgency Score: {row['urgency_score']:.3f}<br>"
                    f"Mentions: {row['n_sentences']}<br>"
                    f"Positive Rate: {row['positive_ratio']:.1%}<br>"
                    f"Avg Stars: {row['avg_stars']:.1f}"
                )
            
            # 버블 크기 계산 (안전하게)
            bubble_sizes = np.sqrt(priority_data['n_sentences']) * 3
            bubble_sizes = np.clip(bubble_sizes, 8, 50)  # 최소 8, 최대 50
            
            fig.add_trace(go.Scatter(
                x=priority_data['impact_score'],
                y=priority_data['urgency_score'],
                mode='markers+text',
                marker=dict(
                    size=bubble_sizes,
                    color=color_map[priority],
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=priority_data['short_label'],
                textposition="top center",
                textfont=dict(size=9, color='black'),
                name=f'{priority} Priority',
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_text,
                showlegend=True
            ))
        
        # 레이아웃 및 참조선
        fig.update_layout(
            title="Aspect Priority Matrix - Size = Volume, Color = Priority Level",
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
            margin=dict(r=150)
        )
        
        # 사분면 라인 (안전하게)
        if not pd.isna(urgency_median) and not pd.isna(impact_median):
            fig.add_hline(y=urgency_median, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=impact_median, line_dash="dash", line_color="gray", opacity=0.5)
        
        # 사분면 라벨
        max_impact = df['impact_score'].max() if not df['impact_score'].empty else 1
        max_urgency = df['urgency_score'].max() if not df['urgency_score'].empty else 1
        
        if max_impact > 0 and max_urgency > 0:
            annotations = [
                dict(x=max_impact*0.8, y=max_urgency*0.9, text="High Impact<br>High Urgency", 
                     showarrow=False, font=dict(size=12, color="red"), 
                     bgcolor="rgba(255,255,255,0.8)", bordercolor="red", borderwidth=1),
                dict(x=max_impact*0.2, y=max_urgency*0.9, text="Low Impact<br>High Urgency", 
                     showarrow=False, font=dict(size=10, color="orange"), 
                     bgcolor="rgba(255,255,255,0.8)"),
                dict(x=max_impact*0.8, y=max_urgency*0.2, text="High Impact<br>Low Urgency", 
                     showarrow=False, font=dict(size=10, color="blue"), 
                     bgcolor="rgba(255,255,255,0.8)"),
                dict(x=max_impact*0.2, y=max_urgency*0.2, text="Low Impact<br>Low Urgency", 
                     showarrow=False, font=dict(size=10, color="green"), 
                     bgcolor="rgba(255,255,255,0.8)"),
            ]
            fig.update_layout(annotations=annotations)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 통계 요약
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("High Priority", len(df[df['priority'] == 'High']))
        with col2:
            st.metric("Medium Priority", len(df[df['priority'] == 'Medium']))
        with col3:
            st.metric("Low Priority", len(df[df['priority'] == 'Low']))
        
        # 액션 추천
        high_priority = df[df['priority'] == 'High'].sort_values('impact_score', ascending=False)
        
        if not high_priority.empty:
            st.subheader("🚨 High Priority Aspects - Immediate Action Required")
            
            for i, (_, aspect) in enumerate(high_priority.head(5).iterrows(), 1):
                with st.expander(f"**{i}. {aspect['aspect']}** - Impact Score: {aspect['impact_score']:.3f}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mentions", f"{int(aspect['n_sentences']):,}")
                    with col2:
                        st.metric("Positive Rate", f"{aspect['positive_ratio']:.1%}")
                    with col3:
                        st.metric("Avg Stars", f"{aspect['avg_stars']:.1f}")
                    with col4:
                        st.metric("Share", f"{aspect['share']:.1%}")
                    
                    # 액션 추천
                    st.markdown("**Recommended Actions:**")
                    impact = aspect['impact_score']
                    urgency = aspect['urgency_score']
                    
                    if impact > 0.1 and urgency > 0.6:
                        st.error("🔥 **Critical**: High volume of negative feedback. Immediate operational review needed.")
                    elif impact > 0.05:
                        st.warning("⚠️ **Important**: Significant customer impact. Schedule improvement plan within 2 weeks.")
                    else:
                        st.info("📋 **Monitor**: Track improvements and customer feedback closely.")
        else:
            st.success("✅ No high-priority aspects identified - overall performance is good!")
        
        # 데이터 품질 정보
        with st.expander("ℹ️ Analysis Details"):
            st.write(f"**Priority Distribution:**")
            priority_counts = df['priority'].value_counts()
            for priority, count in priority_counts.items():
                st.write(f"- {priority} Priority: {count} aspects")
            
            st.write(f"\n**Score Ranges:**")
            st.write(f"- Impact Score: {df['impact_score'].min():.3f} - {df['impact_score'].max():.3f}")
            st.write(f"- Urgency Score: {df['urgency_score'].min():.3f} - {df['urgency_score'].max():.3f}")
            st.write(f"- Median Impact: {impact_median:.3f}, Median Urgency: {urgency_median:.3f}")
    
    except Exception as e:
        st.error(f"Error in priority matrix analysis: {str(e)}")
        st.info("This may be due to insufficient data or data format issues.")
        
        # 디버그 정보
        with st.expander("🔧 Debug Information"):
            st.write("**Available columns:**", list(summary_df.columns))
            st.write("**Data shape:**", summary_df.shape)
            st.write("**Sample data:**")
            st.dataframe(summary_df.head(3))
            if not summary_df.empty:
                st.write("**Data types:**")
                st.write(summary_df.dtypes)
        
        return

def create_aspect_timeline_analysis(sentences_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Aspect별 시간대별 트렌드 분석 - 개선된 안전성"""
    
    st.markdown('<h2 class="section-header">📈 Aspect Performance Trends</h2>', unsafe_allow_html=True)
    
    if 'date' not in sentences_df.columns:
        st.warning("Date information not available for timeline analysis")
        return
    
    try:
        # 데이터 준비
        timeline_df = sentences_df.copy()
        timeline_df['date'] = pd.to_datetime(timeline_df['date'], errors='coerce')
        timeline_df = timeline_df.dropna(subset=['date'])
        
        if timeline_df.empty:
            st.warning("No valid date data available for timeline analysis")
            return
        
        timeline_df['month'] = timeline_df['date'].dt.to_period('M').astype(str)
        
        # Aspect별 데이터 준비 (assigned_aspects 컬럼 사용)
        if 'assigned_aspects' not in timeline_df.columns and 'primary_aspect' not in timeline_df.columns:
            st.warning("Aspect assignment data not available")
            return
        
        # 각 문장에서 할당된 aspect들 전개
        aspect_timeline_data = []
        for _, row in timeline_df.iterrows():
            aspects = []
            
            # assigned_aspects 컬럼 처리
            if 'assigned_aspects' in timeline_df.columns and pd.notna(row.get('assigned_aspects')):
                assigned = row.get('assigned_aspects', [])
                if isinstance(assigned, str):
                    try:
                        # 문자열로 저장된 리스트 파싱
                        aspects = eval(assigned) if assigned.startswith('[') else [assigned]
                    except:
                        aspects = [assigned] if assigned else []
                elif isinstance(assigned, list):
                    aspects = assigned
            
            # primary_aspect 컬럼 fallback
            if not aspects and 'primary_aspect' in timeline_df.columns:
                primary = row.get('primary_aspect')
                if pd.notna(primary) and primary != 'unclassified':
                    aspects = [primary]
            
            # aspect별 데이터 추가
            for aspect in aspects:
                if aspect and str(aspect).strip():
                    aspect_timeline_data.append({
                        'month': row['month'],
                        'aspect': str(aspect).strip(),
                        'sentiment': row.get('sentiment', 0),
                        'stars': row.get('stars', 3),
                        'business_id': row.get('business_id', 'unknown')
                    })
        
        if not aspect_timeline_data:
            st.warning("No aspect timeline data available")
            return
        
        aspect_timeline_df = pd.DataFrame(aspect_timeline_data)
        
        # Top aspects 선택 (최소 데이터 있는 것들만)
        aspect_counts = aspect_timeline_df['aspect'].value_counts()
        top_aspects = aspect_counts[aspect_counts >= 3].head(10).index.tolist()  # 최소 3개 이상
        
        if not top_aspects:
            st.warning("Insufficient data for timeline analysis. Need aspects with at least 3 mentions.")
            return
        
        # 사용자 선택
        col1, col2 = st.columns(2)
        with col1:
            max_selection = min(8, len(top_aspects))  # 최대 8개
            default_selection = top_aspects[:min(5, len(top_aspects))]  # 최대 5개 기본 선택
            
            selected_aspects = st.multiselect(
                "Select aspects to analyze:",
                options=top_aspects,
                default=default_selection,
                key="aspect_timeline_selection"
            )
        
        with col2:
            time_grouping = st.selectbox("Time grouping", ["Monthly", "Quarterly"], index=0)
        
        if not selected_aspects:
            st.warning("Please select at least one aspect")
            return
        
        # 시간 그룹핑
        if time_grouping == "Quarterly":
            aspect_timeline_df['period'] = pd.to_datetime(aspect_timeline_df['month'] + '-01').dt.to_period('Q').astype(str)
        else:
            aspect_timeline_df['period'] = aspect_timeline_df['month']
        
        # 월별 aspect 성과 계산
        monthly_aspect_stats = (
            aspect_timeline_df[aspect_timeline_df['aspect'].isin(selected_aspects)]
            .groupby(['period', 'aspect'])
            .agg({
                'sentiment': 'mean',
                'stars': 'mean',
                'aspect': 'count'
            })
            .rename(columns={'aspect': 'count'})
            .reset_index()
        )
        
        if monthly_aspect_stats.empty:
            st.warning("No data available for selected aspects and time period")
            return
        
        # 시각화
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Positive Sentiment Rate Over Time', 'Mention Volume Over Time'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set3[:len(selected_aspects)]  # 선택된 aspect 수만큼만
        
        for i, aspect in enumerate(selected_aspects):
            aspect_data = monthly_aspect_stats[monthly_aspect_stats['aspect'] == aspect].sort_values('period')
            
            if len(aspect_data) == 0:
                continue
            
            color = colors[i % len(colors)]
            aspect_short = aspect[:25] + "..." if len(aspect) > 25 else aspect
            
            # Sentiment trend
            fig.add_trace(
                go.Scatter(
                    x=aspect_data['period'],
                    y=aspect_data['sentiment'],
                    mode='lines+markers',
                    name=aspect_short,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{aspect}</b><br>' +  # 전체 이름 표시
                                'Period: %{x}<br>' +
                                'Positive Rate: %{y:.1%}<br>' +
                                '<extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Volume trend
            fig.add_trace(
                go.Scatter(
                    x=aspect_data['period'],
                    y=aspect_data['count'],
                    mode='lines+markers',
                    name=f"{aspect_short} (Volume)",
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate=f'<b>{aspect}</b><br>' +  # 전체 이름 표시
                                'Period: %{x}<br>' +
                                'Mentions: %{y}<br>' +
                                '<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Positive Rate", row=1, col=1, tickformat='.0%', range=[0, 1])
        fig.update_yaxes(title_text="Mention Count", row=2, col=1)
        fig.update_layout(height=600, hovermode='x unified', showlegend=True)
        
        # 중성선 추가
        fig.add_hline(y=0.5, row=1, col=1, line_dash="dot", line_color="gray", opacity=0.5, 
                     annotation_text="Neutral (50%)", annotation_position="bottom right")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 트렌드 요약
        if len(monthly_aspect_stats['period'].unique()) >= 3:
            st.subheader("📊 Trend Analysis Summary")
            
            trend_summaries = []
            for aspect in selected_aspects:
                aspect_data = monthly_aspect_stats[monthly_aspect_stats['aspect'] == aspect].sort_values('period')
                
                if len(aspect_data) >= 3:
                    # 안전한 트렌드 계산
                    recent_periods = min(3, len(aspect_data) // 2, len(aspect_data))
                    earlier_periods = min(3, len(aspect_data) // 2, len(aspect_data))
                    
                    if recent_periods > 0 and earlier_periods > 0:
                        recent_sentiment = aspect_data.tail(recent_periods)['sentiment'].mean()
                        earlier_sentiment = aspect_data.head(earlier_periods)['sentiment'].mean()
                        trend = recent_sentiment - earlier_sentiment
                        
                        trend_summaries.append({
                            'aspect': aspect,
                            'trend': trend,
                            'recent_sentiment': recent_sentiment,
                            'total_mentions': aspect_data['count'].sum(),
                            'periods': len(aspect_data)
                        })
            
            if trend_summaries:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Improving Trends")
                    improving = [t for t in trend_summaries if t['trend'] > 0.05]
                    if improving:
                        for trend in sorted(improving, key=lambda x: x['trend'], reverse=True):
                            st.success(f"**{trend['aspect'][:30]}{'...' if len(trend['aspect']) > 30 else ''}** (+{trend['trend']:.1%})")
                            st.caption(f"Recent: {trend['recent_sentiment']:.1%} • {trend['total_mentions']} mentions • {trend['periods']} periods")
                    else:
                        st.info("No significantly improving trends")
                
                with col2:
                    st.markdown("#### 📉 Declining Trends")
                    declining = [t for t in trend_summaries if t['trend'] < -0.05]
                    if declining:
                        for trend in sorted(declining, key=lambda x: x['trend']):
                            st.error(f"**{trend['aspect'][:30]}{'...' if len(trend['aspect']) > 30 else ''}** ({trend['trend']:.1%})")
                            st.caption(f"Recent: {trend['recent_sentiment']:.1%} • {trend['total_mentions']} mentions • {trend['periods']} periods")
                    else:
                        st.info("No significantly declining trends")
        
        # 데이터 요약
        st.subheader("📋 Analysis Summary")
        total_periods = len(monthly_aspect_stats['period'].unique())
        total_mentions = monthly_aspect_stats['count'].sum()
        avg_sentiment = monthly_aspect_stats['sentiment'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time Periods", total_periods)
        with col2:
            st.metric("Total Mentions", f"{total_mentions:,}")
        with col3:
            st.metric("Average Sentiment", f"{avg_sentiment:.1%}")
    
    except Exception as e:
        st.error(f"Error in timeline analysis: {str(e)}")
        st.info("This may be due to data format issues. Please check your aspect assignment data.")
        
        # 디버그 정보
        with st.expander("🔧 Debug Information"):
            st.write("Available columns:", list(sentences_df.columns))
            if 'assigned_aspects' in sentences_df.columns:
                sample_aspects = sentences_df['assigned_aspects'].dropna().head(3).tolist()
                st.write("Sample assigned_aspects:", sample_aspects)
            st.write("Data shape:", sentences_df.shape)
            st.write("Date range:", sentences_df['date'].min() if 'date' in sentences_df.columns else "No date column")

def get_aspect_examples_with_sentiment(sentences_df: pd.DataFrame, high_volume_aspects: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """각 aspect별로 긍정/부정 대표 문장들을 추출"""
    
    aspect_examples = {}
    
    for _, aspect_row in high_volume_aspects.iterrows():
        aspect_name = aspect_row['aspect']
        
        # 해당 aspect가 할당된 문장들 찾기
        if 'assigned_aspects' in sentences_df.columns:
            # assigned_aspects 컬럼에서 해당 aspect가 포함된 문장들 필터링
            aspect_sentences = sentences_df[
                sentences_df['assigned_aspects'].apply(
                    lambda x: aspect_name in (eval(x) if isinstance(x, str) and x.startswith('[') else [x]) if pd.notna(x) else False
                )
            ].copy()
        else:
            # primary_aspect 컬럼 사용 (fallback)
            aspect_sentences = sentences_df[
                sentences_df.get('primary_aspect', '') == aspect_name
            ].copy()
        
        if aspect_sentences.empty:
            continue
        
        # 긍정/부정으로 분리
        positive_sentences = aspect_sentences[aspect_sentences.get('sentiment', 0) == 1]
        negative_sentences = aspect_sentences[aspect_sentences.get('sentiment', 0) == 0]
        
        # 문장 길이와 품질 기준으로 필터링
        def filter_quality_sentences(df, min_words=8, max_words=40):
            if df.empty:
                return df
            
            # 단어 수 기준 필터링
            word_counts = df['sentence'].str.split().str.len()
            quality_mask = (word_counts >= min_words) & (word_counts <= max_words)
            
            # 기본적인 품질 필터 (너무 짧거나 반복적인 문장 제외)
            quality_sentences = df[quality_mask]
            
            # 문장 길이와 aspect 관련성으로 정렬 (aspect_scores가 있다면 활용)
            if 'max_aspect_score' in quality_sentences.columns:
                quality_sentences = quality_sentences.sort_values('max_aspect_score', ascending=False)
            elif 'sentiment_conf' in quality_sentences.columns:
                quality_sentences = quality_sentences.sort_values('sentiment_conf', ascending=False)
            else:
                # 문장 길이 기준으로 정렬 (너무 짧지도 길지도 않은 문장 선호)
                quality_sentences['sentence_length'] = quality_sentences['sentence'].str.len()
                ideal_length = 120  # 적당한 길이
                quality_sentences['length_score'] = 1 / (1 + abs(quality_sentences['sentence_length'] - ideal_length) / 50)
                quality_sentences = quality_sentences.sort_values('length_score', ascending=False)
            
            return quality_sentences
        
        # 품질 좋은 문장들 선별
        quality_positive = filter_quality_sentences(positive_sentences)
        quality_negative = filter_quality_sentences(negative_sentences)
        
        # 상위 문장들 선택 (최대 3개씩)
        positive_examples = quality_positive['sentence'].head(3).tolist()
        negative_examples = quality_negative['sentence'].head(3).tolist()
        
        # 문장 정리 (너무 길면 자르기)
        def clean_sentence(sentence, max_length=150):
            sentence = str(sentence).strip()
            if len(sentence) > max_length:
                sentence = sentence[:max_length-3] + "..."
            return sentence
        
        aspect_examples[aspect_name] = {
            'positive': [clean_sentence(s) for s in positive_examples],
            'negative': [clean_sentence(s) for s in negative_examples]
        }
    
    return aspect_examples


def generate_aspect_insights(summary_df: pd.DataFrame, sentences_df: pd.DataFrame, scan_dir: str = "./outputs_aspect_analysis") -> str:
    """Aspect 분석에 특화된 AI 인사이트 생성 - 실제 고객 목소리 포함"""
    
    API_KEY = os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
    if not API_KEY:
        return None
    
    # 기본 통계
    total_reviews = sentences_df['review_id'].nunique() if 'review_id' in sentences_df.columns else len(sentences_df)
    total_aspects = len(summary_df)
    avg_positive_rate = summary_df['positive_ratio'].mean() if 'positive_ratio' in summary_df.columns else 0
    
    # Top/Bottom aspects
    if not summary_df.empty:
        top_aspects = summary_df.nlargest(5, 'positive_ratio')
        bottom_aspects = summary_df.nsmallest(5, 'positive_ratio')
        high_volume_aspects = summary_df.nlargest(8, 'n_sentences')  # 더 많이 가져와서 예시 포함
    else:
        top_aspects = bottom_aspects = high_volume_aspects = pd.DataFrame()
    
    # 🔥 실제 고객 목소리 추출
    aspect_examples = get_aspect_examples_with_sentiment(sentences_df, high_volume_aspects)
    
    # 프롬프트 구성
    prompt = f"""
You are analyzing restaurant customer feedback using NLI-based aspect classification for 36 predefined restaurant aspects.

=== ANALYSIS SCOPE ===
- Total Reviews: {total_reviews:,}
- Aspects Identified: {total_aspects}/36 predefined restaurant aspects
- Overall Positive Rate: {avg_positive_rate:.1%}

=== ASPECT CATEGORIES ===
The 36 aspects are grouped into:
1. Service & Operations (9 aspects): wait time, service quality, order accuracy, etc.
2. Cleanliness & Safety (3 aspects): tableware, dining area, restroom cleanliness
3. Environment & Ambience (8 aspects): noise, lighting, decor, seating comfort
4. Accessibility & Family (4 aspects): parking, location, ADA compliance, family-friendly
5. Food & Menu (9 aspects): taste, freshness, portion size, value, allergen handling
6. Delivery & Takeout (3 aspects): online ordering, delivery time, packaging

=== TOP PERFORMING ASPECTS ===
"""
    
    if not top_aspects.empty:
        for _, aspect in top_aspects.iterrows():
            prompt += f"\n- **{aspect['aspect']}**: {aspect['positive_ratio']:.1%} positive ({aspect['n_sentences']} mentions)"
    
    prompt += "\n\n=== ASPECTS NEEDING ATTENTION ==="
    
    if not bottom_aspects.empty:
        for _, aspect in bottom_aspects.iterrows():
            prompt += f"\n- **{aspect['aspect']}**: {aspect['positive_ratio']:.1%} positive ({aspect['n_sentences']} mentions)"
    
    prompt += "\n\n=== MOST DISCUSSED ASPECTS WITH REAL CUSTOMER VOICES ==="
    
    # 🔥 각 주요 aspect별 실제 고객 목소리 추가
    for aspect_name, examples in aspect_examples.items():
        aspect_row = high_volume_aspects[high_volume_aspects['aspect'] == aspect_name]
        if not aspect_row.empty:
            aspect_stats = aspect_row.iloc[0]
            prompt += f"\n\n** {aspect_name} **"
            prompt += f"\nStatistics: {aspect_stats['positive_ratio']:.1%} positive | {aspect_stats['n_sentences']} mentions | {aspect_stats['avg_stars']:.1f} avg stars"
            
            if examples['positive']:
                prompt += f"\n\nPOSITIVE Customer Voices:"
                for i, sentence in enumerate(examples['positive'][:3], 1):
                    prompt += f"\n{i}. \"{sentence}\""
            
            if examples['negative']:
                prompt += f"\n\nNEGATIVE Customer Voices:"
                for i, sentence in enumerate(examples['negative'][:3], 1):
                    prompt += f"\n{i}. \"{sentence}\""
            
            prompt += "\n" + "="*50
    
    prompt += """

=== ANALYSIS REQUEST ===
Based on the statistical performance AND the real customer voices above, provide:

1. **Executive Summary** (3-4 key findings about restaurant performance based on both data and customer feedback)

2. **Category-Level Insights**
   - Which aspect categories are strongest/weakest based on customer voices?
   - What are customers specifically praising or complaining about?
   - Category-specific recommendations based on actual feedback themes

3. **Customer Voice Analysis**
   - What are the recurring themes in positive feedback?
   - What are the specific pain points customers mention most?
   - How do customer descriptions match or differ from the statistical ratings?

4. **Operational Priorities**
   - Top 3 aspects requiring immediate attention (backed by specific customer quotes)
   - Quick wins vs long-term improvements based on feedback patterns
   - Resource allocation recommendations with customer impact evidence

5. **Customer Experience Journey**
   - How do the aspects connect in the dining experience based on customer stories?
   - Critical moments that customers mention most frequently
   - Service recovery opportunities mentioned by customers

6. **Evidence-Based Action Plan**
   - Specific, implementable recommendations based on customer feedback
   - Success metrics for each aspect derived from customer expectations
   - Timeline suggestions prioritized by customer pain intensity

IMPORTANT: 
- Reference specific customer quotes to support each recommendation
- Identify patterns between what customers say vs statistical performance
- Focus on actionable insights that address the exact issues customers mention
- Distinguish between perception issues vs operational issues based on feedback tone

Focus on practical, evidence-based insights for restaurant management using both quantitative metrics and qualitative customer voices.
"""
    
    try:
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert restaurant operations consultant with deep knowledge of customer experience management and operational excellence. You excel at connecting quantitative performance data with qualitative customer feedback to provide actionable business insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000  # 더 긴 응답을 위해 증가
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# ========= Streamlit App =========

st.set_page_config(page_title="ReviewToRevenue: Restaurant Review Analysis Dashboard", page_icon="🍽️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🍽️ ReviewToRevenue: Restaurant Review Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📂 Data Source")
data_source = st.sidebar.selectbox(
    "Choose Data Source", 
    ["Load from existing outputs", "Run new pipeline"], 
    index=0
)

if data_source == "Run new pipeline":
    st.sidebar.subheader("🔍 Business Selection")
    
    # Business filtering
    col1, col2 = st.sidebar.columns(2)
    with col1:
        name_substr = st.text_input("Name contains", value="", placeholder="e.g., Pizza")
        city = st.text_input("City", value="", placeholder="e.g., Las Vegas")
    with col2:
        category = st.text_input("Category", value="Restaurants", placeholder="e.g., Restaurants")
        state = st.text_input("State", value="", placeholder="e.g., NV")
    
    limit = st.sidebar.number_input("Max businesses", min_value=1, max_value=50, value=5, step=1)
    
    # Manual business ID input
    biz_ids_manual = st.sidebar.text_area(
        "Or specify Business IDs", 
        value="", 
        placeholder="business_id_1\nbusiness_id_2",
        help="One per line"
    )
    
    st.sidebar.subheader("🎯 Aspect Analysis Settings")
    
    # Analysis scope
    topic_scope = st.sidebar.selectbox(
        "Analysis Scope", 
        ["per-store", "pooled", "both"], 
        index=0
    )
    
    # Quick test options
    quick_test = st.sidebar.checkbox("⚡ Quick Test Mode", value=True, help="Sample 100 sentences per business for faster testing")
    
    if quick_test:
        sample_size = st.sidebar.number_input("Sample size per business", min_value=50, max_value=500, value=100, step=50)
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        nli_model_name = st.selectbox(
            "NLI Model", 
            [
                "MoritzLaurer/DeBERTa-v3-base-mnlii",
                "microsoft/deberta-v3-large-mnli", 
                "microsoft/deberta-base-mnli",
                "facebook/bart-large-mnli",
                "roberta-large-mnli"
            ],
            index=0
        )
        
        min_prob = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
        batch_size = st.number_input("Batch size", min_value=4, max_value=32, value=8, step=4)
    
    # Run pipeline
    if st.sidebar.button("🚀 Run Aspect Analysis", type="primary"):
        with st.status("Running aspect analysis pipeline...", expanded=True) as status:
            try:
                # Business ID preparation
                if biz_ids_manual.strip():
                    biz_ids = [line.strip() for line in biz_ids_manual.strip().splitlines() if line.strip()]
                    st.write(f"Using manually specified business IDs: {len(biz_ids)} businesses")
                else:
                    biz_ids = find_business_ids(
                        business_json_path=str(PATH_BUSINESS),
                        name_substring=name_substr if name_substr else None,
                        category_keyword=category if category else "Restaurants",
                        city=city if city else None,
                        state=state if state else None,
                        limit=int(limit)
                    )
                    st.write(f"Found {len(biz_ids)} businesses")
                
                if not biz_ids:
                    st.error("No businesses found. Please adjust filters.")
                    st.stop()
                
                # Load business metadata
                id2meta = load_business_meta(str(PATH_BUSINESS), biz_ids)
                
                # Show selected businesses
                st.write("**Selected Businesses:**")
                for bid in biz_ids[:5]:
                    meta = id2meta.get(bid, {}) or {}
                    name = meta.get("name", "Unknown")
                    address = meta.get("address", "Unknown")
                    st.write(f"• {name} - {address}")
                if len(biz_ids) > 5:
                    st.write(f"... and {len(biz_ids) - 5} more")
                
                # Step 1: Filter reviews
                st.write("🔍 Filtering reviews...")
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
                
                # Quick test sampling
                sent_df = pd.read_csv(PATH_SENT_WITH_SENT)
                if quick_test:
                    original_size = len(sent_df)
                    sampled_dfs = []
                    for biz_id in sent_df['business_id'].unique():
                        biz_data = sent_df[sent_df['business_id'] == biz_id]
                        biz_sample_size = min(sample_size, len(biz_data))
                        biz_sample = biz_data.sample(n=biz_sample_size, random_state=42)
                        sampled_dfs.append(biz_sample)
                    
                    sent_df = pd.concat(sampled_dfs, ignore_index=True)
                    sent_df.to_csv(PATH_SENT_WITH_SENT, index=False)  # 샘플링된 데이터 저장
                    st.write(f"⚡ Quick test: sampled {len(sent_df):,} from {original_size:,} sentences")
                
                # Step 3: Aspect analysis
                st.write("🏷️ Analyzing restaurant aspects...")
                out_dir = Path("./outputs_aspect_analysis")
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # Run analysis
                def run_analysis(scope: str, business_id, out_path: Path):
                    return apply_nli_aspect_analysis(
                        sentences_csv_path=str(PATH_SENT_WITH_SENT),
                        out_csv_path=str(out_path),
                        business_id=business_id,
                        nli_model=nli_model_name,
                        min_prob=min_prob,
                        batch_size=batch_size,
                        local_files_only=True,
                        business_meta=id2meta
                    )
                
                # Execute analysis based on scope
                if topic_scope in ("pooled", "both"):
                    st.write("🌐 Running pooled analysis...")
                    pooled_path = out_dir / "aspect_analysis_pooled_all-stores.csv"
                    sentences_result, summary_result = run_analysis("pooled", None, pooled_path)
                    st.write(f"✅ Pooled analysis: {len(summary_result)} aspects found")
                
                if topic_scope in ("per-store", "both"):
                    st.write("🏪 Running per-store analysis...")
                    for i, bid in enumerate(biz_ids):
                        st.write(f"  Processing {i+1}/{len(biz_ids)}: {bid}")
                        store_path = out_dir / f"aspect_analysis_perstore_{bid}.csv"
                        try:
                            sentences_result, summary_result = run_analysis("per-store", bid, store_path)
                            st.write(f"    ✅ {len(summary_result)} aspects found")
                        except Exception as e:
                            st.write(f"    ❌ Error: {str(e)[:100]}")
                
                status.update(label="✅ Aspect analysis completed!", state="complete")
                
                # Auto-load results
                st.write("📊 Auto-loading results...")
                
                # Find generated files
                summary_files = list(out_dir.glob("*_aspect_summary.csv"))
                sentence_files = list(out_dir.glob("aspect_analysis_*.csv"))
                sentence_files = [f for f in sentence_files if not f.name.endswith('_aspect_summary.csv')]
                
                if summary_files and sentence_files:
                    # Load and combine
                    summary_dfs = [pd.read_csv(f) for f in summary_files]
                    sentence_dfs = [pd.read_csv(f) for f in sentence_files]
                    
                    summary_df = pd.concat(summary_dfs, ignore_index=True)
                    sentences_df = pd.concat(sentence_dfs, ignore_index=True)
                    
                    # Store in session state
                    st.session_state['summary_df'] = summary_df
                    st.session_state['sentences_df'] = sentences_df
                    st.session_state['data_loaded'] = True
                    st.session_state['scan_directory'] = str(out_dir)  # 출력 디렉토리 저장
                    
                    st.balloons()
                    st.success(f"🎉 Analysis complete! {len(summary_df)} aspects analyzed")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Output files not found")
                
            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
                st.exception(e)

else:
    # Load existing results
    st.sidebar.subheader("📁 Load Existing Results")
    
    scan_dir = st.sidebar.text_input("Results Directory", value="./outputs_aspect_analysis")
    
    if st.sidebar.button("🔍 Discover Files"):
        scan_path = Path(scan_dir)
        if scan_path.exists():
            summary_files = list(scan_path.glob("*_aspect_summary.csv"))
            sentence_files = list(scan_path.glob("aspect_analysis_*.csv"))
            sentence_files = [f for f in sentence_files if not f.name.endswith('_aspect_summary.csv')]
            
            if summary_files or sentence_files:
                st.sidebar.success(f"✅ Found {len(summary_files)} summary files, {len(sentence_files)} sentence files")
            else:
                st.sidebar.warning("⚠️ No aspect analysis files found")
        else:
            st.sidebar.error("❌ Directory does not exist")
    
    if st.sidebar.button("📊 Load Analysis Results", type="primary"):
        try:
            scan_path = Path(scan_dir)
            
            # Auto-discover files
            summary_files = list(scan_path.glob("*_aspect_summary.csv"))
            sentence_files = list(scan_path.glob("aspect_analysis_*.csv"))
            sentence_files = [f for f in sentence_files if not f.name.endswith('_aspect_summary.csv')]
            
            if not summary_files or not sentence_files:
                st.sidebar.error("❌ Required files not found")
                st.stop()
            
            # Load and combine
            summary_dfs = [pd.read_csv(f) for f in summary_files]
            sentence_dfs = [pd.read_csv(f) for f in sentence_files]
            
            summary_df = pd.concat(summary_dfs, ignore_index=True)
            sentences_df = pd.concat(sentence_dfs, ignore_index=True)
            
            # Store in session state
            st.session_state['summary_df'] = summary_df
            st.session_state['sentences_df'] = sentences_df
            st.session_state['data_loaded'] = True
            st.session_state['scan_directory'] = str(scan_path)  # scan directory 저장
            
            st.sidebar.success(f"✅ Loaded {len(summary_df)} aspects, {len(sentences_df)} sentences")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading: {str(e)}")

# Main dashboard
if st.session_state.get('data_loaded', False):
    summary_df = st.session_state['summary_df']
    sentences_df = st.session_state['sentences_df']
    
    # Header
    st.markdown("---")
    st.markdown('<h1 style="color: #e74c3c; text-align: center;">📊 Restaurant Aspect Analysis Results</h1>', unsafe_allow_html=True)
    
    # KPI Cards
    create_aspect_kpi_cards(summary_df, sentences_df)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🏗️ Categories", 
        "🎯 Priorities", 
        "📈 Trends",
        "🤖 AI Insights"
    ])
    
    # Overview Tab
    with tab1:
        create_aspect_performance_overview(summary_df)
        
        # Detailed table
        with st.expander("📋 Complete Aspect Summary", expanded=False):
            if not summary_df.empty:
                display_cols = ['aspect', 'n_sentences', 'positive_ratio', 'avg_stars']
                if 'share' in summary_df.columns:
                    display_cols.append('share')
                display_cols = [col for col in display_cols if col in summary_df.columns]
                
                st.dataframe(
                    summary_df[display_cols].sort_values('n_sentences', ascending=False),
                    use_container_width=True,
                    column_config={
                        "aspect": st.column_config.TextColumn("Restaurant Aspect", width="large"),
                        "n_sentences": st.column_config.NumberColumn("Mentions", width="small"),
                        "positive_ratio": st.column_config.NumberColumn("Positive Rate", format="%.1%", width="small"),
                        "avg_stars": st.column_config.NumberColumn("Avg Stars", format="%.1f", width="small"),
                        "share": st.column_config.NumberColumn("Share", format="%.1%", width="small") if "share" in display_cols else None
                    },
                    hide_index=True
                )
    
    # Categories Tab
    with tab2:
        create_aspect_category_analysis(summary_df, sentences_df)
    
    # Priorities Tab
    with tab3:
        create_aspect_priority_matrix(summary_df)
    
    # Trends Tab
    with tab4:
        create_aspect_timeline_analysis(sentences_df, summary_df)
    
    # AI Insights Tab
    with tab5:
        st.markdown('<h2 class="section-header">🤖 AI-Powered Restaurant Insights</h2>', unsafe_allow_html=True)
        
        API_KEY = os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
        
        if not API_KEY:
            st.warning("🔑 OpenAI API key not configured")
            with st.expander("How to configure OpenAI API key"):
                st.code("""
                # Method 1: Environment variable
                export OPENAI_API_KEY="your-api-key-here"
                
                # Method 2: Streamlit secrets
                # Create .streamlit/secrets.toml:
                OPENAI_API_KEY = "your-api-key-here"
                """)
        else:
            if st.button("🚀 Generate Restaurant Management Insights", type="primary"):
                with st.spinner("🧠 AI is analyzing your restaurant aspects and customer voices..."):
                    # scan_dir 정보를 세션에서 가져오거나 기본값 사용
                    scan_directory = st.session_state.get('scan_directory', './outputs_aspect_analysis')
                    insights = generate_aspect_insights(summary_df, sentences_df, scan_directory)
                    
                    if insights and not insights.startswith("Error"):
                        st.session_state['aspect_insights'] = insights
                        st.success("✅ AI insights generated!")
                    else:
                        st.error(f"❌ Failed: {insights}")
            
            if 'aspect_insights' in st.session_state:
                st.markdown("### 📋 Restaurant Management Analysis")
                st.download_button(
                    "📥 Download Report",
                    data=st.session_state['aspect_insights'],
                    file_name=f"restaurant_insights_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )
                st.markdown(st.session_state['aspect_insights'])
    
    # Export section
    st.markdown("---")
    st.markdown('<h2 class="section-header">📥 Export Results</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not summary_df.empty:
            summary_csv = summary_df.to_csv(index=False)
            st.download_button(
                "📊 Download Aspect Summary",
                data=summary_csv,
                file_name=f"aspect_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if not sentences_df.empty:
            sentences_sample = sentences_df.head(1000)
            sentences_csv = sentences_sample.to_csv(index=False)
            st.download_button(
                "📝 Download Sentences (Sample)",
                data=sentences_csv,
                file_name=f"sentences_sample_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if 'aspect_insights' in st.session_state:
            st.download_button(
                "🤖 Download AI Insights",
                data=st.session_state['aspect_insights'],
                file_name=f"ai_insights_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); border-radius: 10px; color: white; margin: 2rem 0;'>
        <h2>🍽️ Welcome to ReviewToRevenue: Restaurant Review Analysis Dashboard</h2>
        <p style='font-size: 1.2rem; margin: 1rem 0;'>
            Analyze customer feedback across 36 predefined restaurant aspects using advanced NLI classification
        </p>
        <p>👈 Use the sidebar to load results or run a new analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🏷️ 36 Restaurant Aspects
        - Service & Operations (9 aspects)
        - Food & Menu (9 aspects)  
        - Environment & Ambience (8 aspects)
        - Accessibility & Family (4 aspects)
        - Cleanliness & Safety (3 aspects)
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Advanced Analysis
        - NLI-based classification
        - Performance tracking
        - Priority identification
        - Trend analysis
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Management Insights
        - Category-level analysis
        - Operational priorities
        - Customer experience mapping
        - Actionable recommendations
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>ReviewToRevenue: Restaurant Review Analysis Dashboard | "
    "Powered by NLI Classification & Streamlit</div>", 
    unsafe_allow_html=True
)