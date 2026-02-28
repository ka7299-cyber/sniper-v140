import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import requests
import time

# 頁面配置
st.set_page_config(page_title="Sniper X V140 (Final)", layout="wide")

# ==============================================
# 1. 資料庫設定
# ==============================================
US_STRATEGIES = {
    'NVDA': (19, 58), 'MSFT': (21, 53), 'TSLA': (17, 58), 'GOOGL': (26, 55),
    'AMZN': (19, 47), 'META': (24, 76), 'AAPL': (19, 58), 'TSM': (19, 64),
    'INTC': (27, None), 'AMD': (22, 96), 'ADBE': (25, 63), 'ASML': (24, 51),
    'QCOM': (25, 64), 'NFLX': (23, 65), 'COST': (18, 56), 'MA': (33, None),
    'V': (22, 56), 'HD': (17, 53), 'ZTS': (28, 56), 'TTD': (23, None),
    'JNJ': (26, None), 'IBM': (19, None), 'AVGO': (24, None),
    '^DJI': (20, 45), '^GSPC': (19, 55), '^RUT': (22, 56), '^IXIC': (20, None), '^SOX': (20, None),
    'UNH': (26, 59), 'ULTA': (26, None), 'AMG': (22, None), 'AJG': (23, None),
    'BKNG': (23, None), 'NVO': (26, 57), 'IBP': (20, None), 'PAYC': (20, None),
    'URI': (22, None), 'GIB': (21, None), 'CTAS': (19, None), 'CHE': (24, None)
}
US_NAMES = {'NVDA': '輝達', 'MSFT': '微軟', 'TSLA': '特斯拉', 'GOOGL': '谷歌', 'AMZN': '亞馬遜', 'META': '臉書', 'AAPL': '蘋果', 'TSM': '台積電ADR', 'AMD': '超微', 'ADBE': 'Adobe', 'ASML': '艾司摩爾', 'QCOM': '高通', 'NFLX': '奈飛', 'COST': '好市多', 'UNH': '聯合健康', 'NVO': '諾和諾德', 'AVGO': '博通'}

TW_STRATEGIES = {'2330': (17, 57), '2317': (18, 57), '2382': (23, 60), '2357': (21, 57), '2454': (29, 60), '2603': (35, 60), '3081': (20, 60), '3264': (18, 57)}
TW_NAMES = {'2330':'台積電', '2317':'鴻海', '2454':'聯發科', '3081':'聯亞', '2382':'廣達'}

# ==============================================
# 2. 核心 AI 演算法 & 強力連線
# ==============================================

@st.cache_data(ttl=600)
def fetch_data_robust(ticker_symbol):
    # 策略 A: 一般連線
    try:
        df = yf.Ticker(ticker_symbol).history(period="2y")
        if not df.empty: return df
    except: pass
    
    # 策略 B: 偽裝瀏覽器 (針對 Yahoo 擋擋牆)
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})
        df = yf.Ticker(ticker_symbol, session=session).history(period="2y")
        if not df.empty: return df
    except: pass
    
    return pd.DataFrame()

def find_best_ma_golden_bluff_v2(df, start_day, end_day):
    closes = df['Close'].values; lows = df['Low'].values; highs = df['High'].values
    n = len(df)
    best_ma = start_day; best_score = -np.inf
    fib_nums = {21, 34, 55, 89}

    for ma_len in range(start_day, end_day + 1):
        ma_series = df['Close'].rolling(window=ma_len).mean()
        ma_values = ma_series.values
        if n < ma_len + 10: continue 
        
        valid_idx = slice(ma_len, n)
        l_slice = lows[valid_idx]; h_slice = highs[valid_idx]; ma_slice = ma_values[valid_idx]

        min_idxs = argrelextrema(l_slice, np.less, order=3)[0]
        max_idxs = argrelextrema(h_slice, np.greater, order=3)[0]

        total_error = 0; point_count = 0
        if len(min_idxs) > 0:
            diffs = np.abs(l_slice[min_idxs] - ma_slice[min_idxs]) / ma_slice[min_idxs]
            total_error += diffs.sum()
            point_count += len(min_idxs)
        if len(max_idxs) > 0:
            diffs = np.abs(h_slice[max_idxs] - ma_slice[max_idxs]) / ma_slice[max_idxs]
            total_error += diffs.sum()
            point_count += len(max_idxs)
            
        avg_error = (total_error / point_count) if point_count > 0 else 0.05
        score = 100 - (avg_error * 3000)
        
        cross_mask = (closes[valid_idx] > ma_slice) ^ (np.roll(closes[valid_idx], 1) > np.roll(ma_slice, 1))
        crosses_per_month = np.sum(cross_mask[1:]) / (len(ma_slice) / 20.0)
        if crosses_per_month > 3.0: score -= 100
        if ma_len in fib_nums: score += 10
            
        if score > best_score: best_score = score; best_ma = ma_len
    return best_ma

def backtest_stats(df, ma_days):
    ma = df['Close'].rolling(window=ma_days).mean()
    signals = (df['Close'] > ma).astype(int)
    actions = signals.diff()
    wins = 0; total = 0; holding = False; entry = 0
    for i in range(1, len(df)):
        p = df['Close'].iloc[i]
        if actions.iloc[i] == 1 and not holding: entry = p; holding = True
        elif actions.iloc[i] == -1 and holding:
            if p > entry: wins += 1
            total += 1; holding = False
    return (wins / total * 100) if total > 0 else 0, total

# ==============================================
# 3. 介面與顯示
# ==============================================
st.sidebar.header("🕹️ Sniper X (V140)")
market_mode = st.sidebar.radio("市場", ["🇹🇼 台股", "🇺🇸 美股"], horizontal=True)

if "🇹🇼" in market_mode:
    curr_strat, curr_names = TW_STRATEGIES, TW_NAMES
    d_list = ['2330', '2317', '3081']
else:
    curr_strat, curr_names = US_STRATEGIES, US_NAMES
    d_list = ['NVDA', 'TSLA', 'AMD']

state_key = "list_" + market_mode
if state_key not in st.session_state: st.session_state[state_key] = d_list

def add_stock():
    v = st.session_state.new_in.strip().upper()
    if v and v not in st.session_state[state_key]: st.session_state[state_key].append(v)
    st.session_state.new_in = ""

st.sidebar.text_input("輸入代號", key="new_in", on_change=add_stock)
sel_list = st.sidebar.multiselect("清單管理", st.session_state[state_key], st.session_state[state_key])
st.session_state[state_key] = sel_list

stock_id = st.sidebar.selectbox("分析目標", sel_list) if sel_list else None
k_days = st.sidebar.select_slider("顯示K棒", options=[30, 60, 120, 240], value=60)

if stock_id:
    t_symbol = f"{stock_id}.TW" if "🇹🇼" in market_mode else stock_id
    df = fetch_data_robust(t_symbol)
    if df.empty and "🇹🇼" in market_mode: 
        t_symbol = f"{stock_id}.TWO"
        df = fetch_data_robust(t_symbol)

    if not df.empty:
        p_short, p_long = curr_strat.get(stock_id, (None, None))
        
        with st.spinner('🎯 正在運算 (強力連線中)...'):
            final_s = p_short if p_short else find_best_ma_golden_bluff_v2(df, 16, 25)
            final_l = p_long if p_long else find_best_ma_golden_bluff_v2(df, 45, 70)
        
        s_win, s_cnt = backtest_stats(df, final_s)
        
        st.sidebar.markdown("---")
        source = "👑 大師鎖定" if stock_id in curr_strat else "🤖 AI 強化演算"
        st.sidebar.info(f"{source}\n\n短線: {final_s} MA (勝率{s_win:.0f}%)\n長線: {final_l} MA")

        df['MS'] = df['Close'].rolling(window=final_s).mean()
        df['ML'] = df['Close'].rolling(window=final_l).mean()
        df['V5'] = df['Volume'].rolling(window=5).mean()
        
        p_df = df.tail(k_days).copy()
        # 無縫K線設定
        p_df.index = p_df.index.strftime('%Y-%m-%d')
        
        last_c = p_df['Close'].iloc[-1]
        bias = (last_c - p_df['MS'].iloc[-1]) / p_df['MS'].iloc[-1] * 100
        
        st.subheader(f"📊 {curr_names.get(stock_id, stock_id)} ({t_symbol})")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{last_c:.2f}")
        c2.metric(f"短({final_s})", f"{p_df['MS'].iloc[-1]:.2f}", f"{bias:+.1f}%")
        c3.metric(f"長({final_l})", f"{p_df['ML'].iloc[-1]:.2f}")
        
        trend = "🔥 強勢多頭" if last_c > p_df['MS'].iloc[-1] > p_df['ML'].iloc[-1] else "📈 區間偏多" if last_c > p_df['MS'].iloc[-1] else "❄️ 絕對空頭"
        c4.metric("戰情判定", trend)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_width=[0.3, 0.7])
        fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K棒', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'), row=1, col=1)
        fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MS'], name='短線', line=dict(color='#ff9800', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=p_df.index, y=p_df['ML'], name='長線', line=dict(color='#9c27b0', width=2)), row=1, col=1)
        
        v_cols = ['#ef5350' if c >= o else '#26a69a' for c, o in zip(p_df['Close'], p_df['Open'])]
        fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], marker_color=v_cols, name='量'), row=2, col=1)
        fig.add_trace(go.Scatter(x=p_df.index, y=p_df['V5'], line=dict(color='#29b6f6', width=1), name='5MA量'), row=2, col=1)

        fig.update_layout(height=400, template="plotly_white", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=0,r=10,t=5,b=0), hovermode="x unified", dragmode=False)
        fig.update_xaxes(fixedrange=True, type='category', nticks=6)
        fig.update_yaxes(side="right", fixedrange=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.error(f"⚠️ 連線受阻。請嘗試點擊右下角 'Manage app' > 'Clear cache' 後重試。")
