
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import json
import math
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional

# -------------------------------
# Helpers: Column resolution
# -------------------------------
CANDIDATE_COLS = {
    "currency": ["currency", "ccy", "curve_ccy", "ccy_leg", "pair", "tenor_ccy"],
    "trade_date": ["trade_date", "date", "tradedate", "execution_date", "exec_date", "trade time", "created at", "trade time"],
    "notional": ["notional", "size", "quantity", "qty", "nominal", "amount", "notional_amount", "notionals"],
    "dv01": ["dv01", "dv", "dollar_value", "dollar_value_of_01"],
    "maturity": ["maturity", "maturity_date", "end_date", "expiry", "expiration_date", "expiration date"],
    "effective_date": ["effective_date", "start_date", "effective date", "start date"],
    "effective_bucket": ["effective_bucket", "effective bucket", "start_bucket", "start bucket"],
    "expiration_bucket": ["expiration_bucket", "expiration bucket", "end_bucket", "end bucket", "maturity_bucket"],
    "tenor": ["tenor", "term", "tenor_yrs", "tenor_years"],
    "side": ["side", "buy_sell", "direction"],
}

def find_col(df: pd.DataFrame, logical: str) -> Optional[str]:
    cols = {c.lower().strip(): c for c in df.columns}
    for name in CANDIDATE_COLS.get(logical, []):
        if name in cols:
            return cols[name]
    # fuzzy fallback: startswith / contains
    for k, v in cols.items():
        if logical in k:
            return v
    return None

def parse_date(s: Any) -> Optional[pd.Timestamp]:
    if pd.isna(s):
        return None
    try:
        return pd.to_datetime(s, errors="coerce", utc=False).normalize()
    except Exception:
        return None

# -------------------------------
# Bucketing logic
# -------------------------------
def years_between(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    return (d1 - d0).days / 365.25

def bucket_tenor(years: float) -> str:
    # Coarse buckets similar to your narrative
    if years < 0.75:
        return "0-9M"
    if years < 2:
        return "0-2Y"
    if years < 3:
        return "2-3Y"
    if years < 5:
        return "3-5Y"
    if years < 7:
        return "5-7Y"
    if years < 10:
        return "7-10Y"
    if years < 15:
        return "10-15Y"
    if years < 20:
        return "15-20Y"
    if years <= 30:
        return "20-30Y"
    return "30Y+"

# -------------------------------
# Stats per currency for a single date
# -------------------------------
def summarize_currency(df: pd.DataFrame,
                       ccy_col: str,
                       notion_col: str,
                       trade_date: pd.Timestamp,
                       dv01_col: Optional[str] = None,
                       maturity_col: Optional[str] = None,
                       tenor_col: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["trade_date"] = trade_date.strftime("%Y-%m-%d")
    out["n_trades"] = int(df.shape[0])
    total_notional = float(df[notion_col].astype(float).abs().sum())
    out["total_notional"] = total_notional
    out["avg_trade_size"] = float(total_notional / max(1, out["n_trades"]))
    
    # Calculate DV01 stats if DV01 column is available
    if dv01_col and dv01_col in df.columns:
        total_dv01 = float(df[dv01_col].astype(float).abs().sum())
        out["total_dv01"] = total_dv01
        out["avg_dv01_per_trade"] = float(total_dv01 / max(1, out["n_trades"]))
    else:
        out["total_dv01"] = 0.0
        out["avg_dv01_per_trade"] = 0.0

    # Tenor years
    tenor_years_list: List[float] = []
    tenor_bucket_sum: Dict[str, float] = defaultdict(float)
    today = trade_date

    if tenor_col and tenor_col in df.columns:
        # Try to read tenor if numeric years, else parse like "5Y", "6M"
        def parse_tenor_val(x):
            if pd.isna(x):
                return None
            if isinstance(x, (int, float)) and not pd.isna(x):
                return float(x)
            s = str(x).strip().upper()
            try:
                if s.endswith("Y"):
                    return float(s[:-1])
                if s.endswith("M"):
                    return float(s[:-1]) / 12.0
                # fallback
                return float(s)
            except Exception:
                return None
        years_vals = df[tenor_col].apply(parse_tenor_val)
        tenor_years_list = [y for y in years_vals.tolist() if y is not None]
        for y, noz in zip(years_vals.fillna(0.0), df[notion_col].abs()):
            if y and y > 0:
                b = bucket_tenor(y)
                tenor_bucket_sum[b] += float(noz)
        
        # Also calculate DV01-based tenor buckets if DV01 column is available
        if dv01_col and dv01_col in df.columns:
            tenor_dv01_sum: Dict[str, float] = defaultdict(float)
            for y, dv01_val in zip(years_vals.fillna(0.0), df[dv01_col].abs()):
                if y and y > 0:
                    b = bucket_tenor(y)
                    tenor_dv01_sum[b] += float(dv01_val)
            out["tenor_dv01_buckets"] = dict(tenor_dv01_sum)
    elif maturity_col and maturity_col in df.columns:
        # compute tenor as (maturity - trade_date)
        mat_dates = df[maturity_col].apply(parse_date)
        for md, noz in zip(mat_dates, df[notion_col].abs()):
            if isinstance(md, pd.Timestamp):
                y = years_between(trade_date, md)
                tenor_years_list.append(y)
                b = bucket_tenor(y)
                tenor_bucket_sum[b] += float(noz)
        
        # Also calculate DV01-based tenor buckets from maturity if DV01 column is available
        if dv01_col and dv01_col in df.columns:
            tenor_dv01_sum: Dict[str, float] = defaultdict(float)
            for md, dv01_val in zip(mat_dates, df[dv01_col].abs()):
                if isinstance(md, pd.Timestamp):
                    y = years_between(trade_date, md)
                    b = bucket_tenor(y)
                    tenor_dv01_sum[b] += float(dv01_val)
            out["tenor_dv01_buckets"] = dict(tenor_dv01_sum)

    # Most active bucket(s) - prefer DV01 if available
    if out.get("tenor_dv01_buckets"):
        # sort by DV01 desc
        bucket_sorted = sorted(out["tenor_dv01_buckets"].items(), key=lambda kv: kv[1], reverse=True)
        out["top_buckets"] = [{"bucket": b, "dv01": v} for b, v in bucket_sorted[:5]]
    elif tenor_bucket_sum:
        # sort by notional desc
        bucket_sorted = sorted(tenor_bucket_sum.items(), key=lambda kv: kv[1], reverse=True)
        out["top_buckets"] = [{"bucket": b, "notional": v} for b, v in bucket_sorted[:5]]
    else:
        out["top_buckets"] = []

    # Trade structure detection - group by effective and expiration buckets
    structure_details: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "dv01": 0.0, "notional": 0.0})
    
    # Check if we have bucket columns (preferred) or date columns (fallback)
    effective_bucket_col = find_col(df, "effective_bucket")
    expiration_bucket_col = find_col(df, "expiration_bucket")
    
    if effective_bucket_col and expiration_bucket_col and effective_bucket_col in df.columns and expiration_bucket_col in df.columns:
        # Use bucket columns (preferred)
        for idx, row in df.iterrows():
            effective_bucket = str(row[effective_bucket_col]).strip() if pd.notna(row[effective_bucket_col]) else ""
            expiration_bucket = str(row[expiration_bucket_col]).strip() if pd.notna(row[expiration_bucket_col]) else ""
            
            if effective_bucket and expiration_bucket:
                # Create structure key using buckets
                structure_key = f"{effective_bucket} → {expiration_bucket}"
                
                # Calculate DV01 and notional for this structure
                dv01_val = float(row[dv01_col]) if dv01_col and dv01_col in df.columns else 0.0
                notional_val = float(row[notion_col]) if notion_col and notion_col in df.columns else 0.0
                
                structure_details[structure_key]["count"] += 1
                structure_details[structure_key]["dv01"] += abs(dv01_val)
                structure_details[structure_key]["notional"] += abs(notional_val)
    else:
        # Fallback to date columns if bucket columns not available
        start_date_col = find_col(df, "effective_date") or find_col(df, "start_date") 
        end_date_col = find_col(df, "expiration_date") or find_col(df, "maturity")
        
        if start_date_col and end_date_col and start_date_col in df.columns and end_date_col in df.columns:
            for idx, row in df.iterrows():
                start_date = parse_date(row[start_date_col])
                end_date = parse_date(row[end_date_col])
                
                if isinstance(start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp):
                    # Create structure key
                    structure_key = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    
                    # Calculate DV01 and notional for this structure
                    dv01_val = float(row[dv01_col]) if dv01_col and dv01_col in df.columns else 0.0
                    notional_val = float(row[notion_col]) if notion_col and notion_col in df.columns else 0.0
                    
                    structure_details[structure_key]["count"] += 1
                    structure_details[structure_key]["dv01"] += abs(dv01_val)
                    structure_details[structure_key]["notional"] += abs(notional_val)
    
    if structure_details:
        # Sort structures by DV01 if available, otherwise by notional
        if dv01_col and dv01_col in df.columns:
            sorted_structures = sorted(structure_details.items(), 
                                     key=lambda kv: kv[1]["dv01"], reverse=True)
            out["trade_structures"] = [
                {"structure": k, "count": v["count"], "dv01": v["dv01"]} 
                for k, v in sorted_structures[:3]
            ]
        else:
            sorted_structures = sorted(structure_details.items(), 
                                     key=lambda kv: kv[1]["notional"], reverse=True)
            out["trade_structures"] = [
                {"structure": k, "count": v["count"], "notional": v["notional"]} 
                for k, v in sorted_structures[:3]
            ]
    else:
        out["trade_structures"] = []

    # Maturity-year clusters (if maturity provided) - use DV01 if available
    year_counts: Dict[int, float] = defaultdict(float)
    if maturity_col and maturity_col in df.columns:
        if dv01_col and dv01_col in df.columns:
            # Use DV01 for maturity clusters
            for md, dv01_val in zip(df[maturity_col].apply(parse_date), df[dv01_col].abs()):
                if isinstance(md, pd.Timestamp):
                    year_counts[md.year] += float(dv01_val)
            top_years = sorted(year_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            out["maturity_clusters"] = [{"year": int(y), "dv01": v} for y, v in top_years]
        else:
            # Fallback to notional only if DV01 not available
            for md, noz in zip(df[maturity_col].apply(parse_date), df[notion_col].abs()):
                if isinstance(md, pd.Timestamp):
                    year_counts[md.year] += float(noz)
            top_years = sorted(year_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            out["maturity_clusters"] = [{"year": int(y), "notional": v} for y, v in top_years]
    else:
        out["maturity_clusters"] = []

    return out

def format_dv01_k(value):
    """Format DV01 value to nearest thousands/millions with K/MM suffix"""
    if value == 0:
        return "0"
    
    # Round to nearest thousand
    rounded = round(value / 1000) * 1000
    
    if rounded >= 1_000_000:
        # Format as millions with 3 decimal places
        millions = rounded / 1_000_000
        return f"{millions:.3f}MM"
    elif rounded >= 1000:
        # Format as thousands
        return f"{int(rounded/1000)}k"
    else:
        # Less than 1000
        return f"{int(rounded)}"

def facts_to_bullets(ccy: str, stats: Dict[str, Any], biggest_structure: str = "") -> str:
    bullets = []
    bullets.append(f"Currency: {ccy}")
    bullets.append(f"Trade date: {stats.get('trade_date','')}")
    bullets.append(f"Trades: {stats.get('n_trades',0):,}")
    
    # Highlight biggest structure for this currency at the top
    if biggest_structure:
        bullets.append(f"BIGGEST STRUCTURE: {biggest_structure}")
    
    # Use DV01 if available, otherwise fall back to notional
    if stats.get('total_dv01', 0) > 0:
        total_dv01_k = format_dv01_k(stats.get('total_dv01', 0))
        avg_dv01_k = format_dv01_k(stats.get('avg_dv01_per_trade', 0))
        bullets.append(f"Total DV01: ~{total_dv01_k} USD")
        bullets.append(f"Average DV01 per trade: ~{avg_dv01_k} USD")
    else:
        bullets.append(f"Total notional: ~{stats.get('total_notional',0):,.0f}")
        bullets.append(f"Average trade size: ~{stats.get('avg_trade_size',0):,.0f}")
    
    if stats.get("top_buckets"):
        if stats.get('total_dv01', 0) > 0 and 'dv01' in stats["top_buckets"][0]:
            # DV01-based buckets
            tops = ", ".join([f"{b['bucket']} (~{format_dv01_k(b['dv01'])} DV01)" for b in stats["top_buckets"][:3]])
            bullets.append(f"Top tenor buckets by DV01: {tops}")
        elif 'notional' in stats["top_buckets"][0]:
            # Notional-based buckets
            tops = ", ".join([f"{b['bucket']} (~{b['notional']:,.0f})" for b in stats["top_buckets"][:3]])
            bullets.append(f"Top tenor buckets by notional: {tops}")
    
    # Trade structures (most traded by DV01)
    if stats.get("trade_structures"):
        if stats.get('total_dv01', 0) > 0 and 'dv01' in stats["trade_structures"][0]:
            # Use DV01 for trade structures - filter out zero DV01
            structures = ", ".join([f"{x['structure']} ({x['count']} trades, ~{format_dv01_k(x['dv01'])} DV01)" for x in stats["trade_structures"][:2] if x['dv01'] > 0])
            if structures:
                bullets.append(f"Most traded structures (by DV01): {structures}")
        elif 'notional' in stats["trade_structures"][0]:
            # Fallback to notional only if DV01 not available
            structures = ", ".join([f"{x['structure']} ({x['count']} trades, ~{x['notional']:,.0f})" for x in stats["trade_structures"][:2]])
            bullets.append(f"Most traded structures (by notional): {structures}")
    
    if stats.get("maturity_clusters"):
        if stats.get('total_dv01', 0) > 0 and 'dv01' in stats["maturity_clusters"][0]:
            # Use DV01 for maturity clusters - filter out zero DV01
            yrs = ", ".join([f"{x['year']} (~{format_dv01_k(x['dv01'])} DV01)" for x in stats["maturity_clusters"] if x['dv01'] > 0])
            if yrs:
                bullets.append(f"Maturity clusters (by DV01): {yrs}")
        elif 'notional' in stats["maturity_clusters"][0]:
            # Fallback to notional only if DV01 not available
            yrs = ", ".join([f"{x['year']} (~{x['notional']:,.0f})" for x in stats["maturity_clusters"]])
            bullets.append(f"Maturity clusters (by notional): {yrs}")
    
    return "\n".join(f"- {x}" for x in bullets)

# -------------------------------
# LLM prompt
# -------------------------------
SYSTEM_PROMPT = (
    "You are a sell-side rates strategist writing factual commentary on daily swap activity by currency.\n"
    "STRICT CONSTRAINTS:\n"
    "- EXACTLY 2 sentences maximum. NO MORE.\n"
    "- MAXIMUM 100 words total. Count words carefully.\n"
    "- Write as a single continuous block of text for easy copy-paste.\n"
    "CRITICAL STYLE REQUIREMENTS:\n"
    "- Start with the biggest structure and describe what happened.\n"
    "- State WHERE activity concentrates on the curve - be factual only.\n"
    "- Focus on DV01 metrics and trade counts - report facts only.\n"
    "- NO opinions, interpretations, or strategic insights.\n"
    "- NO mention of 'suggests', 'indicating', 'possibly', 'potentially', 'likely', 'strategic positioning', 'relative value', 'liability management'.\n"
    "- Be purely observational - report what happened, not what it means.\n"
    "Tone: Factual, observational, concise. Avoid analysis or interpretation.\n"
)

USER_PROMPT_TEMPLATE = (
    "Write factual commentary for {ccy} using these facts:\n\n"
    "{facts}\n\n"
    "REQUIREMENTS: Write as a single continuous block of text (no line breaks, no bullets, no paragraphs). EXACTLY 2 sentences maximum. MAXIMUM 100 words total. Start with the BIGGEST STRUCTURE and report what happened. State where activity concentrated. Report DV01 amounts and trade counts. Be purely observational - NO analysis, interpretations, or strategic insights.\n"
)

# -------------------------------
# OpenAI client (lazy import)
# -------------------------------
def call_openai(messages, model="gpt-4o", temperature=0.1, max_tokens=150):
    """Requires OPENAI_API_KEY in environment."""
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in environment")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def enforce_sentence_limit(text, max_sentences=2, max_words=100):
    """Enforce maximum sentence limit and make copy-pasteable"""
    # Remove verbose phrases
    text = text.replace("clustering of maturities", "maturities")
    text = text.replace("systematic approach", "activity")
    text = text.replace("likely driven by hedging or asset-liability management strategies", "hedging activity")
    text = text.replace("suggesting a focus on", "indicating")
    text = text.replace("potentially indicating", "indicating")
    text = text.replace("This pattern suggests", "Activity suggests")
    text = text.replace("The concentration of trades", "Trades concentrate")
    text = text.replace("The prevalence of trades", "Trades show")
    text = text.replace("The systematic nature of these trades", "These trades")
    
    # Split by sentence endings
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in '.!?' and len(current_sentence.strip()) > 10:  # Avoid splitting on abbreviations
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # Add remaining text if any
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # Take only the first max_sentences
    sentences = sentences[:max_sentences]
    
    # Join with spaces for single line output
    result = ' '.join(sentences)
    
    # Enforce word limit
    words = result.split()
    if len(words) > max_words:
        result = ' '.join(words[:max_words])
    
    # Ensure it ends with a period if it doesn't already
    if result and not result.rstrip().endswith(('.', '!', '?')):
        result = result.rstrip() + '.'
    
    return result

# -------------------------------
# Main pipeline
# -------------------------------
def main():
    p = argparse.ArgumentParser(description="Generate daily swap commentary by currency from a trades CSV.")
    p.add_argument("csv_path", help="Path to trades CSV (e.g., dtcc_trades.csv)")
    p.add_argument("--out_csv", default="daily_commentary.csv", help="Output CSV path")
    p.add_argument("--out_md", default="daily_commentary.md", help="Output Markdown path")
    p.add_argument("--date", default=None, help="Trade date to filter (YYYY-MM-DD). Defaults to latest date in file.")
    p.add_argument("--model", default="gpt-4o", help="OpenAI chat model name (default: gpt-4o)")
    p.add_argument("--include_yesterday", action="store_true", help="Include yesterday's day-end commentary")
    p.add_argument("--yesterday_only", action="store_true", help="Generate only yesterday's commentary")
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    # Resolve columns
    ccy_col = find_col(df, "currency")
    td_col  = find_col(df, "trade_date")
    noz_col = find_col(df, "notional")
    dv01_col = find_col(df, "dv01")
    mat_col = find_col(df, "maturity")
    ten_col = find_col(df, "tenor")

    missing = [name for name, col in [("currency", ccy_col), ("trade_date", td_col), ("notional", noz_col)] if col is None]
    if missing:
        raise ValueError(f"Missing required columns (or aliases): {missing}. Found columns = {list(df.columns)}")

    # Parse dates
    df["_trade_date"] = df[td_col].apply(parse_date)
    if args.date:
        target = pd.to_datetime(args.date).normalize()
    else:
        target = df["_trade_date"].dropna().max()
        if pd.isna(target):
            raise ValueError("Could not infer a valid trade date from the file; please pass --date YYYY-MM-DD")

    # Filter to the target trade date
    day_mask = df["_trade_date"] == target
    day_df = df.loc[day_mask].copy()
    if day_df.empty:
        # loosen: match by date string
        alt_mask = df["_trade_date"].dt.strftime("%Y-%m-%d") == target.strftime("%Y-%m-%d")
        day_df = df.loc[alt_mask].copy()
    if day_df.empty:
        raise ValueError(f"No rows for trade date {target.date()}")

    # Clean numeric notional and DV01
    def to_float(x):
        try:
            s = str(x).replace(",", "")
            return float(s)
        except Exception:
            return math.nan
    day_df[noz_col] = day_df[noz_col].apply(to_float)
    day_df = day_df.dropna(subset=[noz_col])
    
    # Clean DV01 column if it exists
    if dv01_col and dv01_col in day_df.columns:
        day_df[dv01_col] = day_df[dv01_col].apply(to_float)

    # Top 5 individual trades by DV01 from the last hour
    top_individual_trades = []
    
    if dv01_col and dv01_col in day_df.columns:
        # Find trade time column
        trade_time_col = None
        for col in day_df.columns:
            if col.lower() in ['trade time', 'trade_time', 'created at', 'timestamp']:
                trade_time_col = col
                break
        
        if not trade_time_col:
            # Try finding by pattern
            for col in day_df.columns:
                if 'time' in col.lower() or 'created' in col.lower():
                    trade_time_col = col
                    break
        
        # Filter trades from last hour if trade time column exists
        last_hour_df = day_df
        if trade_time_col and trade_time_col in day_df.columns:
            try:
                # Parse trade times
                day_df['_trade_time'] = pd.to_datetime(day_df[trade_time_col], errors='coerce', utc=True)
                if not day_df['_trade_time'].isna().all():
                    # Get the latest trade time
                    latest_time = day_df['_trade_time'].max()
                    if pd.notna(latest_time):
                        # Filter to last hour
                        one_hour_ago = latest_time - pd.Timedelta(hours=1)
                        last_hour_df = day_df[day_df['_trade_time'] >= one_hour_ago].copy()
            except Exception as e:
                print(f"Warning: Could not filter by trade time: {e}")
                last_hour_df = day_df
        
        # Create list of individual trades with DV01 from last hour
        trades_with_dv01 = []
        effective_bucket_col = find_col(last_hour_df, "effective_bucket")
        expiration_bucket_col = find_col(last_hour_df, "expiration_bucket")
        
        for idx, row in last_hour_df.iterrows():
            dv01_val = float(row[dv01_col]) if pd.notna(row[dv01_col]) else 0.0
            currency = str(row[ccy_col]).strip()
            
            # Skip trades with zero DV01
            if abs(dv01_val) <= 0:
                continue
            
            # Create structure description
            if effective_bucket_col and expiration_bucket_col and effective_bucket_col in last_hour_df.columns and expiration_bucket_col in last_hour_df.columns:
                effective_bucket = str(row[effective_bucket_col]).strip() if pd.notna(row[effective_bucket_col]) else ""
                expiration_bucket = str(row[expiration_bucket_col]).strip() if pd.notna(row[expiration_bucket_col]) else ""
                if effective_bucket and expiration_bucket:
                    structure_desc = f"{effective_bucket} → {expiration_bucket}"
                else:
                    structure_desc = f"{currency} trade"
            else:
                structure_desc = f"{currency} trade"
            
            trades_with_dv01.append({
                "structure": structure_desc,
                "currency": currency,
                "dv01": abs(dv01_val),
                "trade_id": idx
            })
        
        # Sort by DV01 and get top 5
        if trades_with_dv01:
            trades_with_dv01.sort(key=lambda x: x["dv01"], reverse=True)
            top_individual_trades = trades_with_dv01[:5]

    # Find yesterday's date (if available in data and requested)
    yesterday_outputs = []
    yesterday_md_lines = []
    
    if args.include_yesterday or args.yesterday_only:
        # Get all available dates and find the most recent previous date
        all_dates = sorted(df["_trade_date"].dropna().unique())
        if len(all_dates) > 1:
            yesterday_date = all_dates[-2]  # Second most recent date
            
            # Filter to yesterday's trades
            yesterday_mask = df["_trade_date"] == yesterday_date
            yesterday_df = df.loc[yesterday_mask].copy()
            
            if not yesterday_df.empty:
                # Clean numeric notional and DV01 for yesterday's data
                def to_float(x):
                    try:
                        s = str(x).replace(",", "")
                        return float(s)
                    except Exception:
                        return math.nan
                yesterday_df[noz_col] = yesterday_df[noz_col].apply(to_float)
                yesterday_df = yesterday_df.dropna(subset=[noz_col])
                
                # Clean DV01 column if it exists
                if dv01_col and dv01_col in yesterday_df.columns:
                    yesterday_df[dv01_col] = yesterday_df[dv01_col].apply(to_float)
                
                yesterday_md_lines.append(f"# Day-End Commentary ({yesterday_date.strftime('%Y-%m-%d')})")
                yesterday_md_lines.append("")
                
                # Add top 5 trades for yesterday at the top
                if yesterday_df is not None and dv01_col and dv01_col in yesterday_df.columns:
                    # Get yesterday's top 5 trades
                    yesterday_trades_with_dv01 = []
                    effective_bucket_col = find_col(yesterday_df, "effective_bucket")
                    expiration_bucket_col = find_col(yesterday_df, "expiration_bucket")
                    
                    for idx, row in yesterday_df.iterrows():
                        dv01_val = float(row[dv01_col]) if pd.notna(row[dv01_col]) else 0.0
                        currency = str(row[ccy_col]).strip()
                        
                        # Skip trades with zero DV01
                        if abs(dv01_val) <= 0:
                            continue
                        
                        # Create structure description
                        if effective_bucket_col and expiration_bucket_col and effective_bucket_col in yesterday_df.columns and expiration_bucket_col in yesterday_df.columns:
                            effective_bucket = str(row[effective_bucket_col]).strip() if pd.notna(row[effective_bucket_col]) else ""
                            expiration_bucket = str(row[expiration_bucket_col]).strip() if pd.notna(row[expiration_bucket_col]) else ""
                            if effective_bucket and expiration_bucket:
                                structure_desc = f"{effective_bucket} → {expiration_bucket}"
                            else:
                                structure_desc = f"{currency} trade"
                        else:
                            structure_desc = f"{currency} trade"
                        
                        yesterday_trades_with_dv01.append({
                            "structure": structure_desc,
                            "currency": currency,
                            "dv01": abs(dv01_val),
                            "trade_id": idx
                        })
                    
                    # Sort by DV01 and get top 5
                    if yesterday_trades_with_dv01:
                        yesterday_trades_with_dv01.sort(key=lambda x: x["dv01"], reverse=True)
                        yesterday_top_trades = yesterday_trades_with_dv01[:5]
                        
                        yesterday_md_lines.append(f"## Top 5 Trades on {yesterday_date.strftime('%Y-%m-%d')}")
                        for i, trade in enumerate(yesterday_top_trades, 1):
                            dv01_k = format_dv01_k(trade['dv01'])
                            yesterday_md_lines.append(f"{i}. **{trade['structure']}** - ~{dv01_k} DV01 ({trade['currency']})")
                        yesterday_md_lines.append("")
                
                # Generate yesterday's commentary by currency
                for ccy, g in yesterday_df.groupby(ccy_col):
                    stats = summarize_currency(g, ccy_col, noz_col, yesterday_date, dv01_col=dv01_col, maturity_col=mat_col, tenor_col=ten_col)
                    
                    # Find biggest structure for this currency
                    biggest_structure_for_currency = ""
                    if stats.get("trade_structures") and stats.get('total_dv01', 0) > 0:
                        biggest_structure = stats["trade_structures"][0]  # Already sorted by DV01
                        dv01_k = format_dv01_k(biggest_structure['dv01'])
                        biggest_structure_for_currency = f"{biggest_structure['structure']} (~{dv01_k} DV01)"
                    
                    facts = facts_to_bullets(str(ccy), stats, biggest_structure_for_currency)
                    
                    # Generate yesterday's commentary
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(ccy=str(ccy), facts=facts)}
                    ]
                    raw_commentary = call_openai(messages, model=args.model)
                    commentary = enforce_sentence_limit(raw_commentary, max_sentences=3)
                    
                    yesterday_outputs.append({"currency": str(ccy), "trade_date": yesterday_date.strftime("%Y-%m-%d"), "commentary": commentary})
                    yesterday_md_lines.append(f"**{ccy}** — {commentary}")

    # Group by currency for today (unless yesterday_only is requested)
    outputs = []
    md_lines = []
    
    if not args.yesterday_only:
        md_lines = [f"# Daily Swap Commentary ({target.strftime('%Y-%m-%d')})", ""]
        
        # Add top individual trades summary at the top
        if top_individual_trades:
            md_lines.append("## Top 5 Trades by DV01 (Past 1 Hour)")
            for i, trade in enumerate(top_individual_trades, 1):
                dv01_k = format_dv01_k(trade['dv01'])
                md_lines.append(f"{i}. **{trade['structure']}** - ~{dv01_k} DV01 ({trade['currency']})")
            md_lines.append("")
    
    if not args.yesterday_only:
        for ccy, g in day_df.groupby(ccy_col):
            stats = summarize_currency(g, ccy_col, noz_col, target, dv01_col=dv01_col, maturity_col=mat_col, tenor_col=ten_col)
            
            # Find biggest structure for this currency
            biggest_structure_for_currency = ""
            if stats.get("trade_structures") and stats.get('total_dv01', 0) > 0:
                biggest_structure = stats["trade_structures"][0]  # Already sorted by DV01
                dv01_k = format_dv01_k(biggest_structure['dv01'])
                biggest_structure_for_currency = f"{biggest_structure['structure']} (~{dv01_k} DV01)"
            
            facts = facts_to_bullets(str(ccy), stats, biggest_structure_for_currency)

            # Always use GPT-4 for intelligent commentary
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(ccy=str(ccy), facts=facts)}
            ]
            raw_commentary = call_openai(messages, model=args.model)
            # Enforce 3 sentence limit and make copy-pasteable
            commentary = enforce_sentence_limit(raw_commentary, max_sentences=3)

            outputs.append({"currency": str(ccy), "trade_date": target.strftime("%Y-%m-%d"), "commentary": commentary})
            md_lines.append(f"**{ccy}** — {commentary}")

    # Combine yesterday and today outputs for CSV
    all_outputs = yesterday_outputs + outputs
    out_df = pd.DataFrame(all_outputs, columns=["currency","trade_date","commentary"])
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    
    # Combine yesterday and today markdown
    if args.yesterday_only:
        # Yesterday only
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n\n".join(yesterday_md_lines))
        print(f"Wrote {len(yesterday_outputs)} yesterday commentaries to {args.out_csv} and {args.out_md}")
    else:
        # Both or today only
        combined_md_lines = yesterday_md_lines + [""] + md_lines
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n\n".join(combined_md_lines))
        
        if yesterday_outputs:
            print(f"Wrote {len(yesterday_outputs)} yesterday commentaries + {len(outputs)} today commentaries to {args.out_csv} and {args.out_md}")
        else:
            print(f"Wrote {len(outputs)} commentaries to {args.out_csv} and {args.out_md}")

if __name__ == "__main__":
    main()
