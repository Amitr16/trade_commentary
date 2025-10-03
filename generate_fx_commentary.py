
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
    "trade_date": ["trade_date", "date", "tradedate", "execution_date", "exec_date", "trade time", "created at"],
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
        bullets.append(f"Total DV01: ~{stats.get('total_dv01',0):,.0f} USD")
        bullets.append(f"Average DV01 per trade: ~{stats.get('avg_dv01_per_trade',0):,.0f} USD")
    else:
        bullets.append(f"Total notional: ~{stats.get('total_notional',0):,.0f}")
        bullets.append(f"Average trade size: ~{stats.get('avg_trade_size',0):,.0f}")
    
    if stats.get("top_buckets"):
        if stats.get('total_dv01', 0) > 0:
            # DV01-based buckets
            tops = ", ".join([f"{b['bucket']} (~{b['dv01']:,.0f} DV01)" for b in stats["top_buckets"][:3]])
            bullets.append(f"Top tenor buckets by DV01: {tops}")
        else:
            # Notional-based buckets
            tops = ", ".join([f"{b['bucket']} (~{b['notional']:,.0f})" for b in stats["top_buckets"][:3]])
            bullets.append(f"Top tenor buckets by notional: {tops}")
    
    # Trade structures (most traded by DV01)
    if stats.get("trade_structures"):
        if stats.get('total_dv01', 0) > 0 and 'dv01' in str(stats["trade_structures"]):
            # Use DV01 for trade structures
            structures = ", ".join([f"{x['structure']} ({x['count']} trades, ~{x['dv01']:,.0f} DV01)" for x in stats["trade_structures"][:2]])
            bullets.append(f"Most traded structures (by DV01): {structures}")
        else:
            # Fallback to notional only if DV01 not available
            structures = ", ".join([f"{x['structure']} ({x['count']} trades, ~{x['notional']:,.0f})" for x in stats["trade_structures"][:2]])
            bullets.append(f"Most traded structures (by notional): {structures}")
    
    if stats.get("maturity_clusters"):
        if stats.get('total_dv01', 0) > 0 and 'dv01' in str(stats["maturity_clusters"]):
            # Use DV01 for maturity clusters
            yrs = ", ".join([f"{x['year']} (~{x['dv01']:,.0f} DV01)" for x in stats["maturity_clusters"]])
            bullets.append(f"Maturity clusters (by DV01): {yrs}")
        else:
            # Fallback to notional only if DV01 not available
            yrs = ", ".join([f"{x['year']} (~{x['notional']:,.0f})" for x in stats["maturity_clusters"]])
            bullets.append(f"Maturity clusters (by notional): {yrs}")
    
    return "\n".join(f"- {x}" for x in bullets)

# -------------------------------
# LLM prompt
# -------------------------------
SYSTEM_PROMPT = (
    "You are a sell-side rates strategist writing crisp, one-paragraph commentary on daily swap activity by currency.\n"
    "Constraints:\n"
    "- One paragraph (3-6 sentences), no bullets.\n"
    "- ALWAYS start with the biggest structure for this currency (highlighted as BIGGEST STRUCTURE).\n"
    "- Lead with where activity concentrates on the curve (belly/front/long end).\n"
    "- ALWAYS focus on DV01 metrics when available - do NOT mention notional amounts.\n"
    "- Mention most traded structures and largest DV01 buckets.\n"
    "- Note average DV01 per trade and total DV01 flow in round figures.\n"
    "- Highlight common trade structures that indicate systematic trading patterns.\n"
    "- Finish with an interpretation (e.g., hedging/ALM vs RV), without over-claiming.\n"
    "- Never reference notional amounts - use DV01 exclusively for all risk metrics.\n"
    "Avoid numbers beyond what is given; round to whole thousands/millions where sensible; keep tone neutral and professional.\n"
)

USER_PROMPT_TEMPLATE = (
    "Write a neat one-paragraph summary for {ccy} using only these facts:\n\n"
    "{facts}\n\n"
    "Style: Same voice as the JPY example you were shown previously — concise, professional, no bullets. Do not invent data.\n"
)

# -------------------------------
# OpenAI client (lazy import)
# -------------------------------
def call_openai(messages, model="gpt-4o", temperature=0.2, max_tokens=220):
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

# -------------------------------
# Main pipeline
# -------------------------------
def main():
    p = argparse.ArgumentParser(description="Generate daily swap commentary by currency from a trades CSV.")
    p.add_argument("csv_path", help="Path to trades CSV (e.g., dtcc_trades.csv)")
    p.add_argument("--out_csv", default="daily_commentary.csv", help="Output CSV path")
    p.add_argument("--out_md", default="daily_commentary.md", help="Output Markdown path")
    p.add_argument("--date", default=None, help="Trade date to filter (YYYY-MM-DD). Defaults to latest date in file.")
    p.add_argument("--dry_run", action="store_true", help="If set, do not call OpenAI; output facts only.")
    p.add_argument("--model", default="gpt-4o", help="OpenAI chat model name")
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

    # Top 5 individual trades by DV01 (currency agnostic)
    top_individual_trades = []
    
    if dv01_col and dv01_col in day_df.columns:
        # Create list of individual trades with DV01
        trades_with_dv01 = []
        effective_bucket_col = find_col(day_df, "effective_bucket")
        expiration_bucket_col = find_col(day_df, "expiration_bucket")
        
        for idx, row in day_df.iterrows():
            dv01_val = float(row[dv01_col]) if pd.notna(row[dv01_col]) else 0.0
            currency = str(row[ccy_col]).strip()
            
            # Create structure description
            if effective_bucket_col and expiration_bucket_col and effective_bucket_col in day_df.columns and expiration_bucket_col in day_df.columns:
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

    # Group by currency
    outputs = []
    md_lines = [f"# Daily Swap Commentary ({target.strftime('%Y-%m-%d')})", ""]
    
    # Add top individual trades summary at the top
    if top_individual_trades:
        md_lines.append("## Top 5 Trades by DV01 (Past 1 Hour)")
        for i, trade in enumerate(top_individual_trades, 1):
            md_lines.append(f"{i}. **{trade['structure']}** - ~{trade['dv01']:,.0f} DV01 ({trade['currency']})")
        md_lines.append("")
    
    for ccy, g in day_df.groupby(ccy_col):
        stats = summarize_currency(g, ccy_col, noz_col, target, dv01_col=dv01_col, maturity_col=mat_col, tenor_col=ten_col)
        
        # Find biggest structure for this currency
        biggest_structure_for_currency = ""
        if stats.get("trade_structures") and stats.get('total_dv01', 0) > 0:
            biggest_structure = stats["trade_structures"][0]  # Already sorted by DV01
            biggest_structure_for_currency = f"{biggest_structure['structure']} (~{biggest_structure['dv01']:,.0f} DV01)"
        
        facts = facts_to_bullets(str(ccy), stats, biggest_structure_for_currency)

        if args.dry_run:
            commentary = "(dry-run) " + facts.replace("\n", " ")
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(ccy=str(ccy), facts=facts)}
            ]
            commentary = call_openai(messages, model=args.model)

        outputs.append({"currency": str(ccy), "trade_date": target.strftime("%Y-%m-%d"), "commentary": commentary})
        md_lines.append(f"**{ccy}** — {commentary}")

    out_df = pd.DataFrame(outputs, columns=["currency","trade_date","commentary"])
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_lines))

    print(f"Wrote {len(outputs)} commentaries to {args.out_csv} and {args.out_md}")

if __name__ == "__main__":
    main()
