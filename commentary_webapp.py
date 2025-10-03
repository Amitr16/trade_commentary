#!/usr/bin/env python3
"""
Flask Web Application for DTCC Trade Commentary
Displays AI-generated swap commentary in a web interface
"""

import os
import sys
import subprocess
import pandas as pd
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

# Configuration
CSV_FILE = "dtcc_trades.csv"
TODAYS_COMMENTARY_CSV = "todays_commentary.csv"
TODAYS_COMMENTARY_MD = "todays_commentary.md"
YESTERDAYS_COMMENTARY_CSV = "yesterdays_commentary.csv"
YESTERDAYS_COMMENTARY_MD = "yesterdays_commentary.md"

def generate_todays_commentary(date_filter=None):
    """Generate today's commentary"""
    try:
        cmd = ["python", "generate_fx_commentary.py", CSV_FILE, "--out_csv", TODAYS_COMMENTARY_CSV, "--out_md", TODAYS_COMMENTARY_MD]
        if date_filter:
            cmd.extend(["--date", date_filter])
        
        # Run the commentary generation
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, "Today's commentary generated successfully"
        else:
            return False, f"Error: {result.stderr}"
            
    except Exception as e:
        return False, f"Exception: {str(e)}"

def generate_yesterdays_commentary():
    """Generate yesterday's commentary"""
    try:
        cmd = ["python", "generate_fx_commentary.py", CSV_FILE, "--yesterday_only", "--out_csv", YESTERDAYS_COMMENTARY_CSV, "--out_md", YESTERDAYS_COMMENTARY_MD]
        
        # Run the commentary generation
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, "Yesterday's commentary generated successfully"
        else:
            return False, f"Error: {result.stderr}"
            
    except Exception as e:
        return False, f"Exception: {str(e)}"

def load_todays_commentary_data():
    """Load today's commentary data from CSV"""
    try:
        if os.path.exists(TODAYS_COMMENTARY_CSV):
            df = pd.read_csv(TODAYS_COMMENTARY_CSV)
            return df.to_dict('records')
        return []
    except Exception as e:
        print(f"Error loading today's commentary: {e}")
        return []

def load_yesterdays_commentary_data():
    """Load yesterday's commentary data from CSV"""
    try:
        if os.path.exists(YESTERDAYS_COMMENTARY_CSV):
            df = pd.read_csv(YESTERDAYS_COMMENTARY_CSV)
            return df.to_dict('records')
        return []
    except Exception as e:
        print(f"Error loading yesterday's commentary: {e}")
        return []

def load_todays_markdown_commentary():
    """Load today's markdown commentary"""
    try:
        if os.path.exists(TODAYS_COMMENTARY_MD):
            with open(TODAYS_COMMENTARY_MD, 'r', encoding='utf-8') as f:
                return f.read()
        return "No today's commentary available"
    except Exception as e:
        print(f"Error loading today's markdown: {e}")
        return "Error loading today's commentary"

def load_yesterdays_markdown_commentary():
    """Load yesterday's markdown commentary"""
    try:
        if os.path.exists(YESTERDAYS_COMMENTARY_MD):
            with open(YESTERDAYS_COMMENTARY_MD, 'r', encoding='utf-8') as f:
                return f.read()
        return "No yesterday's commentary available"
    except Exception as e:
        print(f"Error loading yesterday's markdown: {e}")
        return "Error loading yesterday's commentary"

def parse_top_trades(markdown_content):
    """Extract top 5 trades from markdown content"""
    lines = markdown_content.split('\n')
    top_trades = []
    
    in_top_trades_section = False
    for line in lines:
        line = line.strip()
        if line.startswith('## Top 5 Trades by DV01'):
            in_top_trades_section = True
            continue
        elif in_top_trades_section and line.startswith('##') and not line.startswith('## Top 5'):
            break
        elif in_top_trades_section and line.startswith(('1.', '2.', '3.', '4.', '5.')):
            # Parse trade line: "1. **Spot → 2030-01-31** - ~267,377 DV01 (ZAR)"
            match = re.match(r'\d+\.\s*\*\*(.*?)\*\*\s*-\s*~(.*?)\s+DV01\s*\((.*?)\)', line)
            if match:
                structure, dv01, currency = match.groups()
                top_trades.append({
                    'structure': structure,
                    'dv01': dv01,
                    'currency': currency
                })
    
    return top_trades

def markdown_to_html(markdown_content):
    """Convert basic markdown to HTML"""
    html = markdown_content
    
    # Convert headers
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # Convert bold text
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    
    # Convert list items
    html = re.sub(r'^\d+\.\s*(.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # Wrap consecutive list items in <ul>
    html = re.sub(r'(<li>.*</li>\s*)+', lambda m: f'<ul>{m.group(0)}</ul>', html, flags=re.DOTALL)
    
    # Convert line breaks
    html = html.replace('\n', '<br>\n')
    
    return html

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/commentary')
def commentary():
    """Display commentary page with tabs"""
    todays_markdown = load_todays_markdown_commentary()
    todays_html = markdown_to_html(todays_markdown)
    
    yesterdays_markdown = load_yesterdays_markdown_commentary()
    yesterdays_html = markdown_to_html(yesterdays_markdown)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return render_template('commentary.html', 
                         todays_content=todays_html,
                         yesterdays_content=yesterdays_html,
                         current_time=current_time)

@app.route('/api/refresh_today', methods=['POST'])
def api_refresh_today():
    """API endpoint to refresh today's commentary"""
    if request.json:
        date_filter = request.json.get('date')
    else:
        date_filter = None
    
    success, message = generate_todays_commentary(date_filter)
    
    if success:
        markdown_content = load_todays_markdown_commentary()
        html_content = markdown_to_html(markdown_content)
        return jsonify({"success": True, "html_content": html_content})
    else:
        return jsonify({"success": False, "error": message})

@app.route('/api/refresh_yesterday', methods=['POST'])
def api_refresh_yesterday():
    """API endpoint to refresh yesterday's commentary"""
    success, message = generate_yesterdays_commentary()
    
    if success:
        markdown_content = load_yesterdays_markdown_commentary()
        html_content = markdown_to_html(markdown_content)
        return jsonify({"success": True, "html_content": html_content})
    else:
        return jsonify({"success": False, "error": message})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Check if running on Render (production) or locally (development)
    if os.environ.get('RENDER'):
        # Production mode (Render)
        port = int(os.environ.get('PORT', 10000))
        print(f"Starting DTCC Commentary Web App on Render (port {port})...")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # Development mode (local)
        print("Starting DTCC Commentary Web App locally...")
        print("Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
