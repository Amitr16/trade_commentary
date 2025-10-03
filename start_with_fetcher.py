#!/usr/bin/env python3
"""
Combined startup script for Render deployment
Starts dtcc_fetcher.py in background and then starts the Flask web app
"""

import os
import sys
import subprocess
import threading
import time
from commentary_webapp import app

def run_fetcher():
    """Run dtcc_fetcher.py in a separate thread"""
    try:
        print("Starting dtcc_fetcher.py in background...")
        subprocess.run([sys.executable, "dtcc_fetcher.py"])
    except Exception as e:
        print(f"Error running dtcc_fetcher.py: {e}")

def run_webapp():
    """Run the Flask web app"""
    try:
        print("Starting Flask web app...")
        
        # Get port from environment variable (Render provides this)
        port = int(os.environ.get('PORT', 5000))
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False  # Always False in production
        )
    except Exception as e:
        print(f"Error running web app: {e}")

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
        print("Starting DTCC Commentary Web App on Render...")
        
        # Start dtcc_fetcher in background thread
        fetcher_thread = threading.Thread(target=run_fetcher, daemon=True)
        fetcher_thread.start()
        
        # Wait a bit for fetcher to initialize
        time.sleep(5)
        
        # Start the web app (this will block)
        run_webapp()
        
    else:
        # Development mode (local)
        print("Starting DTCC Commentary Web App locally...")
        print("Note: Run dtcc_fetcher.py separately in another terminal")
        print("Open your browser and go to: http://localhost:5000")
        
        # Just start the web app locally
        run_webapp()
