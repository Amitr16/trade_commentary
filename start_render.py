#!/usr/bin/env python3
"""
Render deployment startup script for DTCC Commentary Web App
"""

import os
import sys
from commentary_webapp import app

if __name__ == '__main__':
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Always False in production
    )