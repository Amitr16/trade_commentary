#!/usr/bin/env python3
"""
Test script to verify DTCC Trade Commentary setup
"""

import os
import sys
import pandas as pd
from datetime import datetime

def test_files():
    """Test if all required files exist"""
    required_files = [
        'dtcc_trades.csv',
        'fx.csv', 
        'MPC_Dates.csv',
        'IMM_Dates.csv',
        'dtcc_fetcher.py',
        'generate_fx_commentary.py',
        'commentary_webapp.py',
        'requirements_webapp.txt',
        'start_with_fetcher.py',
        'render.yaml'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_csv_files():
    """Test CSV file structure"""
    try:
        # Test fx.csv
        fx_df = pd.read_csv('fx.csv')
        if 'Currency' in fx_df.columns and 'Value' in fx_df.columns:
            print(f"✅ fx.csv: {len(fx_df)} currencies loaded")
        else:
            print("❌ fx.csv missing required columns")
            return False
        
        # Test MPC_Dates.csv
        mpc_df = pd.read_csv('MPC_Dates.csv')
        if 'Bank' in mpc_df.columns and 'Date' in mpc_df.columns:
            print(f"✅ MPC_Dates.csv: {len(mpc_df)} dates loaded")
        else:
            print("❌ MPC_Dates.csv missing required columns")
            return False
        
        # Test IMM_Dates.csv
        imm_df = pd.read_csv('IMM_Dates.csv')
        if 'Month' in imm_df.columns and 'Code' in imm_df.columns:
            print(f"✅ IMM_Dates.csv: {len(imm_df)} dates loaded")
        else:
            print("❌ IMM_Dates.csv missing required columns")
            return False
        
        # Test dtcc_trades.csv (can be empty)
        if os.path.exists('dtcc_trades.csv'):
            trades_df = pd.read_csv('dtcc_trades.csv')
            print(f"✅ dtcc_trades.csv: {len(trades_df)} trades loaded")
        else:
            print("⚠️ dtcc_trades.csv not found (will be created by fetcher)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading CSV files: {e}")
        return False

def test_dependencies():
    """Test Python dependencies"""
    try:
        import flask
        import pandas
        import requests
        print("✅ Core dependencies available")
        
        # Test OpenAI (optional for dry run)
        try:
            import openai
            print("✅ OpenAI dependency available")
        except ImportError:
            print("⚠️ OpenAI not available (needed for commentary generation)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_dtcc_fetcher():
    """Test DTCC fetcher import"""
    try:
        sys.path.append('.')
        from dtcc_fetcher import DTCCFetcher
        fetcher = DTCCFetcher()
        print("✅ DTCC Fetcher class imported successfully")
        return True
    except Exception as e:
        print(f"❌ DTCC Fetcher import failed: {e}")
        return False

def test_commentary_generator():
    """Test commentary generator import"""
    try:
        sys.path.append('.')
        import generate_fx_commentary
        print("✅ Commentary generator imported successfully")
        return True
    except Exception as e:
        print(f"❌ Commentary generator import failed: {e}")
        return False

def test_flask_app():
    """Test Flask app import"""
    try:
        sys.path.append('.')
        from commentary_webapp import app
        print("✅ Flask app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Flask app import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing DTCC Trade Commentary Setup")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_files),
        ("CSV Files", test_csv_files),
        ("Dependencies", test_dependencies),
        ("DTCC Fetcher", test_dtcc_fetcher),
        ("Commentary Generator", test_commentary_generator),
        ("Flask App", test_flask_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment.")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
