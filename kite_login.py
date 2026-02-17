#!/usr/bin/env python3
"""
Quick Kite Connect Login Script
Run this daily before market opens to get fresh access token.
"""

import json
import os
import logging
import webbrowser
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from logging.handlers import RotatingFileHandler

# Check if kiteconnect is installed
try:
    from kiteconnect import KiteConnect
except ImportError:
    print("Installing kiteconnect...")
    os.system("pip install --user kiteconnect")
    from kiteconnect import KiteConnect

CREDS_FILE = "/u/tarar/repos/.kite_creds.json"
SESSION_FILE = "/u/tarar/repos/.kite_session.json"
LOG_DIR = "/u/tarar/repos/logs"

# ============== LOGGING SETUP ==============
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging():
    """Setup logging for login script"""
    logger = logging.getLogger('login')
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - append to daily log
    today = datetime.now().strftime('%Y-%m-%d')
    file_handler = RotatingFileHandler(
        f"{LOG_DIR}/login_{today}.log",
        maxBytes=5*1024*1024,
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

log = setup_logging()

def load_creds():
    log.info(f"Loading credentials from {CREDS_FILE}")
    with open(CREDS_FILE, 'r') as f:
        creds = json.load(f)
    log.info(f"Loaded API key: {creds['api_key'][:8]}...")
    return creds

def save_session(access_token, api_key):
    session_data = {
        "access_token": access_token,
        "api_key": api_key,
        "login_time": datetime.now().isoformat(),
        "valid_until": "End of day"
    }
    with open(SESSION_FILE, 'w') as f:
        json.dump(session_data, f, indent=2)
    log.info(f"Session saved to {SESSION_FILE}")
    log.info(f"Access token: {access_token[:10]}...")
    print(f"\n✅ Session saved to {SESSION_FILE}")

def extract_request_token(url_or_token):
    """Extract request_token from full URL or return as-is if already a token"""
    log.debug(f"Extracting request_token from input (length: {len(url_or_token)})")
    if url_or_token.startswith("http"):
        parsed = urlparse(url_or_token)
        params = parse_qs(parsed.query)
        log.debug(f"Parsed URL params: {list(params.keys())}")
        if 'request_token' in params:
            token = params['request_token'][0]
            log.info(f"Extracted request_token from URL: {token[:8]}...")
            return token
        else:
            log.error("No request_token found in URL")
            raise ValueError("No request_token found in URL")
    log.info(f"Using raw input as request_token: {url_or_token[:8]}...")
    return url_or_token.strip()

def main():
    log.info("="*60)
    log.info("KITE CONNECT LOGIN SESSION STARTED")
    log.info(f"Timestamp: {datetime.now().isoformat()}")
    log.info("="*60)
    
    print("=" * 60)
    print("ZERODHA KITE CONNECT LOGIN")
    print("=" * 60)
    
    # Load credentials
    try:
        creds = load_creds()
        api_key = creds['api_key']
        api_secret = creds['api_secret']
    except Exception as e:
        log.error(f"Failed to load credentials: {e}")
        print(f"❌ Failed to load credentials: {e}")
        return 1
    
    # Initialize Kite
    log.info("Initializing KiteConnect")
    kite = KiteConnect(api_key=api_key)
    
    # Generate login URL
    login_url = kite.login_url()
    log.info(f"Generated login URL: {login_url}")
    print(f"\n📌 Login URL:\n{login_url}")
    
    # Try to open browser
    try:
        webbrowser.open(login_url)
        log.info("Browser opened for login")
        print("\n🌐 Browser opened. Please login to Zerodha.")
    except Exception as e:
        log.warning(f"Could not open browser: {e}")
        print("\n⚠️  Could not open browser. Please copy the URL above manually.")
    
    print("\n" + "-" * 60)
    print("After logging in, you'll be redirected to a URL like:")
    print("https://127.0.0.1/?request_token=XXXX&action=login&status=success")
    print("-" * 60)
    
    # Get request token - accept full URL or just the token
    print("\n📋 Paste the FULL redirect URL (or just the request_token):")
    user_input = input("> ").strip()
    
    log.info("User provided redirect URL/token")
    
    try:
        request_token = extract_request_token(user_input)
        print(f"\n🔑 Extracted request_token: {request_token[:8]}...")
        
        # Generate session
        print("\n⏳ Generating session...")
        log.info("Calling generate_session API")
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        log.info(f"Session generated successfully. Access token: {access_token[:10]}...")
        log.debug(f"Session data keys: {list(data.keys())}")
        
        # Set access token
        kite.set_access_token(access_token)
        log.info("Access token set on KiteConnect instance")
        
        # Test the connection
        log.info("Testing connection - fetching profile")
        profile = kite.profile()
        log.info(f"Profile fetched: {profile['user_name']} ({profile['email']})")
        log.info(f"User ID: {profile.get('user_id', 'N/A')}")
        log.info(f"Broker: {profile['broker']}")
        
        print(f"\n✅ Login successful!")
        print(f"   User: {profile['user_name']}")
        print(f"   Email: {profile['email']}")
        print(f"   Broker: {profile['broker']}")
        
        # Save session
        save_session(access_token, api_key)
        
        # Show margins
        log.info("Fetching margin details")
        margins = kite.margins()
        equity = margins.get('equity', {})
        available = equity.get('available', {})
        
        cash = available.get('live_balance', 0)
        collateral = available.get('collateral', 0)
        
        log.info(f"Margin - Cash: ₹{cash:,.2f}, Collateral: ₹{collateral:,.2f}")
        print(f"\n💰 Available Margin: ₹{cash:,.2f}")
        
        log.info("="*60)
        log.info("LOGIN SUCCESSFUL - Session ready for trading")
        log.info("="*60)
        
        print("\n" + "=" * 60)
        print("You can now run the trading engine!")
        print(f"📁 Logs saved to: {LOG_DIR}/")
        print("=" * 60)
        
    except Exception as e:
        log.error(f"Login failed: {e}")
        log.error(f"Exception type: {type(e).__name__}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you copied the complete URL")
        print("2. The request_token expires in 60 seconds - be quick!")
        print("3. Check if API key and secret are correct in .kite_creds.json")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
