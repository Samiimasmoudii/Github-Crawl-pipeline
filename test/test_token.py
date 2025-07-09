#!/usr/bin/env python3
"""
Quick test to verify GitHub token is loaded from .env file
"""

from config import Config

def test_token_loading():
    """Test if GitHub token is loaded correctly"""
    print("🔑 Testing GitHub Token Loading from .env")
    print("="*50)
    
    token = Config.GITHUB_TOKEN
    
    if token:
        # Show partial token for verification (security safe)
        if len(token) > 10:
            masked_token = f"{token[:7]}...{token[-4:]}"
        else:
            masked_token = f"{token[:3]}..."
        
        print(f"✅ Token loaded: {masked_token}")
        print(f"🔒 Token length: {len(token)} characters")
        
        # Basic validation
        if token.startswith('ghp_') or token.startswith('github_pat_'):
            print("✅ Token format looks correct")
            return True
        else:
            print("⚠️  Token doesn't start with 'ghp_' or 'github_pat_' - might be wrong format")
            return False
    else:
        print("❌ No token found!")
        print("💡 Make sure you have a .env file with:")
        print("   GITHUB_TOKEN=your_token_here")
        return False

if __name__ == "__main__":
    test_token_loading() 