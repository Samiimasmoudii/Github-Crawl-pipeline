#!/usr/bin/env python3
"""
Test script for GH Archive functionality
Run this to verify everything works before proceeding
"""

import sys
import os
from gharchive_client import GHArchiveClient

def test_download_and_parse():
    """Test downloading and parsing a single GH Archive file"""
    print("🔍 Testing GH Archive Download & Parse...")
    
    client = GHArchiveClient()
    
    # Test with a known good date/hour
    date_str = "2024-01-01"
    hour = 12
    
    print(f"📥 Downloading archive: {date_str}-{hour:02d}.json.gz")
    filepath = client.download_archive(date_str, hour)
    
    if not filepath:
        print("❌ Download failed!")
        return False
    
    print(f"📦 Parsing archive...")
    push_events = client.get_push_events(filepath)
    
    if not push_events:
        print("❌ No push events found!")
        return False
    
    print(f"✅ Found {len(push_events)} push events")
    
    # Test extracting repository info
    print("\n📊 Sample repositories:")
    for i, event in enumerate(push_events[:3]):
        repo_info = client.extract_repo_info(event)
        print(f"  {i+1}. {repo_info['repo_name']}")
        print(f"     📝 Commits: {len(repo_info['commits'])}")
        print(f"     👤 Actor: {repo_info['actor']}")
        print(f"     🕐 Time: {repo_info['created_at']}")
        
        # Show commit messages for LLM detection testing
        if repo_info['commits']:
            for j, commit in enumerate(repo_info['commits'][:2]):  # First 2 commits
                message = commit.get('message', '')[:100] + '...' if len(commit.get('message', '')) > 100 else commit.get('message', '')
                print(f"        💬 Commit {j+1}: {message}")
        print()
    
    # Clean up
    os.remove(filepath)
    print("✅ Test completed successfully!")
    return True

def test_config():
    """Test configuration"""
    print("⚙️  Testing Configuration...")
    
    from config import Config
    
    print(f"📍 GH Archive URL: {Config.GH_ARCHIVE_BASE_URL}")
    print(f"📅 Date range: {Config.ARCHIVE_START_DATE} to {Config.ARCHIVE_END_DATE}")
    print(f"🎯 Target languages: {Config.TARGET_LANGUAGES}")
    print(f"📄 Target extensions: {Config.TARGET_EXTENSIONS}")
    print(f"⭐ Min stars: {Config.MIN_STARS}")
    print(f"🔍 LLM keywords: {len(Config.LLM_KEYWORDS)} keywords")
    
    # Test URL generation
    test_url = Config.get_archive_url("2024-01-01", 12)
    expected_url = "https://data.gharchive.org/2024-01-01-12.json.gz"
    
    if test_url == expected_url:
        print("✅ URL generation works correctly")
        return True
    else:
        print(f"❌ URL generation failed: got {test_url}, expected {expected_url}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting GH Archive Pipeline Tests\n")
    
    tests = [
        ("Configuration", test_config),
        ("GH Archive Download & Parse", test_download_and_parse),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("🎯 TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready to proceed with next steps.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 