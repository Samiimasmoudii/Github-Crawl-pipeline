"""
Test CreateEvents approach for finding newly created repositories

This demonstrates the efficient approach that uses GH Archive CreateEvents
instead of PushEvents + API calls.
"""

from repository_finder import RepositoryFinder
from config import Config

def test_createevents_approach():
    """Test the CreateEvents approach"""
    print("🔍 Testing CreateEvents Approach")
    print("="*50)
    
    print(f"Configuration:")
    print(f"  📅 Target date: {Config.MIN_CREATION_DATE}")
    print(f"  🎯 Languages: {Config.TARGET_LANGUAGES}")
    print(f"  🔍 Processing: 2024-01-01 hour 12")
    
    finder = RepositoryFinder()
    
    # Use CreateEvents to find newly created repositories
    print(f"\n📦 Finding repositories created after {Config.MIN_CREATION_DATE}...")
    newly_created_repos = finder.find_newly_created_repositories("2024-01-01", 12)
    
    if newly_created_repos:
        print(f"\n✅ Found {len(newly_created_repos)} newly created repositories")
        
        # Show some samples
        print(f"\n📋 Sample repositories:")
        for i, repo in enumerate(newly_created_repos[:5]):
            print(f"\n{i+1}. {repo['repo_name']}")
            print(f"   📅 Created: {repo['created_at']}")
            print(f"   👤 Creator: {repo['actor']}")
            print(f"   🎯 Language: {repo['likely_language']}")
            print(f"   🤖 LLM Score: {repo['llm_score']}")
            
            if repo['description']:
                desc = repo['description'][:80] + "..." if len(repo['description']) > 80 else repo['description']
                print(f"   📝 Description: {desc}")
        
        print(f"\n💡 Advantages of CreateEvents approach:")
        print(f"  • ✅ No API calls needed for date filtering")
        print(f"  • ✅ Direct access to creation timestamps")
        print(f"  • ✅ More efficient for creation-based filtering")
        print(f"  • ✅ No rate limiting issues")
        
        return True
    else:
        print(f"\n⚠️  No newly created repositories found")
        print(f"   Try adjusting MIN_CREATION_DATE in config.py")
        return False

if __name__ == "__main__":
    success = test_createevents_approach()
    if success:
        print(f"\n🎉 CreateEvents approach is working perfectly!")
    else:
        print(f"\n❌ No results found - try different date/time") 