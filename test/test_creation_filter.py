"""
Test for finding newly created repositories

This script demonstrates how to find repositories created after a specific date
using GH Archive CreateEvents.
"""

from repository_finder import RepositoryFinder
from config import Config

def test_newly_created_repositories():
    """Test finding newly created repositories"""
    print("🔍 Finding Newly Created Repositories")
    print("="*50)
    
    finder = RepositoryFinder()
    
    # Find repositories from one hour
    print(f"📅 Finding repositories from 2024-01-01 hour 12...")
    repositories = finder.find_repositories_from_archive("2024-01-01", 12)
    
    print(f"✅ Found {len(repositories)} active repositories")
    
    # Find newly created repositories using CreateEvents
    print(f"\n📅 Finding repositories created after {Config.MIN_CREATION_DATE}...")
    newly_created_repos = finder.find_newly_created_repositories("2024-01-01", 12)
    
    # Show sample repositories
    if newly_created_repos:
        print(f"\n📋 Sample repositories created after {Config.MIN_CREATION_DATE}:")
        for i, repo in enumerate(newly_created_repos[:3]):
            print(f"\n{i+1}. {repo['repo_name']}")
            print(f"   🎯 Language: {repo['likely_language']}")
            print(f"   📅 Created: {repo['created_at']}")
            print(f"   👤 Creator: {repo['actor']}")
            print(f"   🤖 LLM Score: {repo['llm_score']}")
            if repo['description']:
                print(f"   📝 Description: {repo['description'][:100]}...")
    
    return len(newly_created_repos) > 0

if __name__ == "__main__":
    success = test_newly_created_repositories()
    if success:
        print("\n✅ Finding newly created repositories is working!")
        print("🚀 This approach uses CreateEvents - no API calls needed!")
    else:
        print("\n⚠️  No newly created repositories found")
        print("   Try adjusting the MIN_CREATION_DATE in config.py") 