"""
Repository finder that uses GH Archive for discovery and GitHub API for file details
"""

import re
from collections import defaultdict
from gharchive_client import GHArchiveClient
from config import Config

class RepositoryFinder:
    def __init__(self):
        self.language_indicators = {
            'python': ['python', 'py-', '-py', 'django', 'flask', 'fastapi', 'pytorch', 'tensorflow'],
            'java': ['java', 'spring', 'android', 'maven', 'gradle'],
            'javascript': ['javascript', 'js-', '-js', 'node', 'react', 'vue', 'angular', 'npm']
        }
        
    def find_repositories_from_archive(self, date_str, hour):
        """
        Find active repositories from GH Archive
        
        Args:
            date_str: Date string (YYYY-MM-DD)
            hour: Hour (0-23)
            
        Returns:
            List of repository candidates
        """
        client = GHArchiveClient()
        
        # Download archive
        filepath = client.download_archive(date_str, hour)
        if not filepath:
            return []
        
        # Get push events
        push_events = client.get_push_events(filepath)
        
        # Process repositories
        repo_stats = defaultdict(lambda: {
            'repo_name': '',
            'commit_count': 0,
            'actors': set(),
            'latest_commit': '',
            'commit_messages': [],
            'likely_language': 'unknown',
            'llm_score': 0
        })
        
        for event in push_events:
            repo_name = event.get('repo', {}).get('name', '')
            if not repo_name:
                continue
                
            # Update repository stats
            repo_stats[repo_name]['repo_name'] = repo_name
            repo_stats[repo_name]['commit_count'] += 1
            repo_stats[repo_name]['actors'].add(event.get('actor', {}).get('login', ''))
            
            # Collect commit messages for LLM detection
            commits = event.get('payload', {}).get('commits', [])
            for commit in commits:
                message = commit.get('message', '')
                if message:
                    repo_stats[repo_name]['commit_messages'].append(message)
                    repo_stats[repo_name]['latest_commit'] = commit.get('sha', '')
            
            # Try to determine language from repository name
            repo_stats[repo_name]['likely_language'] = self._guess_language(repo_name)
            
            # Basic LLM detection on commit messages
            repo_stats[repo_name]['llm_score'] = self._calculate_llm_score(
                repo_stats[repo_name]['commit_messages']
            )
        
        # Clean up
        import os
        os.remove(filepath)
        
        # Convert to list and filter
        repositories = []
        for repo_name, stats in repo_stats.items():
            # Convert set to list
            stats['actors'] = list(stats['actors'])
            
            # Filter by activity and language
            if (stats['commit_count'] >= 1 and 
                stats['likely_language'] in ['python', 'java', 'javascript'] and
                stats['llm_score'] < 50):  # Basic LLM filtering
                repositories.append(stats)
        
        return repositories
    
    def _guess_language(self, repo_name):
        """Guess repository language from name"""
        repo_lower = repo_name.lower()
        
        for language, indicators in self.language_indicators.items():
            if any(indicator in repo_lower for indicator in indicators):
                return language
        
        return 'unknown'
    
    def _calculate_llm_score(self, commit_messages):
        """Calculate LLM score based on commit messages"""
        score = 0
        
        for message in commit_messages:
            message_lower = message.lower()
            for keyword in Config.LLM_KEYWORDS:
                if keyword in message_lower:
                    score += 10
        
        return min(score, 100)  # Cap at 100
    
    def filter_target_repositories(self, repositories):
        """Filter repositories to target ones"""
        target_repos = []
        
        for repo in repositories:
            # Basic filtering criteria
            if (repo['likely_language'] in ['python', 'java', 'javascript'] and
                repo['commit_count'] >= 1 and
                repo['llm_score'] < 20):  # Stricter LLM filtering
                target_repos.append(repo)
        
        return target_repos


def test_repository_finder():
    """Test the repository finder"""
    print("🔍 Testing Repository Finder...")
    
    finder = RepositoryFinder()
    
    # Find repositories from one hour
    repositories = finder.find_repositories_from_archive("2024-01-01", 12)
    
    print(f"✅ Found {len(repositories)} active repositories")
    
    # Filter to target repositories
    target_repos = finder.filter_target_repositories(repositories)
    
    print(f"🎯 Found {len(target_repos)} target repositories")
    
    # Show sample results
    print("\n📊 Sample target repositories:")
    for i, repo in enumerate(target_repos[:5]):
        print(f"\n{i+1}. {repo['repo_name']}")
        print(f"   🎯 Language: {repo['likely_language']}")
        print(f"   📝 Commits: {repo['commit_count']}")
        print(f"   👥 Contributors: {len(repo['actors'])}")
        print(f"   🤖 LLM Score: {repo['llm_score']}")
        print(f"   📄 Latest commit: {repo['latest_commit'][:8]}...")
        
        # Show sample commit message
        if repo['commit_messages']:
            sample_msg = repo['commit_messages'][0][:80] + "..." if len(repo['commit_messages'][0]) > 80 else repo['commit_messages'][0]
            print(f"   💬 Sample commit: {sample_msg}")
    
    if target_repos:
        print(f"\n✅ Success! Found {len(target_repos)} candidate repositories")
        print("🚀 Next step: Use GitHub API to get file contents from these repositories")
        return True
    else:
        print("\n⚠️  No target repositories found. Try a different time period or adjust filtering.")
        return False


if __name__ == "__main__":
    test_repository_finder() 