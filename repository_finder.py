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
        
    def find_newly_created_repositories(self, date_str, hour):
        """
        Find newly created repositories from GH Archive CreateEvents
        
        Args:
            date_str: Date string (YYYY-MM-DD)
            hour: Hour (0-23)
            
        Returns:
            List of newly created repositories
        """
        client = GHArchiveClient()
        
        # Download archive
        filepath = client.download_archive(date_str, hour)
        if not filepath:
            return []
            
        # Parse archive and get CreateEvents
        print(f"📦 Parsing: {filepath}")
        
        repositories = []
        for event in client.parse_archive(filepath):
            if event.get('type') == 'CreateEvent':
                payload = event.get('payload', {})
                
                # Only repository creations (not branches/tags)
                if payload.get('ref_type') == 'repository':
                    repo_name = event.get('repo', {}).get('name', '')
                    created_at = event.get('created_at', '')
                    
                    if repo_name and created_at:
                        # Check if creation date is after our minimum date
                        if self._is_created_after_date_string(created_at):
                            repo_info = {
                                'repo_name': repo_name,
                                'created_at': created_at,
                                'description': payload.get('description', ''),
                                'master_branch': payload.get('master_branch', 'main'),
                                'actor': event.get('actor', {}).get('login', ''),
                                'likely_language': self._guess_language(repo_name),
                                'llm_score': self._calculate_llm_score_from_description(payload.get('description', '')),
                                'commit_count': 0,  # New repos have no commits yet
                                'actors': [event.get('actor', {}).get('login', '')],
                                'commit_messages': [],
                                'latest_commit': ''
                            }
                            repositories.append(repo_info)
        
        print(f"✅ Found {len(repositories)} newly created repositories after {Config.MIN_CREATION_DATE}")
        return repositories

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
        
        if not push_events:
            print("❌ No push events found")
            return []
        
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
        
        print(f"📊 Processing {len(push_events)} push events...")
        
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
        
        print(f"✅ Found {len(repositories)} repositories with target languages")
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
    
    def filter_target_repositories(self, repositories, check_creation_date=False):
        """Filter repositories to target ones"""
        target_repos = []
        
        for repo in repositories:
            # Basic filtering criteria
            if (repo['likely_language'] in ['python', 'java', 'javascript'] and
                repo['commit_count'] >= 1 and
                repo['llm_score'] < 20):  # Stricter LLM filtering
                
                # Check creation date if requested
                if check_creation_date:
                    if not self._is_created_after_date(repo['repo_name']):
                        continue
                
                target_repos.append(repo)
        
        return target_repos
    
    def _is_created_after_date_string(self, created_at_str):
        """Check if repository was created after the minimum date (using date string directly)"""
        from datetime import datetime
        
        try:
            # Parse creation date from GH Archive
            creation_date = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            min_date = datetime.fromisoformat(Config.MIN_CREATION_DATE + 'T00:00:00+00:00')
            
            return creation_date >= min_date
            
        except Exception as e:
            print(f"⚠️ Error parsing creation date {created_at_str}: {e}")
            return False

    def _calculate_llm_score_from_description(self, description):
        """Calculate LLM score based on repository description"""
        if not description:
            return 0
            
        score = 0
        description_lower = description.lower()
        for keyword in Config.LLM_KEYWORDS:
            if keyword in description_lower:
                score += 10
        
        return min(score, 100)  # Cap at 100

    def _is_created_after_date(self, repo_name):
        """Check if repository was created after the minimum date"""
        from github_client import GitHubAPIClient
        from datetime import datetime
        
        try:
            client = GitHubAPIClient()
            repo_info = client.get_repository_info(repo_name)
            
            if not repo_info:
                return False
            
            created_at = repo_info.get('created_at')
            if not created_at:
                return False
            
            # Parse creation date
            creation_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            min_date = datetime.fromisoformat(Config.MIN_CREATION_DATE + 'T00:00:00+00:00')
            
            return creation_date >= min_date
            
        except Exception as e:
            print(f"⚠️ Error checking creation date for {repo_name}: {e}")
            return False


def test_repository_finder():
    """Test the repository finder"""
    print("🔍 Testing Repository Finder")
    print("="*50)
    
    finder = RepositoryFinder()
    
    # Find repositories from one hour
    print(f"📅 Finding repositories from 2024-01-01 hour 12...")
    repositories = finder.find_repositories_from_archive("2024-01-01", 12)
    
    print(f"✅ Found {len(repositories)} active repositories")
    
    # Filter to target repositories
    target_repos = finder.filter_target_repositories(repositories)
    
    print(f"🎯 Found {len(target_repos)} target repositories")
    
    # Test CreateEvent-based discovery for newly created repos
    print(f"\n📅 Finding newly created repositories (created after {Config.MIN_CREATION_DATE})...")
    newly_created_repos = finder.find_newly_created_repositories("2024-01-01", 12)
    
    # Show sample results
    print("\n📊 Sample target repositories:")
    for i, repo in enumerate(target_repos[:5]):
        print(f"\n{i+1}. {repo['repo_name']}")
        print(f"   🎯 Language: {repo['likely_language']}")
        print(f"   📝 Commits: {repo['commit_count']}")
        print(f"   👥 Contributors: {len(repo['actors'])}")
        print(f"   🤖 LLM Score: {repo['llm_score']}")
        print(f"   📄 Latest commit: {repo['latest_commit'][:8] if repo['latest_commit'] else 'N/A'}...")
        
        # Show sample commit message
        if repo['commit_messages']:
            sample_msg = repo['commit_messages'][0][:80] + "..." if len(repo['commit_messages'][0]) > 80 else repo['commit_messages'][0]
            print(f"   💬 Sample commit: {sample_msg}")
    
    if newly_created_repos:
        print(f"\n✅ Success! Found {len(newly_created_repos)} newly created repositories")
        print("🚀 Next step: Use GitHub API to get file contents from these repositories")
        return True
    else:
        print("\n⚠️  No newly created repositories found. Try a different time period or adjust filtering.")
        return False


if __name__ == "__main__":
    test_repository_finder() 