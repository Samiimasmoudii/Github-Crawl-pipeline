"""Process GH Archive events to extract file and repository information"""

import re
from config import Config

class EventProcessor:
    def __init__(self):
        self.target_extensions = Config.TARGET_EXTENSIONS
        self.target_languages = Config.TARGET_LANGUAGES
        
    def extract_file_changes(self, push_event):
        """
        Extract file changes from a push event
        
        Args:
            push_event: PushEvent from GH Archive
            
        Returns:
            List of file change information
        """
        repo_info = self.extract_repo_info(push_event)
        file_changes = []
        
        for commit in repo_info['commits']:
            # Some commits have 'added', 'removed', 'modified' arrays
            # But most just have a list of files in different formats
            
            # Try to extract from commit message or payload
            if 'added' in commit:
                for file_path in commit['added']:
                    if self.is_target_file(file_path):
                        file_changes.append({
                            'action': 'added',
                            'file_path': file_path,
                            'commit_sha': commit.get('sha', ''),
                            'commit_message': commit.get('message', ''),
                            'repo_name': repo_info['repo_name'],
                            'timestamp': repo_info['created_at']
                        })
            
            if 'modified' in commit:
                for file_path in commit['modified']:
                    if self.is_target_file(file_path):
                        file_changes.append({
                            'action': 'modified',
                            'file_path': file_path,
                            'commit_sha': commit.get('sha', ''),
                            'commit_message': commit.get('message', ''),
                            'repo_name': repo_info['repo_name'],
                            'timestamp': repo_info['created_at']
                        })
        
        return file_changes
    
    def is_target_file(self, file_path):
        """Check if file matches our target extensions"""
        for ext in self.target_extensions:
            if file_path.endswith(ext):
                return True
        return False
    
    def extract_repo_info(self, push_event):
        """Extract repository information from push event"""
        repo = push_event.get('repo', {})
        payload = push_event.get('payload', {})
        
        return {
            'repo_name': repo.get('name', ''),
            'repo_id': repo.get('id', ''),
            'commits': payload.get('commits', []),
            'ref': payload.get('ref', ''),
            'created_at': push_event.get('created_at', ''),
            'actor': push_event.get('actor', {}).get('login', '')
        }
    
    def get_unique_repositories(self, push_events):
        """Get unique repositories from push events"""
        repos = {}
        
        for event in push_events:
            repo_info = self.extract_repo_info(event)
            repo_name = repo_info['repo_name']
            
            if repo_name not in repos:
                repos[repo_name] = {
                    'repo_name': repo_name,
                    'repo_id': repo_info['repo_id'],
                    'first_seen': repo_info['created_at'],
                    'actors': set(),
                    'file_changes': []
                }
            
            repos[repo_name]['actors'].add(repo_info['actor'])
            
            # Extract file changes
            file_changes = self.extract_file_changes(event)
            repos[repo_name]['file_changes'].extend(file_changes)
        
        # Convert sets to lists for JSON serialization
        for repo in repos.values():
            repo['actors'] = list(repo['actors'])
        
        return repos


def test_file_extraction():
    """Test extracting file information from GH Archive"""
    print("=== Testing File Extraction ===")
    
    from gharchive_client import GHArchiveClient
    
    # Download and parse archive
    client = GHArchiveClient()
    date_str = "2024-01-01"
    hour = 12
    
    filepath = client.download_archive(date_str, hour)
    if not filepath:
        print("❌ Failed to download archive")
        return False
    
    push_events = client.get_push_events(filepath)
    processor = EventProcessor()
    
    # Extract unique repositories
    repositories = processor.get_unique_repositories(push_events)
    
    print(f"✅ Found {len(repositories)} unique repositories")
    
    # Show repositories with target file changes
    target_repos = []
    for repo_name, repo_info in repositories.items():
        if repo_info['file_changes']:
            target_repos.append((repo_name, repo_info))
    
    print(f"✅ Found {len(target_repos)} repositories with target file changes")
    
    # Show sample files we would download
    print("\n📁 Sample files we would download via GitHub API:")
    for i, (repo_name, repo_info) in enumerate(target_repos[:3]):
        print(f"\n{i+1}. Repository: {repo_name}")
        print(f"   👤 Contributors: {', '.join(repo_info['actors'][:3])}")
        
        for j, file_change in enumerate(repo_info['file_changes'][:3]):
            print(f"   📄 {file_change['action'].upper()}: {file_change['file_path']}")
            print(f"      📝 Commit: {file_change['commit_message'][:60]}...")
            print(f"      🔗 Would download: https://github.com/{repo_name}/blob/{file_change['commit_sha']}/{file_change['file_path']}")
    
    # Clean up
    import os
    os.remove(filepath)
    
    if target_repos:
        print(f"\n✅ Success! Found {len(target_repos)} repositories with {sum(len(r[1]['file_changes']) for r in target_repos)} target files")
        return True
    else:
        print("\n⚠️  No repositories with target file changes found (this can happen with small time windows)")
        return False


if __name__ == "__main__":
    test_file_extraction() 