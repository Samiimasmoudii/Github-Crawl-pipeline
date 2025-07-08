"""GH Archive client for downloading and parsing GitHub archive data"""

import requests
import gzip
import json
import os
from datetime import datetime, timedelta
from config import Config

class GHArchiveClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GH-Archive-Pipeline/1.0'
        })
    
    def download_archive(self, date_str, hour, save_path=None):
        """
        Download a single GH Archive file
        
        Args:
            date_str: Date string in format YYYY-MM-DD
            hour: Hour (0-23)
            save_path: Optional path to save the file
        
        Returns:
            Path to downloaded file or None if failed
        """
        url = Config.get_archive_url(date_str, hour)
        filename = f"{date_str}-{hour:02d}.json.gz"
        
        if save_path:
            filepath = os.path.join(save_path, filename)
        else:
            filepath = filename
            
        print(f"Downloading: {url}")
        
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            # Save the compressed file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Downloaded: {filepath}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to download {url}: {e}")
            return None
    
    def parse_archive(self, filepath):
        """
        Parse a GH Archive file and extract events
        
        Args:
            filepath: Path to the .json.gz file
        
        Returns:
            Generator of parsed events
        """
        print(f"Parsing: {filepath}")
        
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        event = json.loads(line.strip())
                        yield event
                    except json.JSONDecodeError as e:
                        print(f"✗ JSON decode error at line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            print(f"✗ Error parsing {filepath}: {e}")
    
    def get_push_events(self, filepath):
        """
        Extract only PushEvent types from archive
        
        Args:
            filepath: Path to the .json.gz file
        
        Returns:
            List of push events
        """
        push_events = []
        
        for event in self.parse_archive(filepath):
            if event.get('type') == 'PushEvent':
                push_events.append(event)
        
        print(f"✓ Found {len(push_events)} push events")
        return push_events
    
    def extract_repo_info(self, push_event):
        """
        Extract repository information from a push event
        
        Args:
            push_event: A PushEvent from GH Archive
        
        Returns:
            Dictionary with repository information
        """
        repo = push_event.get('repo', {})
        payload = push_event.get('payload', {})
        
        return {
            'repo_name': repo.get('name', ''),
            'repo_id': repo.get('id', ''),
            'commits': payload.get('commits', []),
            'ref': payload.get('ref', ''),
            'push_id': payload.get('push_id', ''),
            'size': payload.get('size', 0),
            'created_at': push_event.get('created_at', ''),
            'actor': push_event.get('actor', {}).get('login', '')
        }


def test_basic_functionality():
    """Test basic GH Archive functionality"""
    print("=== Testing GH Archive Client ===")
    
    # Create client
    client = GHArchiveClient()
    
    # Download one hour of data (small test)
    date_str = "2024-01-01"
    hour = 12  # noon UTC
    
    # Download archive
    filepath = client.download_archive(date_str, hour)
    
    if filepath:
        # Parse and get push events
        push_events = client.get_push_events(filepath)
        
        # Show some sample repository info
        print(f"\n=== Sample Repository Info ===")
        for i, event in enumerate(push_events[:5]):  # Show first 5
            repo_info = client.extract_repo_info(event)
            print(f"{i+1}. {repo_info['repo_name']}")
            print(f"   Commits: {len(repo_info['commits'])}")
            print(f"   Actor: {repo_info['actor']}")
            print(f"   Time: {repo_info['created_at']}")
            print()
        
        # Clean up test file
        os.remove(filepath)
        print("✓ Test completed successfully!")
    else:
        print("✗ Test failed - could not download archive")


if __name__ == "__main__":
    test_basic_functionality() 