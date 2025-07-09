"""
GitHub API client for downloading file contents and repository metadata
"""

import requests
import time
import json
import base64
from datetime import datetime
from config import Config

class GitHubAPIClient:
    def __init__(self, token=None):
        self.token = token or Config.GITHUB_TOKEN
        self.base_url = "https://api.github.com"
        self.session = requests.Session()
        
        # Set up authentication
        if self.token:
            self.session.headers.update({
                'Authorization': f'token {self.token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'GH-Archive-Pipeline/1.0'
            })
        else:
            self.session.headers.update({
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'GH-Archive-Pipeline/1.0'
            })
        
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = time.time() + 3600
    
    def _make_request(self, url, params=None):
        """Make a request with rate limiting"""
        # Check rate limit
        if self.rate_limit_remaining <= 10:
            wait_time = self.rate_limit_reset - time.time()
            if wait_time > 0:
                print(f"⏳ Rate limit reached. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
        
        try:
            response = self.session.get(url, params=params)
            
            # Update rate limit info
            self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
            self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', time.time() + 3600))
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                print(f"⚠️  Rate limit exceeded. Reset at {datetime.fromtimestamp(self.rate_limit_reset)}")
                return None
            elif response.status_code == 404:
                print(f"⚠️  Resource not found: {url}")
                return None
            else:
                print(f"⚠️  HTTP {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def get_repository_info(self, repo_name):
        """
        Get repository information including stars, language, etc.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            
        Returns:
            Dictionary with repository information or None
        """
        url = f"{self.base_url}/repos/{repo_name}"
        data = self._make_request(url)
        
        if data:
            return {
                'name': data.get('name'),
                'full_name': data.get('full_name'),
                'description': data.get('description'),
                'language': data.get('language'),
                'stars': data.get('stargazers_count', 0),
                'forks': data.get('forks_count', 0),
                'size': data.get('size', 0),
                'created_at': data.get('created_at'),
                'updated_at': data.get('updated_at'),
                'default_branch': data.get('default_branch', 'main'),
                'private': data.get('private', False)
            }
        return None
    
    def get_repository_contents(self, repo_name, path="", ref=None):
        """
        Get repository contents (files and directories)
        
        Args:
            repo_name: Repository name in format "owner/repo"
            path: Path within repository (default: root)
            ref: Git reference (branch/commit/tag)
            
        Returns:
            List of file/directory information
        """
        url = f"{self.base_url}/repos/{repo_name}/contents/{path}"
        params = {}
        if ref:
            params['ref'] = ref
        
        data = self._make_request(url, params)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]  # Single file
        return []
    
    def download_file_content(self, repo_name, file_path, ref=None):
        """
        Download specific file content
        
        Args:
            repo_name: Repository name in format "owner/repo"
            file_path: Path to file within repository
            ref: Git reference (branch/commit/tag)
            
        Returns:
            Dictionary with file information and content
        """
        url = f"{self.base_url}/repos/{repo_name}/contents/{file_path}"
        params = {}
        if ref:
            params['ref'] = ref
        
        data = self._make_request(url, params)
        
        if data and data.get('type') == 'file':
            # Decode base64 content
            content = base64.b64decode(data.get('content', '')).decode('utf-8', errors='ignore')
            
            return {
                'name': data.get('name'),
                'path': data.get('path'),
                'sha': data.get('sha'),
                'size': data.get('size'),
                'content': content,
                'encoding': data.get('encoding'),
                'download_url': data.get('download_url')
            }
        return None
    
    def find_target_files(self, repo_name, extensions=None, ref=None, max_depth=3):
        """
        Find files with target extensions in repository
        
        Args:
            repo_name: Repository name in format "owner/repo"
            extensions: List of file extensions to search for
            ref: Git reference (branch/commit/tag)
            max_depth: Maximum directory depth to search
            
        Returns:
            List of file paths matching extensions
        """
        extensions = extensions or Config.TARGET_EXTENSIONS
        target_files = []
        
        def search_directory(path="", depth=0):
            if depth > max_depth:
                return
            
            contents = self.get_repository_contents(repo_name, path, ref)
            
            for item in contents:
                if item.get('type') == 'file':
                    file_path = item.get('path', '')
                    if any(file_path.endswith(ext) for ext in extensions):
                        target_files.append({
                            'path': file_path,
                            'name': item.get('name'),
                            'size': item.get('size', 0),
                            'sha': item.get('sha')
                        })
                elif item.get('type') == 'dir' and depth < max_depth:
                    search_directory(item.get('path', ''), depth + 1)
        
        search_directory()
        return target_files
    
    def get_rate_limit_status(self):
        """Get current rate limit status"""
        return {
            'remaining': self.rate_limit_remaining,
            'reset_time': datetime.fromtimestamp(self.rate_limit_reset),
            'authenticated': bool(self.token)
        }


def test_github_client():
    """Test GitHub API client functionality"""
    print("🔍 Testing GitHub API Client...")
    
    # Test without token first (limited rate)
    client = GitHubAPIClient()
    
    # Test repository info
    print("\n📊 Testing repository info...")
    repo_info = client.get_repository_info("octocat/Hello-World")
    
    if repo_info:
        print(f"✅ Repository: {repo_info['full_name']}")
        print(f"   ⭐ Stars: {repo_info['stars']}")
        print(f"   🎯 Language: {repo_info['language']}")
        print(f"   📝 Description: {repo_info['description']}")
    else:
        print("❌ Failed to get repository info")
    
    # Test file discovery
    print("\n📁 Testing file discovery...")
    target_files = client.find_target_files("octocat/Hello-World", ['.txt', '.md'])
    
    if target_files:
        print(f"✅ Found {len(target_files)} target files:")
        for file_info in target_files[:3]:
            print(f"   📄 {file_info['path']} ({file_info['size']} bytes)")
    else:
        print("❌ No target files found")
    
    # Test file download
    print("\n📥 Testing file download...")
    if target_files:
        file_content = client.download_file_content("octocat/Hello-World", target_files[0]['path'])
        
        if file_content:
            print(f"✅ Downloaded: {file_content['name']}")
            print(f"   📄 Size: {file_content['size']} bytes")
            print(f"   📝 Content preview: {file_content['content'][:100]}...")
        else:
            print("❌ Failed to download file")
    
    # Show rate limit status
    rate_limit = client.get_rate_limit_status()
    print(f"\n⏱️  Rate limit: {rate_limit['remaining']} requests remaining")
    print(f"   🔑 Authenticated: {rate_limit['authenticated']}")
    
    print("\n✅ GitHub API Client test completed!")


if __name__ == "__main__":
    test_github_client() 