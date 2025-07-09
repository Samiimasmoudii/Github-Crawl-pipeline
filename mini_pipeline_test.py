#!/usr/bin/env python3
"""
Mini pipeline test - combines GH Archive discovery with GitHub API file downloads
"""

import os
import json
from datetime import datetime
from repository_finder import RepositoryFinder
from github_client import GitHubAPIClient
from config import Config

def mini_pipeline_test():
    """Test the complete pipeline flow"""
    print("🚀 Starting Mini Pipeline Test...")
    print("="*60)
    
    # Step 1: Find repositories from GH Archive
    print("\n📦 Step 1: Finding repositories from GH Archive...")
    finder = RepositoryFinder()
    
    # Use a small time window for testing
    repositories = finder.find_repositories_from_archive("2024-01-01", 12)
    target_repos = finder.filter_target_repositories(repositories)
    
    print(f"✅ Found {len(target_repos)} target repositories")
    
    if not target_repos:
        print("❌ No repositories found. Try a different date/time.")
        return False
    
    # Step 2: Initialize GitHub API client
    print("\n🔑 Step 2: Initializing GitHub API client...")
    github_client = GitHubAPIClient()
    
    rate_limit = github_client.get_rate_limit_status()
    print(f"⏱️  Rate limit: {rate_limit['remaining']} requests remaining")
    print(f"🔐 Authenticated: {rate_limit['authenticated']}")
    
    # Step 3: Validate repositories and download files
    print("\n📊 Step 3: Validating repositories and downloading files...")
    
    successful_downloads = []
    failed_repos = []
    
    # Test with first 3 repositories to avoid rate limiting
    for i, repo in enumerate(target_repos[:3]):
        repo_name = repo['repo_name']
        print(f"\n--- Processing {i+1}/3: {repo_name} ---")
        
        # Get repository info
        repo_info = github_client.get_repository_info(repo_name)
        
        if not repo_info:
            print(f"❌ Failed to get info for {repo_name}")
            failed_repos.append(repo_name)
            continue
        
        # Check if repository meets criteria
        if repo_info['stars'] < Config.MIN_STARS:
            print(f"⚠️  {repo_name} has only {repo_info['stars']} stars (min: {Config.MIN_STARS})")
            failed_repos.append(repo_name)
            continue
        
        print(f"✅ {repo_name}: {repo_info['stars']} stars, {repo_info['language']}")
        
        # Find target files
        target_files = github_client.find_target_files(
            repo_name, 
            Config.TARGET_EXTENSIONS,
            max_depth=2  # Limit depth for testing
        )
        
        if not target_files:
            print(f"⚠️  No target files found in {repo_name}")
            failed_repos.append(repo_name)
            continue
        
        print(f"📁 Found {len(target_files)} target files")
        
        # Download first file as test
        if target_files:
            file_info = target_files[0]
            print(f"📥 Downloading: {file_info['path']}")
            
            file_content = github_client.download_file_content(
                repo_name, 
                file_info['path']
            )
            
            if file_content:
                successful_downloads.append({
                    'repo_name': repo_name,
                    'repo_info': repo_info,
                    'file_info': file_info,
                    'file_content': file_content,
                    'archive_repo': repo  # Original GH Archive data
                })
                print(f"✅ Downloaded {file_content['name']} ({file_content['size']} bytes)")
                
                # Show content preview
                content_preview = file_content['content'][:200].replace('\n', '\\n')
                print(f"📝 Content preview: {content_preview}...")
            else:
                print(f"❌ Failed to download {file_info['path']}")
                failed_repos.append(repo_name)
    
    # Step 4: Results summary
    print("\n📊 Step 4: Results Summary")
    print("="*60)
    
    print(f"✅ Successful downloads: {len(successful_downloads)}")
    print(f"❌ Failed repositories: {len(failed_repos)}")
    
    if successful_downloads:
        print("\n🎉 Successfully downloaded files:")
        for download in successful_downloads:
            print(f"  📄 {download['repo_name']}/{download['file_info']['path']}")
            print(f"     ⭐ {download['repo_info']['stars']} stars")
            print(f"     🎯 {download['repo_info']['language']}")
            print(f"     📝 {download['file_content']['size']} bytes")
    
    if failed_repos:
        print(f"\n⚠️  Failed repositories: {', '.join(failed_repos)}")
    
    # Step 5: Create sample output structure
    if successful_downloads:
        print("\n📂 Step 5: Creating sample output structure...")
        Config.create_output_dirs()
        
        # Save sample files
        for download in successful_downloads:
            repo_name = download['repo_name'].replace('/', '_')
            file_name = download['file_info']['name']
            
            # Create repo directory
            repo_dir = os.path.join(Config.EXTRACTED_FILES_DIR, repo_name)
            os.makedirs(repo_dir, exist_ok=True)
            
            # Save file
            file_path = os.path.join(repo_dir, file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(download['file_content']['content'])
            
            print(f"💾 Saved: {file_path}")
        
        # Save metadata
        metadata_file = os.path.join(Config.OUTPUT_DIR, "sample_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(successful_downloads, f, indent=2, default=str)
        
        print(f"💾 Metadata saved: {metadata_file}")
    
    # Final rate limit check
    final_rate_limit = github_client.get_rate_limit_status()
    print(f"\n⏱️  Final rate limit: {final_rate_limit['remaining']} requests remaining")
    
    success = len(successful_downloads) > 0
    if success:
        print("\n🎉 Mini pipeline test completed successfully!")
        print(f"✅ Downloaded {len(successful_downloads)} files from {len(set(d['repo_name'] for d in successful_downloads))} repositories")
    else:
        print("\n❌ Mini pipeline test failed - no files downloaded")
    
    return success

if __name__ == "__main__":
    success = mini_pipeline_test()
    exit(0 if success else 1) 