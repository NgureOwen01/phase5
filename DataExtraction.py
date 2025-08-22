#!/usr/bin/env python3
"""
Satellite Data Downloader
Downloads satellite data from Hugging Face for cropland classification
Run this script first before running our notebook.
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

def install_requirements():
    """Install required packages"""
    packages = ["huggingface_hub", "tqdm", "requests"]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"{sys.executable} -m pip install {package}")

def download_file(url, filepath):
    """Download a single file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {filepath.name}: {e}")
        return False

def flatten_path(original_path):
    """
    Remove 'data/' prefix from file paths to flatten the structure
    e.g., 'data/train/image.tif' -> 'train/image.tif'
    """
    path_parts = Path(original_path).parts
    if path_parts and path_parts[0].lower() == 'data':
        # Remove the first 'data' component
        return str(Path(*path_parts[1:])) if len(path_parts) > 1 else ""
    return original_path

def download_satellite_data():
    """Download all satellite data files"""
    
    # Install requirements
    print("Checking requirements...")
    install_requirements()
    
    from huggingface_hub import hf_hub_url, list_repo_files
    
    # Configuration
    dataset_id = "cnyagaka/satellite-data"
    download_dir = Path("Data")
    
    print(f"Downloading from: {dataset_id}")
    print(f"Saving to: {download_dir.absolute()}")
    
    # Create directory
    download_dir.mkdir(exist_ok=True)
    
    try:
        # Get file list
        print("Fetching file list...")
        files = list_repo_files(dataset_id, repo_type="dataset")
        data_files = [f for f in files if not f.startswith('.') and not f.startswith('README')]
        
        print(f"Found {len(data_files)} files to download")
        
        # Download files
        successful = 0
        for i, filename in enumerate(data_files, 1):
            print(f"\n[{i}/{len(data_files)}] {filename}")
            
            # Flatten the path by removing 'data/' prefix
            flattened_filename = flatten_path(filename)
            
            # Skip empty paths (files that were only 'data' with no subpath)
            if not flattened_filename:
                print(f"Skipping empty path")
                continue
            
            filepath = download_dir / flattened_filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if exists
            if filepath.exists() and filepath.stat().st_size > 0:
                print(f"Skipping (already exists): {flattened_filename}")
                successful += 1
                continue
            
            # Download
            print(f"Downloading to: {flattened_filename}")
            file_url = hf_hub_url(dataset_id, filename, repo_type="dataset")
            if download_file(file_url, filepath):
                successful += 1
        
        # Summary
        print(f"\nDownload complete: {successful}/{len([f for f in data_files if flatten_path(f)])} files")
        
        # Show downloaded files
        print("\nDownloaded files:")
        for file in sorted(download_dir.rglob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  {file.relative_to(download_dir)} ({size_mb:.1f} MB)")
        
        print(f"\nData ready! You can now run your notebook.")
        return successful == len([f for f in data_files if flatten_path(f)])
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print(" Satellite Data Downloader")
    print("=" * 40)
    
    success = download_satellite_data()
    
    if success:
        print("\n All files downloaded successfully!")
    else:
        print("\n Some files failed to download")
    
    print("\nNext step: Run your cropland classification notebook")
