#!/usr/bin/env python3
"""
Setup verification script for the Change Detection System
"""

import os
import sys
import importlib
from pathlib import Path

def check_imports():
    """Check if all required packages are installed"""
    required_packages = [
        'ee', 'geemap', 'geopandas', 'rasterio', 
        'matplotlib', 'numpy', 'yaml', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    return True

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        'config', 'src', 'data/exports', 'logs'
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ (creating...)")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ {dir_path}/ (created)")
    
    return True

def check_config():
    """Check if config file exists and is valid"""
    config_path = "config/config.yaml"
    if Path(config_path).exists():
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration file loaded")
            return True
        except Exception as e:
            print(f"❌ Configuration file error: {e}")
            return False
    else:
        print(f"❌ Configuration file not found: {config_path}")
        return False

def check_gee_auth():
    """Check Google Earth Engine authentication"""
    try:
        import ee
        ee.Initialize(project='composite-depot-428316-f1')
        print("✅ Google Earth Engine authenticated")
        return True
    except Exception as e:
        print(f"❌ Google Earth Engine authentication failed: {e}")
        print("Run: earthengine authenticate")
        return False

def main():
    """Main setup check function"""
    print("🔍 Change Detection System - Setup Check")
    print("=" * 50)
    
    checks = [
        ("Package Dependencies", check_imports),
        ("Directory Structure", check_directories),
        ("Configuration File", check_config),
        ("Google Earth Engine", check_gee_auth)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All checks passed! System ready to use.")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)