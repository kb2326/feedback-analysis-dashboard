#!/usr/bin/env python3
"""
Codebase verification script to ensure the project is ready for GitHub.

This script checks for:
- No sensitive data in files
- All required files are present
- Code quality standards
- Documentation completeness
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Sensitive patterns to check for
SENSITIVE_PATTERNS = [
    r'password\s*=\s*["\'][^"\']*["\']',
    r'secret\s*=\s*["\'][^"\']*["\']',
    r'api_key\s*=\s*["\'][^"\']*["\']',
    r'token\s*=\s*["\'][^"\']*["\']',
    r'Santha@1967',
    r'Test Company Nutan Roy',
    r'Humana',
    # Add more patterns as needed
]

# Required files for a complete project
REQUIRED_FILES = [
    'README.md',
    'LICENSE',
    'requirements.txt',
    '.gitignore',
    'setup.py',
    'CHANGELOG.md',
    'CONTRIBUTING.md',
    'Dockerfile',
    'docker-compose.yml',
    '.env.template',
    'app.py',
    'ml/__init__.py',
    'tests/__init__.py',
    'config/database_config.py.template',
    'sql/schema.sql',
    'docs/API.md',
    'docs/DEPLOYMENT.md',
]

# Files that should NOT be present
FORBIDDEN_FILES = [
    '.env',
    'config/database_config.py',
    '*.log',
    '__pycache__',
    '*.pyc',
    '.DS_Store',
    'Thumbs.db',
]

def check_sensitive_data(file_path: Path) -> List[str]:
    """Check a file for sensitive data patterns."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        for pattern in SENSITIVE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                issues.append(f"Sensitive pattern found: {pattern}")
                
    except Exception as e:
        issues.append(f"Error reading file: {e}")
        
    return issues

def check_required_files(project_root: Path) -> List[str]:
    """Check if all required files are present."""
    missing_files = []
    
    for file_path in REQUIRED_FILES:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            
    return missing_files

def check_forbidden_files(project_root: Path) -> List[str]:
    """Check for files that should not be present."""
    found_files = []
    
    for pattern in FORBIDDEN_FILES:
        if '*' in pattern:
            # Handle glob patterns
            for file_path in project_root.rglob(pattern):
                found_files.append(str(file_path.relative_to(project_root)))
        else:
            full_path = project_root / pattern
            if full_path.exists():
                found_files.append(pattern)
                
    return found_files

def check_code_quality(project_root: Path) -> List[str]:
    """Basic code quality checks."""
    issues = []
    
    # Check Python files for basic issues
    python_files = list(project_root.rglob('*.py'))
    
    for py_file in python_files:
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for print statements (should use logging)
            if re.search(r'\bprint\s*\(', content) and 'test_' not in py_file.name:
                issues.append(f"{py_file.relative_to(project_root)}: Contains print statements")
                
            # Check for TODO comments
            if re.search(r'#\s*TODO', content, re.IGNORECASE):
                issues.append(f"{py_file.relative_to(project_root)}: Contains TODO comments")
                
        except Exception as e:
            issues.append(f"Error checking {py_file}: {e}")
            
    return issues

def check_documentation(project_root: Path) -> List[str]:
    """Check documentation completeness."""
    issues = []
    
    # Check README.md
    readme_path = project_root / 'README.md'
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
            
        required_sections = [
            'installation', 'usage', 'features', 'contributing',
            'license', 'requirements'
        ]
        
        for section in required_sections:
            if section.lower() not in readme_content.lower():
                issues.append(f"README.md missing section: {section}")
    else:
        issues.append("README.md not found")
        
    return issues

def main():
    """Main verification function."""
    project_root = Path(__file__).parent.parent
    print(f"Verifying codebase at: {project_root}")
    print("=" * 50)
    
    all_issues = []
    
    # Check for sensitive data
    print("🔍 Checking for sensitive data...")
    sensitive_files = []
    for file_path in project_root.rglob('*'):
        if file_path.is_file() and file_path.suffix in ['.py', '.md', '.txt', '.yml', '.yaml', '.json']:
            issues = check_sensitive_data(file_path)
            if issues:
                sensitive_files.append((file_path.relative_to(project_root), issues))
                
    if sensitive_files:
        print("❌ Sensitive data found:")
        for file_path, issues in sensitive_files:
            print(f"  {file_path}:")
            for issue in issues:
                print(f"    - {issue}")
        all_issues.extend([f"{file_path}: {issue}" for file_path, issues in sensitive_files for issue in issues])
    else:
        print("✅ No sensitive data found")
    
    # Check required files
    print("\n📁 Checking required files...")
    missing_files = check_required_files(project_root)
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        all_issues.extend([f"Missing file: {f}" for f in missing_files])
    else:
        print("✅ All required files present")
    
    # Check forbidden files
    print("\n🚫 Checking for forbidden files...")
    forbidden_files = check_forbidden_files(project_root)
    if forbidden_files:
        print("❌ Forbidden files found:")
        for file_path in forbidden_files:
            print(f"  - {file_path}")
        all_issues.extend([f"Forbidden file: {f}" for f in forbidden_files])
    else:
        print("✅ No forbidden files found")
    
    # Check code quality
    print("\n🔧 Checking code quality...")
    quality_issues = check_code_quality(project_root)
    if quality_issues:
        print("⚠️  Code quality issues:")
        for issue in quality_issues:
            print(f"  - {issue}")
        all_issues.extend(quality_issues)
    else:
        print("✅ Code quality checks passed")
    
    # Check documentation
    print("\n📚 Checking documentation...")
    doc_issues = check_documentation(project_root)
    if doc_issues:
        print("⚠️  Documentation issues:")
        for issue in doc_issues:
            print(f"  - {issue}")
        all_issues.extend(doc_issues)
    else:
        print("✅ Documentation checks passed")
    
    # Final summary
    print("\n" + "=" * 50)
    if all_issues:
        print(f"❌ Verification failed with {len(all_issues)} issues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
        sys.exit(1)
    else:
        print("✅ All checks passed! Codebase is ready for GitHub.")
        sys.exit(0)

if __name__ == "__main__":
    main()