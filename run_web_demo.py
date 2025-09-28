#!/usr/bin/env python3
"""
Alternative way to run the web demo without installing the package.
This script adds the project root to the Python path so imports work.
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now run the web demo
if __name__ == "__main__":
    import subprocess
    import sys
    
    # Run streamlit with the web demo
    subprocess.run([sys.executable, "-m", "streamlit", "run", "demo/web_demo.py"] + sys.argv[1:])
