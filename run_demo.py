#!/usr/bin/env python3
"""
Simple runner for Financial Forecast Demo
Usage: python3 run_demo.py [command] [args]
"""
import sys
import subprocess

def main():
    """Run the demo with python3 explicitly"""
    if len(sys.argv) == 1:
        # No arguments, show help
        cmd = ["python3", "-m", "delphi.scripts.run", "--help"]
    else:
        # Pass through all arguments
        cmd = ["python3", "-m", "delphi.scripts.run"] + sys.argv[1:]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)

if __name__ == '__main__':
    main() 