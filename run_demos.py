#!/usr/bin/env python3
"""
Chatterbox TTS Demo Launcher
Simple launcher script for the demo menu
"""

import sys
from pathlib import Path

def main():
    """Launch the demo menu"""
    
    # Check if demo_menu.py exists
    demo_menu_path = Path(__file__).parent / "demo_menu.py"
    
    if not demo_menu_path.exists():
        print("❌ Demo menu not found. Please ensure demo_menu.py is in the same directory.")
        sys.exit(1)
    
    # Import and run the demo menu
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import and run the menu
        from demo_menu import main as run_menu
        run_menu()
        
    except ImportError as e:
        print(f"❌ Failed to import demo menu: {e}")
        print("Please ensure all required files are present.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()