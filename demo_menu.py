#!/usr/bin/env python3
"""
Chatterbox TTS Demo Menu
Interactive CLI menu for running all available demos and tests
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class DemoMenu:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.venv_path = self.base_dir / "chatterbox_venv"
        self.demos = {
            "1": {
                "name": "📋 Installation Test",
                "file": "installation_test.py",
                "description": "Test all dependencies and system compatibility",
                "duration": "~30 seconds",
                "requirements": "None (lightweight test)"
            },
            "2": {
                "name": "⚡ Minimal Demo",
                "file": "minimal_tts_demo.py", 
                "description": "Quick overview without model download",
                "duration": "~10 seconds",
                "requirements": "None (no model loading)"
            },
            "3": {
                "name": "🌐 Web Interface Test",
                "file": "web_interface_test.py",
                "description": "Test Gradio web interface capabilities",
                "duration": "~15 seconds", 
                "requirements": "Gradio installed"
            },
            "4": {
                "name": "🎬 Quick TTS Demo",
                "file": "quick_tts_demo.py",
                "description": "Full TTS with actual audio generation",
                "duration": "~2-5 minutes",
                "requirements": "~2GB model download (first run)"
            },
            "5": {
                "name": "📱 Example TTS (Official)",
                "file": "example_tts.py",
                "description": "Official basic TTS example",
                "duration": "~2-5 minutes",
                "requirements": "~2GB model download (first run)"
            },
            "6": {
                "name": "🍎 Example for Mac",
                "file": "example_for_mac.py", 
                "description": "Mac-optimized TTS with device detection",
                "duration": "~2-5 minutes",
                "requirements": "~2GB model download (first run)"
            },
            "7": {
                "name": "🔄 Voice Conversion Demo",
                "file": "example_vc.py",
                "description": "Voice conversion example (needs audio files)",
                "duration": "~2-5 minutes", 
                "requirements": "Audio files + model download"
            },
            "8": {
                "name": "🌐 Gradio TTS App",
                "file": "gradio_tts_app.py",
                "description": "Full web interface for TTS",
                "duration": "Runs until stopped",
                "requirements": "~2GB model download + web browser"
            },
            "9": {
                "name": "🔄 Gradio VC App", 
                "file": "gradio_vc_app.py",
                "description": "Full web interface for voice conversion",
                "duration": "Runs until stopped",
                "requirements": "Model download + web browser"
            }
        }

    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """Print the main header"""
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}🎭 CHATTERBOX TTS DEMO MENU{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print()

    def print_system_info(self):
        """Print system information"""
        print(f"{Colors.BOLD}🖥️  System Information:{Colors.END}")
        print(f"   🐍 Python: {sys.version.split()[0]}")
        print(f"   💻 Platform: {platform.system()} {platform.machine()}")
        print(f"   📁 Working Directory: {self.base_dir}")
        
        # Check for virtual environment
        venv_status = "✅ Found" if self.venv_path.exists() else "❌ Not found"
        print(f"   🐍 Virtual Environment: {venv_status}")
        
        # Check for key demo files
        demo_files = ["installation_test.py", "quick_tts_demo.py", "gradio_tts_app.py"]
        missing_files = [f for f in demo_files if not (self.base_dir / f).exists()]
        
        if missing_files:
            print(f"   ⚠️  Missing files: {', '.join(missing_files)}")
        else:
            print(f"   ✅ All demo files present")
        print()

    def print_menu(self):
        """Print the main menu options"""
        print(f"{Colors.BOLD}📋 Available Demos:{Colors.END}")
        print()
        
        for key, demo in self.demos.items():
            status_icon = "✅" if (self.base_dir / demo["file"]).exists() else "❌"
            print(f"{Colors.BOLD}{key}.{Colors.END} {status_icon} {demo['name']}")
            print(f"     📋 {demo['description']}")
            print(f"     ⏱️  Duration: {demo['duration']}")
            print(f"     📦 Requirements: {demo['requirements']}")
            print()
        
        print(f"{Colors.BOLD}🛠️  Utility Options:{Colors.END}")
        print(f"{Colors.BOLD}s.{Colors.END} 📊 Show system information")
        print(f"{Colors.BOLD}i.{Colors.END} ℹ️  Show installation instructions") 
        print(f"{Colors.BOLD}h.{Colors.END} ❓ Show help and usage tips")
        print(f"{Colors.BOLD}q.{Colors.END} 🚪 Quit")
        print()

    def run_demo(self, demo_key):
        """Run a specific demo"""
        if demo_key not in self.demos:
            print(f"{Colors.RED}❌ Invalid demo selection: {demo_key}{Colors.END}")
            return False
            
        demo = self.demos[demo_key]
        demo_file = self.base_dir / demo["file"]
        
        if not demo_file.exists():
            print(f"{Colors.RED}❌ Demo file not found: {demo_file}{Colors.END}")
            return False
        
        print(f"{Colors.BOLD}{Colors.GREEN}🚀 Running: {demo['name']}{Colors.END}")
        print(f"📋 {demo['description']}")
        print(f"⏱️  Expected duration: {demo['duration']}")
        print(f"📦 Requirements: {demo['requirements']}")
        print()
        
        # Ask for confirmation for longer demos
        if "minutes" in demo['duration'] or "download" in demo['requirements'].lower():
            response = input(f"{Colors.YELLOW}⚠️  This demo requires model download (~2GB). Continue? (y/N): {Colors.END}")
            if response.lower() not in ['y', 'yes']:
                print("Demo cancelled.")
                return False
        
        print(f"{Colors.CYAN}{'='*50}{Colors.END}")
        
        try:
            # Prepare command
            if self.venv_path.exists():
                if os.name == 'nt':  # Windows
                    activate_cmd = f"{self.venv_path}/Scripts/activate"
                    cmd = f'"{activate_cmd}" && python "{demo_file}"'
                else:  # Unix/Linux/Mac
                    activate_cmd = f"source {self.venv_path}/bin/activate"
                    cmd = f'{activate_cmd} && python "{demo_file}"'
            else:
                cmd = f'python "{demo_file}"'
            
            # Run the demo
            result = subprocess.run(cmd, shell=True, cwd=self.base_dir)
            
            print(f"{Colors.CYAN}{'='*50}{Colors.END}")
            
            if result.returncode == 0:
                print(f"{Colors.GREEN}✅ Demo completed successfully!{Colors.END}")
            else:
                print(f"{Colors.RED}❌ Demo failed with exit code: {result.returncode}{Colors.END}")
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️  Demo interrupted by user{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}❌ Error running demo: {e}{Colors.END}")
        
        input(f"\n{Colors.BLUE}Press Enter to return to menu...{Colors.END}")
        return True

    def show_installation_instructions(self):
        """Show installation instructions"""
        self.clear_screen()
        self.print_header()
        
        print(f"{Colors.BOLD}📦 Installation Instructions{Colors.END}")
        print(f"{Colors.CYAN}{'='*40}{Colors.END}")
        print()
        
        if not self.venv_path.exists():
            print(f"{Colors.YELLOW}⚠️  Virtual environment not found. Setting up...{Colors.END}")
            print()
            print("🔧 Setup commands:")
            print("   1. Create virtual environment:")
            print(f"      python3 -m venv {self.venv_path}")
            print()
            print("   2. Activate virtual environment:")
            if os.name == 'nt':
                print(f"      {self.venv_path}\\Scripts\\activate")
            else:
                print(f"      source {self.venv_path}/bin/activate")
            print()
            print("   3. Install Chatterbox TTS:")
            print("      pip install chatterbox-tts")
            print()
            print("   4. Install Gradio (for web interfaces):")
            print("      pip install gradio")
            print()
            
            response = input(f"{Colors.BLUE}Would you like me to set this up automatically? (y/N): {Colors.END}")
            if response.lower() in ['y', 'yes']:
                self.setup_environment()
        else:
            print(f"{Colors.GREEN}✅ Virtual environment found!{Colors.END}")
            print()
            print("🎯 You're ready to run demos!")
            print()
            print("💡 Additional installation options:")
            print("   • PyPI: pip install chatterbox-tts")
            print("   • From source: git clone + pip install -e .")
            print()
        
        input(f"{Colors.BLUE}Press Enter to return to menu...{Colors.END}")

    def setup_environment(self):
        """Automatically set up the virtual environment"""
        print(f"\n{Colors.CYAN}🔧 Setting up environment...{Colors.END}")
        
        try:
            # Create virtual environment
            print("📦 Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
            
            # Install packages
            print("📥 Installing Chatterbox TTS...")
            if os.name == 'nt':
                pip_cmd = str(self.venv_path / "Scripts" / "pip")
            else:
                pip_cmd = str(self.venv_path / "bin" / "pip")
            
            subprocess.run([pip_cmd, "install", "chatterbox-tts"], check=True)
            subprocess.run([pip_cmd, "install", "gradio"], check=True)
            
            print(f"{Colors.GREEN}✅ Environment setup completed!{Colors.END}")
            
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}❌ Setup failed: {e}{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.END}")

    def show_help(self):
        """Show help and usage tips"""
        self.clear_screen()
        self.print_header()
        
        print(f"{Colors.BOLD}❓ Help & Usage Tips{Colors.END}")
        print(f"{Colors.CYAN}{'='*30}{Colors.END}")
        print()
        
        print(f"{Colors.BOLD}🎯 Getting Started:{Colors.END}")
        print("   1. Start with the 📋 Installation Test (option 1)")
        print("   2. Try the ⚡ Minimal Demo (option 2) for a quick overview")
        print("   3. Run 🎬 Quick TTS Demo (option 4) for actual audio generation")
        print()
        
        print(f"{Colors.BOLD}🎭 Demo Categories:{Colors.END}")
        print("   🔍 Testing: Options 1-3 (no model download required)")
        print("   🎵 Audio Generation: Options 4-6 (requires model download)")  
        print("   🌐 Web Interfaces: Options 8-9 (browser-based)")
        print("   🔄 Voice Conversion: Option 7 (advanced feature)")
        print()
        
        print(f"{Colors.BOLD}⚡ Performance Notes:{Colors.END}")
        print("   • First run: ~2GB model download (5-10 minutes)")
        print("   • Subsequent runs: Much faster (30-60 seconds)")
        print("   • Apple Silicon (M1/M2/M3): Native MPS acceleration")
        print("   • NVIDIA GPU: CUDA acceleration (if available)")
        print("   • CPU fallback: Works but slower")
        print()
        
        print(f"{Colors.BOLD}🎤 Voice Cloning Tips:{Colors.END}")
        print("   • Use 10-15 seconds of clear speech")
        print("   • Single speaker, minimal background noise")
        print("   • WAV, MP3, or FLAC formats supported")
        print("   • Conversational speech works best")
        print()
        
        print(f"{Colors.BOLD}🔗 Resources:{Colors.END}")
        print("   📖 Documentation: https://github.com/jlwainwright/chatterbox")
        print("   🎵 Demo Samples: https://resemble-ai.github.io/chatterbox_demopage/")
        print("   🤗 Try Online: https://huggingface.co/spaces/ResembleAI/Chatterbox")
        print("   💬 Discord: https://discord.gg/rJq9cRJBJ6")
        print()
        
        input(f"{Colors.BLUE}Press Enter to return to menu...{Colors.END}")

    def show_detailed_system_info(self):
        """Show detailed system information"""
        self.clear_screen() 
        self.print_header()
        
        print(f"{Colors.BOLD}📊 System Information{Colors.END}")
        print(f"{Colors.CYAN}{'='*30}{Colors.END}")
        print()
        
        # Python info
        print(f"{Colors.BOLD}🐍 Python Environment:{Colors.END}")
        print(f"   Version: {sys.version}")
        print(f"   Executable: {sys.executable}")
        print(f"   Platform: {platform.platform()}")
        print()
        
        # Check PyTorch availability
        try:
            import torch
            print(f"{Colors.BOLD}🔥 PyTorch:{Colors.END}")
            print(f"   Version: {torch.__version__}")
            print(f"   CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   CUDA Device: {torch.cuda.get_device_name()}")
            print(f"   MPS Available: {torch.backends.mps.is_available()}")
            print()
        except ImportError:
            print(f"{Colors.RED}❌ PyTorch not installed{Colors.END}")
            print()
        
        # Virtual environment status
        print(f"{Colors.BOLD}🐍 Virtual Environment:{Colors.END}")
        if self.venv_path.exists():
            print(f"   Status: {Colors.GREEN}✅ Found{Colors.END}")
            print(f"   Path: {self.venv_path}")
            
            # Check installed packages
            try:
                if os.name == 'nt':
                    pip_cmd = str(self.venv_path / "Scripts" / "pip")
                else:
                    pip_cmd = str(self.venv_path / "bin" / "pip")
                
                result = subprocess.run([pip_cmd, "list"], capture_output=True, text=True)
                if "chatterbox-tts" in result.stdout:
                    print(f"   Chatterbox TTS: {Colors.GREEN}✅ Installed{Colors.END}")
                else:
                    print(f"   Chatterbox TTS: {Colors.RED}❌ Not installed{Colors.END}")
                    
                if "gradio" in result.stdout:
                    print(f"   Gradio: {Colors.GREEN}✅ Installed{Colors.END}")
                else:
                    print(f"   Gradio: {Colors.YELLOW}⚠️  Not installed{Colors.END}")
                    
            except Exception:
                print(f"   Package check: {Colors.YELLOW}⚠️  Could not verify{Colors.END}")
        else:
            print(f"   Status: {Colors.RED}❌ Not found{Colors.END}")
            print(f"   Expected path: {self.venv_path}")
        
        print()
        
        # File system info
        print(f"{Colors.BOLD}📁 Demo Files:{Colors.END}")
        total_files = len(self.demos)
        existing_files = sum(1 for demo in self.demos.values() if (self.base_dir / demo["file"]).exists())
        print(f"   Total demos: {total_files}")
        print(f"   Available: {Colors.GREEN}{existing_files}{Colors.END}")
        if existing_files < total_files:
            print(f"   Missing: {Colors.RED}{total_files - existing_files}{Colors.END}")
        print()
        
        input(f"{Colors.BLUE}Press Enter to return to menu...{Colors.END}")

    def run(self):
        """Main menu loop"""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_system_info()
            self.print_menu()
            
            choice = input(f"{Colors.BOLD}Enter your choice: {Colors.END}").strip().lower()
            
            if choice == 'q':
                print(f"\n{Colors.GREEN}👋 Thanks for using Chatterbox TTS! Goodbye!{Colors.END}")
                break
            elif choice == 's':
                self.show_detailed_system_info()
            elif choice == 'i':
                self.show_installation_instructions()
            elif choice == 'h':
                self.show_help()
            elif choice in self.demos:
                self.run_demo(choice)
            else:
                print(f"{Colors.RED}❌ Invalid choice: {choice}{Colors.END}")
                time.sleep(1)

def main():
    """Entry point"""
    try:
        menu = DemoMenu()
        menu.run()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Menu interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.END}")

if __name__ == "__main__":
    main()