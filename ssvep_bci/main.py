#!/usr/bin/env python3
"""
SSVEP BCI Main Entry Point

This is the main entry point for the SSVEP Brain-Computer Interface.

For Phase 1 (CLI testing), run:
    python main.py --cli

For Phase 2 (GUI), run:
    python main.py

The system connects to an OpenBCI Cyton board and performs real-time
SSVEP classification using CCA (Canonical Correlation Analysis).

Target frequencies (matching Arduino LED flicker rates):
- 8.57 Hz (Pin D2)
- 10.0 Hz (Pin D3)
- 12.0 Hz (Pin D4)
- 15.0 Hz (Pin D5)
"""

import sys
import argparse
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent))


def run_cli_test(args):
    """Run the command-line test interface."""
    from cli_test import main as cli_main

    # Build argv for cli_test
    cli_argv = []

    if args.port:
        cli_argv.extend(['--port', args.port])
    if args.duration:
        cli_argv.extend(['--duration', str(args.duration)])
    if args.target:
        cli_argv.extend(['--target', str(args.target)])
    if args.confidence:
        cli_argv.extend(['--confidence', str(args.confidence)])
    if args.margin:
        cli_argv.extend(['--margin', str(args.margin)])
    if args.no_log:
        cli_argv.append('--no-log')

    # Replace sys.argv for cli_test's argparse
    original_argv = sys.argv
    sys.argv = ['cli_test.py'] + cli_argv

    try:
        return cli_main()
    finally:
        sys.argv = original_argv


def run_gui():
    """Run the GUI application (Phase 2)."""
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.main_window import MainWindow
    except ImportError:
        print("PyQt6 not installed. Please install it with:")
        print("  pip install PyQt6")
        print("\nAlternatively, run in CLI mode:")
        print("  python main.py --cli")
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("SSVEP BCI")
    app.setOrganizationName("BCI Lab")

    window = MainWindow()
    window.show()

    return app.exec()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SSVEP Brain-Computer Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Launch GUI (Phase 2)
  python main.py --cli                  # Run CLI test
  python main.py --cli --port COM3      # CLI with real Cyton
  python main.py --cli --target 12.0    # CLI with synthetic 12 Hz
        """
    )

    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run in command-line mode instead of GUI'
    )

    parser.add_argument(
        '--port', '-p',
        type=str,
        default=None,
        help='Serial port for Cyton (e.g., COM3)'
    )

    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=30.0,
        help='Duration for CLI test in seconds'
    )

    parser.add_argument(
        '--target', '-t',
        type=float,
        default=10.0,
        help='Target frequency for synthetic data'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.55,
        help='Confidence threshold'
    )

    parser.add_argument(
        '--margin',
        type=float,
        default=0.15,
        help='Margin threshold'
    )

    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Disable logging'
    )

    args = parser.parse_args()

    if args.cli:
        return run_cli_test(args)
    else:
        return run_gui()


if __name__ == "__main__":
    sys.exit(main())
