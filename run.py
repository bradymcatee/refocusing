#!/usr/bin/env python3
"""
Simple launcher for Interactive Refocusing Web App
Usage:
    python run.py              # Run development server
    python run.py --prod       # Run production server with gunicorn
"""

import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Interactive Refocusing Web App")
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Run in production mode with gunicorn (requires: pip install gunicorn)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument(
        "--workers", type=int, default=4, help="Worker processes for production mode"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists("checkpoints/best_model.pth"):
        print("ERROR: Model checkpoint not found at checkpoints/best_model.pth")
        print("Please ensure you have the trained model file.")
        sys.exit(1)

    if args.prod:
        # Production mode with gunicorn
        try:
            import gunicorn.app.base
        except ImportError:
            print("ERROR: gunicorn not installed.")
            print("Install it with: pip install gunicorn")
            sys.exit(1)

        print(f"Starting production server on {args.host}:{args.port}")
        print(f"Workers: {args.workers}")
        print("Press Ctrl+C to stop\n")

        os.system(
            f"gunicorn --bind {args.host}:{args.port} --workers {args.workers} --timeout 120 app:app"
        )
    else:
        # Development mode
        print(f"Starting development server on {args.host}:{args.port}")
        print("Press Ctrl+C to stop\n")

        # Set environment and run app
        os.environ["FLASK_ENV"] = "development"
        os.environ["FLASK_DEBUG"] = "1"

        from app import app, load_model

        # Load the model before starting server
        print("Loading model...")
        if not load_model():
            print(
                "ERROR: Failed to load model. Please check checkpoints/best_model.pth"
            )
            sys.exit(1)
        print("Model loaded successfully!\n")

        app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
