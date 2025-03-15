#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Analysis Script
"""

import os
import sys
import argparse
import signal
import traceback

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.image_analyzer import ImageAnalyzer

# Global variables
analyzer = None

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print("\n⚠️ Interrupt signal received, safely stopping the program...")
    if analyzer:
        analyzer.stop_flag = True
        print("Program will generate reports with partial results before exiting.")
    else:
        print("Program not yet initialized, exiting directly.")
        sys.exit(1)

def main():
    """Main function"""
    global analyzer
    
    # Set signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='Image Analysis Tool')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Source directory for images (required)')
    parser.add_argument('--output-dir', type=str, default='results/analysis',
                        help='Output directory for analysis results (default: results/analysis)')
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads (default: 4)')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Batch size for processing images (default: 0, process all images at once)')
    parser.add_argument('--max-memory', type=float, default=0,
                        help='Maximum memory usage in GB (default: 0, no limit)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Image Analysis Tool")
    print("=" * 50)
    print(f"Image source directory: {args.image_dir}")
    print(f"Analysis output directory: {args.output_dir}")
    print(f"Number of threads: {args.threads}")
    if args.batch_size > 0:
        print(f"Batch size: {args.batch_size} images")
    else:
        print("Batch processing: Off (processing all images at once)")
    if args.max_memory > 0:
        print(f"Maximum memory limit: {args.max_memory} GB")
    else:
        print("Maximum memory limit: Off (no memory limit)")
    print("-" * 50)
    print("Tip: Press Ctrl+C to stop processing at any time")
    print("     Reports will still be generated with partial results")
    print("-" * 50)
    
    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f"❌ Error: Image directory '{args.image_dir}' does not exist!")
        return 1
    
    try:
        # Check if psutil is installed
        try:
            import psutil
        except ImportError:
            if args.max_memory > 0:
                print("⚠️ Warning: psutil package not installed, cannot monitor memory usage")
                print("Please install with 'pip install psutil'")
                args.max_memory = 0
        
        # Create and run analyzer
        analyzer = ImageAnalyzer(
            image_folder=args.image_dir,
            output_folder=args.output_dir,
            num_threads=args.threads,
            batch_size=args.batch_size,
            max_memory_gb=args.max_memory
        )
        
        success = analyzer.analyze()
        
        if success:
            print("\n✅ Image analysis completed successfully!")
            print(f"📊 Analysis results and charts saved to: {args.output_dir}")
            print(f"📄 Summary report: {os.path.join(args.output_dir, 'summary_report.html')}")
            print(f"📈 Charts directory: {os.path.join(args.output_dir, 'charts')}")
            print(f"📋 Detailed data: {os.path.join(args.output_dir, 'result.csv')}")
        else:
            print("\n⚠️ Image analysis was interrupted or incomplete!")
            if os.path.exists(os.path.join(args.output_dir, "result.csv")):
                print(f"💡 Partial results have been saved in {args.output_dir}")
                print(f"📄 Summary report: {os.path.join(args.output_dir, 'summary_report.html')}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
        # Check if results were saved
        if os.path.exists(os.path.join(args.output_dir, "result.csv")):
            print(f"💡 Partial results have been saved in {args.output_dir}")
        return 1
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        print("Detailed error information:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unhandled error: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 