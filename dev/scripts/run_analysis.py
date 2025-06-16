#!/usr/bin/env python3
"""
Standalone script to run migration analysis.
Can be used independently or as part of the pipeline.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer import MigrationAnalyzer
from utils.common import setup_logging, Timer, start_run, end_run
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description='Run Migration Analysis')
    parser.add_argument('--config', '-c', required=True, help='Configuration file')
    parser.add_argument('--tables', '-t', help='Comma-separated table names')
    parser.add_argument('--output', '-o', help='Output directory override')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--range', help='Row range (start,end)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        start_run()
        
        # Parse table filter
        table_filter = None
        if args.tables:
            table_filter = [t.strip() for t in args.tables.split(',')]
        
        # Parse range
        row_range = None
        if args.range:
            start_row, end_row = map(int, args.range.split(','))
            row_range = (start_row, end_row)
        
        # Run analysis
        logger.info("Starting migration analysis...")
        
        with Timer() as timer:
            analyzer = MigrationAnalyzer(args.config)
            result_file = analyzer.run_analysis(
                table_filter=table_filter,
                output_override=args.output,
                row_range=row_range
            )
        
        logger.info(f"Analysis completed in {timer.format_duration(timer.elapsed)}")
        logger.info(f"Results saved to: {result_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)
    finally:
        end_run()


if __name__ == '__main__':
    main()