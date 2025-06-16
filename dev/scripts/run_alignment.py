#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aligner import DataAligner
from utils.common import setup_logging, Timer, start_run, end_run
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description='Run Data Alignment')
    parser.add_argument('--config', '-c', required=True, help='Configuration file')
    parser.add_argument('--metadata-file', required=True, help='Upload metadata file')
    parser.add_argument('--stats-comparison', required=True, help='Stats comparison file')
    parser.add_argument('--tables', '-t', help='Comma-separated table names')
    parser.add_argument('--filter-mismatched', action='store_true', help='Filter mismatched columns')
    parser.add_argument('--log-level', default='INFO')
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    try:
        start_run()
        table_filter = args.tables.split(',') if args.tables else None
        
        with Timer() as timer:
            aligner = DataAligner(args.config)
            result = aligner.align_all_tables(
                metadata_file=args.metadata_file,
                stats_comparison_file=args.stats_comparison,
                table_filter=table_filter,
                filter_mismatched=args.filter_mismatched
            )
        
        logger.info(f"Alignment completed in {timer.format_duration(timer.elapsed)}")
        logger.info(f"Results: {result}")
        
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        sys.exit(1)
    finally:
        end_run()

if __name__ == '__main__':
    main()