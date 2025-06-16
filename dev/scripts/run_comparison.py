#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.comparator import StatisticsComparator
from utils.common import setup_logging, Timer, start_run, end_run
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description='Run Statistics Comparison')
    parser.add_argument('--config', '-c', required=True, help='Configuration file')
    parser.add_argument('--metadata-file', required=True, help='Upload metadata file')
    parser.add_argument('--tables', '-t', help='Comma-separated table names')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--log-level', default='INFO')
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    try:
        start_run()
        table_filter = args.tables.split(',') if args.tables else None
        
        with Timer() as timer:
            comparator = StatisticsComparator(args.config)
            result = comparator.compare_all_tables(
                metadata_file=args.metadata_file,
                table_filter=table_filter,
                output_dir=args.output
            )
        
        logger.info(f"Comparison completed in {timer.format_duration(timer.elapsed)}")
        logger.info(f"Results: {result}")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        sys.exit(1)
    finally:
        end_run()

if __name__ == '__main__':
    main()