#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.uploader import DataUploader
from utils.common import setup_logging, Timer, start_run, end_run
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description='Run Data Upload')
    parser.add_argument('--config', '-c', required=True, help='Configuration file')
    parser.add_argument('--analysis-output', required=True, help='Analysis output file')
    parser.add_argument('--tables', '-t', help='Comma-separated table names')
    parser.add_argument('--partition-strategy', default='date', choices=['date', 'none'])
    parser.add_argument('--log-level', default='INFO')
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    try:
        start_run()
        table_filter = args.tables.split(',') if args.tables else None
        
        with Timer() as timer:
            uploader = DataUploader(args.config)
            result = uploader.run_migration(
                analysis_output_path=args.analysis_output,
                table_filter=table_filter,
                partition_strategy=args.partition_strategy
            )
        
        logger.info(f"Upload completed in {timer.format_duration(timer.elapsed)}")
        logger.info(f"Results: {result}")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        sys.exit(1)
    finally:
        end_run()

if __name__ == '__main__':
    main()