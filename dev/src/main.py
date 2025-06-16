#!/usr/bin/env python3
"""
Main entry point for the data migration pipeline.
Provides CLI interface for running individual components or the full pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from loguru import logger
from confection import Config

# Import pipeline components
from .analyzer import MigrationAnalyzer
from .uploader import DataUploader
from .comparator import StatisticsComparator
from .aligner import DataAligner

# Import utilities
from utils.common import start_run, end_run, Timer
from utils.aws import get_aws_manager
from utils.types import MetaConfig


class MigrationPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.aws_manager = get_aws_manager()
        
    def _load_config(self) -> MetaConfig:
        """Load and validate configuration."""
        try:
            config_dict = Config().from_disk(self.config_path)
            return MetaConfig(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load configuration from {self.config_path}: {e}")
            sys.exit(1)
    
    def run_analysis(self, table_filter: Optional[List[str]] = None) -> str:
        """Run migration analysis step."""
        logger.info("Starting migration analysis...")
        
        analyzer = MigrationAnalyzer(self.config_path)
        
        with Timer() as timer:
            result_file = analyzer.run_analysis(table_filter=table_filter)
        
        logger.info(f"Analysis completed in {Timer.format_duration(timer.elapsed)}")
        return result_file
    
    def run_upload(self, analysis_output: str, table_filter: Optional[List[str]] = None,
                   partition_strategy: str = 'date') -> str:
        """Run data upload step."""
        logger.info("Starting data upload...")
        
        uploader = DataUploader(self.config_path)
        
        with Timer() as timer:
            metadata_file = uploader.run_migration(
                analysis_output_path=analysis_output,
                table_filter=table_filter,
                partition_strategy=partition_strategy
            )
        
        logger.info(f"Upload completed in {Timer.format_duration(timer.elapsed)}")
        return metadata_file
    
    def run_comparison(self, metadata_file: str, table_filter: Optional[List[str]] = None) -> str:
        """Run statistics comparison step."""
        logger.info("Starting statistics comparison...")
        
        comparator = StatisticsComparator(self.config_path)
        
        with Timer() as timer:
            comparison_results = comparator.compare_all_tables(
                metadata_file=metadata_file,
                table_filter=table_filter
            )
        
        logger.info(f"Comparison completed in {Timer.format_duration(timer.elapsed)}")
        return comparison_results
    
    def run_alignment(self, metadata_file: str, stats_comparison: str,
                     table_filter: Optional[List[str]] = None) -> str:
        """Run data alignment step."""
        logger.info("Starting data alignment...")
        
        aligner = DataAligner(self.config_path)
        
        with Timer() as timer:
            alignment_results = aligner.align_all_tables(
                metadata_file=metadata_file,
                stats_comparison_file=stats_comparison,
                table_filter=table_filter
            )
        
        logger.info(f"Alignment completed in {Timer.format_duration(timer.elapsed)}")
        return alignment_results
    
    def run_full_pipeline(self, table_filter: Optional[List[str]] = None) -> dict:
        """Run the complete migration pipeline."""
        logger.info("Starting full migration pipeline...")
        
        results = {}
        
        with Timer() as total_timer:
            # Step 1: Analysis
            results['analysis_output'] = self.run_analysis(table_filter)
            
            # Step 2: Upload
            results['metadata_file'] = self.run_upload(
                results['analysis_output'], table_filter
            )
            
            # Step 3: Comparison
            results['comparison_results'] = self.run_comparison(
                results['metadata_file'], table_filter
            )
            
            # Step 4: Alignment
            results['alignment_results'] = self.run_alignment(
                results['metadata_file'], 
                results['comparison_results'], 
                table_filter
            )
        
        logger.info(f"Full pipeline completed in {Timer.format_duration(total_timer.elapsed)}")
        return results


def setup_logging(level: str = "INFO", output_file: str = None):
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    
    # Console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File handler if specified
    if output_file:
        logger.add(
            output_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB"
        )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Data Migration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python src/main.py --config config/prod.cfg --step full
  
  # Run individual steps
  python src/main.py --config config/dev.cfg --step analysis
  python src/main.py --config config/dev.cfg --step upload --analysis-output analysis.pkl
  
  # Filter specific tables
  python src/main.py --config config/dev.cfg --step full --tables table1,table2
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--step', '-s',
        type=str,
        choices=['analysis', 'upload', 'comparison', 'alignment', 'full'],
        default='full',
        help='Pipeline step to run (default: full)'
    )
    
    parser.add_argument(
        '--tables', '-t',
        type=str,
        help='Comma-separated list of tables to process (default: all)'
    )
    
    parser.add_argument(
        '--analysis-output',
        type=str,
        help='Path to analysis output file (required for upload step)'
    )
    
    parser.add_argument(
        '--metadata-file',
        type=str,
        help='Path to upload metadata file (required for comparison/alignment steps)'
    )
    
    parser.add_argument(
        '--stats-comparison',
        type=str,
        help='Path to statistics comparison file (required for alignment step)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log output file (default: console only)'
    )
    
    parser.add_argument(
        '--partition-strategy',
        type=str,
        choices=['date', 'none'],
        default='date',
        help='Data partitioning strategy (default: date)'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Parse table filter
    table_filter = None
    if args.tables:
        table_filter = [t.strip() for t in args.tables.split(',')]
    
    try:
        start_run()
        
        # Initialize pipeline
        pipeline = MigrationPipeline(args.config)
        
        # Run specified step(s)
        if args.step == 'full':
            results = pipeline.run_full_pipeline(table_filter)
            logger.info("Pipeline Results:")
            for step, output in results.items():
                logger.info(f"  {step}: {output}")
        
        elif args.step == 'analysis':
            result = pipeline.run_analysis(table_filter)
            logger.info(f"Analysis output: {result}")
        
        elif args.step == 'upload':
            if not args.analysis_output:
                logger.error("--analysis-output is required for upload step")
                sys.exit(1)
            result = pipeline.run_upload(args.analysis_output, table_filter, args.partition_strategy)
            logger.info(f"Upload metadata: {result}")
        
        elif args.step == 'comparison':
            if not args.metadata_file:
                logger.error("--metadata-file is required for comparison step")
                sys.exit(1)
            result = pipeline.run_comparison(args.metadata_file, table_filter)
            logger.info(f"Comparison results: {result}")
        
        elif args.step == 'alignment':
            if not args.metadata_file or not args.stats_comparison:
                logger.error("--metadata-file and --stats-comparison are required for alignment step")
                sys.exit(1)
            result = pipeline.run_alignment(args.metadata_file, args.stats_comparison, table_filter)
            logger.info(f"Alignment results: {result}")
        
        logger.info("Pipeline execution completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
    finally:
        end_run()


if __name__ == '__main__':
    main()