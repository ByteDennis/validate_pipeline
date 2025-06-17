import time
import json
from pathlib import Path
from utils.aws import S3Utils
from src.comparator import StatisticsComparator
from src.aligner import DataAligner

class RemoteAnalysisWorker:
    def __init__(self, signal_bucket: str, result_bucket: str):
        self.s3_utils = S3Utils()
        self.signal_bucket = signal_bucket
        self.result_bucket = result_bucket
        self.processed_jobs = set()
    
    def start_watching(self, check_interval: int = 300):  # 5 minutes
        """Start watching for signals in S3."""
        
        logger.info("Starting remote analysis worker...")
        
        while True:
            try:
                self._check_for_signals()
                time.sleep(check_interval)
            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(check_interval)
    
    def _check_for_signals(self):
        """Check S3 for new analysis signals."""
        
        signal_prefix = f"s3://{self.signal_bucket}/signals/"
        signal_files = self.s3_utils.list_objects(signal_prefix)
        
        for signal_file in signal_files:
            if signal_file.endswith('.json'):
                job_id = Path(signal_file).stem
                
                if job_id not in self.processed_jobs:
                    logger.info(f"Found new job: {job_id}")
                    self._process_job(signal_file, job_id)
    
    def _process_job(self, signal_file: str, job_id: str):
        """Process a single analysis job."""
        
        try:
            # Load signal data
            signal_data = self.s3_utils.load_json(signal_file)
            
            # Mark as processing
            self.processed_jobs.add(job_id)
            
            # Update status to processing
            signal_data['status'] = 'processing'
            signal_data['started_at'] = datetime.now().isoformat()
            self.s3_utils.save_json(signal_data, signal_file)
            
            # Run the analysis
            result_path = self._run_analysis(signal_data)
            
            # Save completion signal
            completion_data = {
                'job_id': job_id,
                'status': 'completed',
                'completed_at': datetime.now().isoformat(),
                'output_path': result_path,
                'input_signal': signal_data
            }
            
            completion_path = f"s3://{self.result_bucket}/results/{job_id}_complete.json"
            self.s3_utils.save_json(completion_data, completion_path)
            
            # Clean up signal file
            self.s3_utils.delete(signal_file)
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            # Save failure signal
            failure_data = {
                'job_id': job_id,
                'status': 'failed',
                'failed_at': datetime.now().isoformat(),
                'error': str(e)
            }
            
            failure_path = f"s3://{self.result_bucket}/results/{job_id}_complete.json"
            self.s3_utils.save_json(failure_data, failure_path)
    
    def _run_analysis(self, signal_data: dict) -> str:
        """Run the actual statistics comparison and alignment."""
        
        job_id = signal_data['job_id']
        metadata_file = signal_data['metadata_file']
        tables = signal_data.get('tables_to_process')
        
        # Download metadata file from S3 to local temp
        temp_metadata = f"/tmp/{job_id}_metadata.json"
        self.s3_utils.download_file(metadata_file, temp_metadata)
        
        # Run statistics comparison
        logger.info(f"Running statistics comparison for {job_id}")
        comparator = StatisticsComparator('config/aws_server.cfg')  # Server config
        comparison_results = comparator.compare_all_tables(
            metadata_file=temp_metadata,
            table_filter=tables
        )
        
        # Run data alignment
        logger.info(f"Running data alignment for {job_id}")
        aligner = DataAligner('config/aws_server.cfg')
        alignment_results = aligner.align_all_tables(
            metadata_file=temp_metadata,
            stats_comparison_file=comparison_results,
            table_filter=tables
        )
        
        # Upload results to S3
        results_folder = f"s3://{self.result_bucket}/analysis_results/{job_id}/"
        
        self.s3_utils.upload_file(comparison_results, f"{results_folder}comparison_results.json")
        self.s3_utils.upload_file(alignment_results, f"{results_folder}alignment_results.json")
        
        # Create summary
        summary = {
            'job_id': job_id,
            'comparison_results': f"{results_folder}comparison_results.json",
            'alignment_results': f"{results_folder}alignment_results.json",
            'completed_at': datetime.now().isoformat()
        }
        
        summary_path = f"{results_folder}summary.json"
        self.s3_utils.save_json(summary, summary_path)
        
        return summary_path

# AWS server startup script
if __name__ == "__main__":
    worker = RemoteAnalysisWorker(
        signal_bucket="migration-signals", 
        result_bucket="migration-results"
    )
    worker.start_watching(check_interval=300)  # Check every 5 minutes