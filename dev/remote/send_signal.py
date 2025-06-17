import json
import time
from datetime import datetime
from utils.aws import S3Utils

class RemoteAnalysisOrchestrator:
    def __init__(self, signal_bucket: str, result_bucket: str):
        self.s3_utils = S3Utils()
        self.signal_bucket = signal_bucket
        self.result_bucket = result_bucket
    
    def trigger_remote_analysis(self, analysis_metadata: dict) -> str:
        """Send signal to trigger remote analysis and wait for results."""
        
        # Create unique job ID
        job_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare signal payload
        signal_data = {
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'metadata_file': analysis_metadata['metadata_file'],
            'tables_to_process': analysis_metadata.get('tables', []),
            'config': analysis_metadata.get('config', {}),
            'status': 'pending'
        }
        
        # Send signal to S3
        signal_path = f"s3://{self.signal_bucket}/signals/{job_id}.json"
        self.s3_utils.save_json(signal_data, signal_path)
        
        logger.info(f"Signal sent for job {job_id}")
        
        # Wait for results
        result_path = f"s3://{self.result_bucket}/results/{job_id}_complete.json"
        return self._wait_for_results(result_path, job_id)
    
    def _wait_for_results(self, result_path: str, job_id: str, 
                         timeout_minutes: int = 60) -> str:
        """Wait for remote analysis to complete."""
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        logger.info(f"Waiting for results at {result_path}")
        
        while time.time() - start_time < timeout_seconds:
            if self.s3_utils.exists(result_path):
                # Load and return results
                results = self.s3_utils.load_json(result_path)
                
                if results.get('status') == 'completed':
                    logger.info(f"Remote analysis completed for job {job_id}")
                    return results['output_path']
                elif results.get('status') == 'failed':
                    raise Exception(f"Remote analysis failed: {results.get('error')}")
            
            # Check every 30 seconds
            time.sleep(30)
            logger.info(f"Still waiting for job {job_id}...")
        
        raise TimeoutError(f"Remote analysis timed out after {timeout_minutes} minutes")


class MigrationPipeline:
    def __init__(self, config_path: str, use_remote_analysis: bool = True):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.use_remote_analysis = use_remote_analysis
        
        if use_remote_analysis:
            self.remote_orchestrator = RemoteAnalysisOrchestrator(
                signal_bucket="migration-signals",
                result_bucket="migration-results"
            )
    
    def run_full_pipeline(self, table_filter: Optional[List[str]] = None) -> dict:
        """Run the complete migration pipeline with remote analysis."""
        
        results = {}
        
        # Step 1 & 2: Local analysis and upload
        results['analysis_output'] = self.run_analysis(table_filter)
        results['metadata_file'] = self.run_upload(results['analysis_output'], table_filter)
        
        if self.use_remote_analysis:
            # Step 3 & 4: Remote comparison and alignment
            logger.info("Triggering remote analysis...")
            
            analysis_metadata = {
                'metadata_file': results['metadata_file'],
                'tables': table_filter,
                'config': {}  # Add any needed config
            }
            
            # This will hang until remote analysis completes
            remote_results_path = self.remote_orchestrator.trigger_remote_analysis(analysis_metadata)
            results['remote_analysis_results'] = remote_results_path
            
            # Step 5: Local detailed analysis with xlwings
            results['detailed_analysis'] = self.run_detailed_analysis(remote_results_path)
        
        return results
    
    def run_detailed_analysis(self, remote_results_path: str) -> str:
        """Run detailed analysis with xlwings using remote results."""
        
        # Download remote results
        remote_summary = self.remote_orchestrator.s3_utils.load_json(remote_results_path)
        
        # Download comparison and alignment files
        comparison_file = "/tmp/comparison_results.json"
        alignment_file = "/tmp/alignment_results.json"
        
        self.remote_orchestrator.s3_utils.download_file(
            remote_summary['comparison_results'], comparison_file
        )
        self.remote_orchestrator.s3_utils.download_file(
            remote_summary['alignment_results'], alignment_file
        )
        
        # Run xlwings analysis
        return self._run_xlwings_analysis(comparison_file, alignment_file)
    
    def _run_xlwings_analysis(self, comparison_file: str, alignment_file: str) -> str:
        """Run detailed Excel analysis using xlwings."""
        import xlwings as xw
        
        # Your xlwings logic here
        logger.info("Running detailed Excel analysis...")
        
        # Create detailed Excel reports
        # ... xlwings code ...
        
        return "detailed_analysis_complete.xlsx"