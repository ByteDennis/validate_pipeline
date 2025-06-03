# Database Migration Pipeline (PCDS → AWS)

A comprehensive data migration and validation pipeline for migrating Oracle tables to AWS Athena with full data integrity verification.


## 📋 Components

### 1. **Migration Analyzer** (`migration_analysis.py`)
Analyzes schema differences between PCDS and AWS tables.

**Key Features:**
- Schema comparison (columns, data types, row counts)
- Column mapping validation
- Temporal data distribution analysis
- Mismatch identification and reporting

**Usage:**
```bash
python migration_analysis.py --config config_meta.cfg
```

**Outputs:** 
- `migration_analysis.pkl` - Detailed analysis results
- `migration_results.csv` - Summary report
- Upload logs and metadata

### 2. **Migration Uploader** (`migration_uploader.py`)
Extracts PCDS data and uploads to AWS S3 as compressed Parquet files.

**Key Features:**
- Chunked data extraction with memory management
- Data type conversion (Oracle → Pandas → Parquet)
- Automatic S3 upload with partitioning
- Upload timestamp tracking for consistency
- Athena table creation

**Usage:**
```bash
python migration_uploader.py \
  --config migration_config.cfg \
  --analysis-output migration_analysis.pkl \
  --partition-strategy date
```

**Outputs:**
- S3 Parquet files with metadata columns
- `upload_metadata_{timestamp}.json` - Upload timing data
- Athena tables ready for querying

### 3. **Statistics Comparator** (`statistics_comparison.py`)
Validates data integrity by comparing column statistics between S3 and Athena.

**Key Features:**
- Column-by-column statistical comparison
- Null counts, unique values, min/max, means
- Upload cutoff alignment for consistency
- Mismatch detection and reporting

**Usage:**
```bash
python statistics_comparison.py
```

**Outputs:**
- `{table}_stats_comparison.json` - Detailed statistics comparison per table

### 4. **Data Aligner** (`data_alignment.py`)
Aligns datasets by removing problematic columns and time periods for final validation.

**Key Features:**
- Filters columns with statistical mismatches
- Excludes time periods with row count discrepancies
- Value-level comparison on cleaned datasets
- Final data integrity verification

**Usage:**
```bash
python data_alignment.py
```

**Outputs:**
- `data_alignment_results.json` - Final alignment and match results

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas awswrangler pyarrow boto3 loguru confection tqdm pyathena
```

### Configuration Files
- `config_meta.cfg` - Migration analyzer configuration
- `migration_config.cfg` - Uploader configuration with AWS settings

### Complete Pipeline Run
```bash
# 1. Analyze schema differences
python migration_analysis.py --config config_meta.cfg

# 2. Upload data to AWS
python migration_uploader.py \
  --config migration_config.cfg \
  --analysis-output migration_analysis.pkl

# 3. Compare statistics
python statistics_comparison.py

# 4. Align and validate
python data_alignment.py
```

## 📊 Key Outputs

| File | Purpose |
|------|---------|
| `migration_analysis.pkl` | Schema analysis results |
| `upload_metadata_{timestamp}.json` | Upload timing for consistency |
| `{table}_stats_comparison.json` | Column statistics comparison |
| `data_alignment_results.json` | Final validation results |

## 🔧 Configuration

### Migration Analyzer Config (`config_meta.cfg`)
```ini
[input]
table = "input_tables.xlsx"
column_maps = "column_mappings.json"
env = ".env"

[output]
folder = "output/"
csv.path = "migration_results.csv"
```

### Uploader Config (`migration_config.cfg`)
```json
{
  "chunk_size": 50000,
  "max_memory_mb": 512,
  "aws": {
    "s3_bucket": "your-migration-bucket",
    "profile": "your-aws-profile"
  }
}
```

## 📈 Data Consistency Features

### Upload Timestamp Tracking
All uploaded data includes metadata columns:
- `_upload_timestamp` - Exact upload time
- `_upload_date` - Upload date
- `_upload_batch_id` - Batch identifier

### Temporal Alignment
Ensures PCDS and AWS data are compared using identical time cutoffs:
```sql
WHERE _upload_timestamp <= TIMESTAMP '2025-06-03 14:30:15'
```

### Validation Hierarchy
1. **Schema Analysis** - Identifies structural differences
2. **Statistics Comparison** - Validates data distributions  
3. **Value-Level Alignment** - Confirms exact data matches

## 🚨 Error Handling

- **Missing Tables**: Logged and skipped
- **Data Type Mismatches**: Automatic conversion with logging
- **Memory Issues**: Chunked processing with configurable limits
- **AWS Failures**: Retry logic with detailed error reporting

## 📝 Logging

All components use structured logging with:
- Progress tracking via `tqdm`
- Detailed error messages and stack traces
- Performance metrics (timing, row counts, file sizes)
- Configurable log levels and output files

## 💡 Best Practices

1. **Test on small datasets** before full migration
2. **Monitor memory usage** during large table processing
3. **Verify AWS credentials** and S3 permissions before starting
4. **Keep upload metadata files** for audit trails
5. **Review alignment results** before declaring migration complete
