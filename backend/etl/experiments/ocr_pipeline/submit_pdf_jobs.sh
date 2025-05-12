#!/bin/bash
# submit_pdf_jobs.sh

# Default values
PDF_DIR="downloaded_papers"
OUTPUT_DIR="extracted_texts"
TEXT_PARSER="marker"
OCR_ENGINE="mistral"
USE_CACHE=true
MAX_JOBS=10  # Maximum concurrent jobs
MEM_PER_JOB="30G"  # Memory per job
TIME_PER_JOB="1:00:00"  # Time per job (1 hour)
PARTITION="test"  # Partition to use

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --pdf_dir)
        PDF_DIR="$2"
        shift
        shift
        ;;
        --output_dir)
        OUTPUT_DIR="$2"
        shift
        shift
        ;;
        --text_parser)
        TEXT_PARSER="$2"
        shift
        shift
        ;;
        --ocr_engine)
        OCR_ENGINE="$2"
        shift
        shift
        ;;
        --no_cache)
        USE_CACHE=false
        shift
        ;;
        --max_jobs)
        MAX_JOBS="$2"
        shift
        shift
        ;;
        --mem)
        MEM_PER_JOB="$2"
        shift
        shift
        ;;
        --time)
        TIME_PER_JOB="$2"
        shift
        shift
        ;;
        --partition)
        PARTITION="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/logs"

# Create a file to store all PDF paths
PDF_LIST="${OUTPUT_DIR}/pdf_list.txt"
find "$PDF_DIR" -name "*.pdf" > "$PDF_LIST"
TOTAL_PDFS=$(wc -l < "$PDF_LIST")

echo "Found $TOTAL_PDFS PDF files to process"
echo "Submitting jobs to process PDFs with:"
echo "- Text parser: $TEXT_PARSER"
echo "- OCR engine: $OCR_ENGINE"
echo "- Output directory: $OUTPUT_DIR"
echo "- Cache enabled: $USE_CACHE"
echo "- Maximum concurrent jobs: $MAX_JOBS"
echo "- Memory per job: $MEM_PER_JOB"
echo "- Time per job: $TIME_PER_JOB"
echo "- Partition: $PARTITION"

# Create a job submission script template
JOB_SCRIPT="${OUTPUT_DIR}/job_template.sh"
cat > "$JOB_SCRIPT" << EOL
#!/bin/bash
#SBATCH -c 4                           # 4 cores (adjust as needed)
#SBATCH -t $TIME_PER_JOB               # Max runtime
#SBATCH -p $PARTITION                  # Partition
#SBATCH --mem=$MEM_PER_JOB             # Memory
#SBATCH -o ${OUTPUT_DIR}/logs/pdf_%j.out  # Standard output
#SBATCH -e ${OUTPUT_DIR}/logs/pdf_%j.err  # Standard error

# Load necessary modules
source /n/hausmann_lab/lab/kdaryanani/deeplearn/gl_deep_search/.venv/bin/activate

# Run the processing script for a single PDF
CACHE_FLAG=""
if [ "$USE_CACHE" = false ]; then
    CACHE_FLAG="--no_cache"
fi

python process_single_pdf.py \$1 --output_dir "$OUTPUT_DIR" --text_parser "$TEXT_PARSER" --ocr_engine "$OCR_ENGINE" \$CACHE_FLAG

# Deactivate virtual environment
deactivate
EOL

chmod +x "$JOB_SCRIPT"

# Create a results collection script
COLLECT_SCRIPT="${OUTPUT_DIR}/collect_results.sh"
cat > "$COLLECT_SCRIPT" << EOL
#!/bin/bash

# Find all result.json files and combine them
echo "[" > "${OUTPUT_DIR}/processing_summary.json"
find "$OUTPUT_DIR" -name "*_result.json" -type f | while read -r file; do
    cat "\$file" | sed 's/$/,/'
done | sed '\$ s/,$//' >> "${OUTPUT_DIR}/processing_summary.json"
echo "]" >> "${OUTPUT_DIR}/processing_summary.json"

# Generate statistics
echo "Processing complete!"
echo "Total PDFs processed: \$(find "$OUTPUT_DIR" -name "*_result.json" -type f | wc -l)"
echo "Results saved to ${OUTPUT_DIR}/processing_summary.json"
EOL

chmod +x "$COLLECT_SCRIPT"

# Submit jobs with a limit on concurrent jobs
SUBMITTED=0
COMPLETED=0

echo "Submitting jobs..."
while read -r pdf_path; do
    # Check if we've reached the maximum number of concurrent jobs
    while [ $(squeue -h -u $USER | wc -l) -ge $MAX_JOBS ]; do
        echo "Job limit reached ($MAX_JOBS). Waiting for jobs to complete..."
        sleep 30
        
        # Count completed jobs for progress update
        NEW_COMPLETED=$(find "$OUTPUT_DIR" -name "*_result.json" -type f | wc -l)
        if [ $NEW_COMPLETED -gt $COMPLETED ]; then
            COMPLETED=$NEW_COMPLETED
            echo "Progress: $COMPLETED/$TOTAL_PDFS PDFs processed ($(( COMPLETED * 100 / TOTAL_PDFS ))%)"
        fi
    done
    
    # Submit job
    sbatch "$JOB_SCRIPT" "$pdf_path"
    SUBMITTED=$((SUBMITTED + 1))
    echo "Submitted job $SUBMITTED/$TOTAL_PDFS: $pdf_path"
    
    # Small delay to prevent overwhelming the scheduler
    sleep 0.5
done < "$PDF_LIST"

echo "All $SUBMITTED PDF processing jobs submitted."
echo "When all jobs complete, run: ${COLLECT_SCRIPT} to collect results"