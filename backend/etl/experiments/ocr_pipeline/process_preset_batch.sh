#!/bin/bash
# process_preset_batch.sh - Process all PDFs with one preset/model combination

# Default values
PDF_DIR="data/raw_pdfs"
BASE_OUTPUT_DIR="data/processed"
MODULE="marker"
PRESET="baseline"
MODEL="none"
USE_CACHE=true
MAX_JOBS=10
MEM_PER_JOB="20G"
TIME_PER_JOB="1:00:00"
PARTITION="test"
SUBMIT_DELAY=10
PILOT_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pdf_dir) PDF_DIR="$2"; shift 2 ;;
        --output_dir) BASE_OUTPUT_DIR="$2"; shift 2 ;;
        --module) MODULE="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --no_cache) USE_CACHE=false; shift ;;
        --max_jobs) MAX_JOBS="$2"; shift 2 ;;
        --mem) MEM_PER_JOB="$2"; shift 2 ;;
        --time) TIME_PER_JOB="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --delay) SUBMIT_DELAY="$2"; shift 2 ;;
        --pilot) PILOT_MODE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Function to get expected output path for a PDF
get_expected_output_path() {
    local pdf_path="$1"
    local pdf_dir=$(dirname "$pdf_path")
    local pdf_name=$(basename "$pdf_path" .pdf)
    local parent_folder=$(basename "$pdf_dir")
    
    # Determine file extension based on preset
    local ext=".md"
    if [ "$PRESET" = "json" ]; then
        ext=".json"
    fi
    
    # Match the organize_by_folder logic from Python script
    echo "${OUTPUT_DIR}/${parent_folder}/${pdf_name}_${MODULE}_${PRESET}${ext}"
}

# Create organized output directory based on preset and model
if [ "$MODEL" = "none" ]; then
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODULE}_${PRESET}"
else
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODULE}_${PRESET}_${MODEL}"
fi

# Setup
mkdir -p "$OUTPUT_DIR/logs"
PDF_LIST="${OUTPUT_DIR}/pdf_list.txt"

# Create PDF list - just first one if pilot mode
if [ "$PILOT_MODE" = true ]; then
    find "$PDF_DIR" -name "*.pdf" | head -1 > "$PDF_LIST"
    echo "🧪 PILOT MODE: Processing only the first PDF"
else
    find "$PDF_DIR" -name "*.pdf" > "$PDF_LIST"
fi

TOTAL_PDFS=$(wc -l < "$PDF_LIST")

if [ $TOTAL_PDFS -eq 0 ]; then
    echo "❌ No PDFs found in $PDF_DIR"
    exit 1
fi

echo "🚀 Processing $TOTAL_PDFS PDF$([ $TOTAL_PDFS -gt 1 ] && echo "s") with:"
echo "   Module: $MODULE"
echo "   Preset: $PRESET"
echo "   Model: $MODEL"
echo "   Output: $OUTPUT_DIR"
echo "   Max concurrent jobs: $MAX_JOBS"
[ "$PILOT_MODE" = true ] && echo "   🧪 PILOT MODE ENABLED"
echo ""

# Create job script
JOB_SCRIPT="${OUTPUT_DIR}/job_template.sh"
cat > "$JOB_SCRIPT" << EOL
#!/bin/bash
#SBATCH -c 4
#SBATCH -t $TIME_PER_JOB
#SBATCH -p $PARTITION
#SBATCH --mem=$MEM_PER_JOB
#SBATCH -o ${OUTPUT_DIR}/logs/pdf_%j.out
#SBATCH -e ${OUTPUT_DIR}/logs/pdf_%j.err
#SBATCH -J ${MODULE}_${PRESET}

cd /n/hausmann_lab/lab/kdaryanani/deeplearn/gl_deep_search/backend/etl/experiments/ocr_pipeline

echo "Processing: \$(basename "\$1")"
echo "Preset: $PRESET, Model: $MODEL"
echo "Started: \$(date)"

CACHE_FLAG=""
if [ "$USE_CACHE" = false ]; then
    CACHE_FLAG="--no_cache"
fi

uv run python pdf_modules.py "\$1" \\
    --module "$MODULE" \\
    --preset "$PRESET" \\
    --model "$MODEL" \\
    --output_dir "$OUTPUT_DIR" \\
    --no_organize \\
    \$CACHE_FLAG

echo "Completed: \$(date)"
EOL

chmod +x "$JOB_SCRIPT"

# Submit jobs with queue monitoring
SUBMITTED=0
COMPLETED=0

echo "Submitting jobs..."
while read -r pdf_path; do
    # Check if output already exists
    expected_output=$(get_expected_output_path "$pdf_path")
    if [ -f "$expected_output" ]; then
        echo "⏭️  Skipping $(basename "$pdf_path") - output already exists"
        continue
    fi
    
    # Wait for free job slots
    while [ $(squeue -h -u $USER | wc -l) -ge $MAX_JOBS ]; do
        echo "Queue full ($(squeue -h -u $USER | wc -l)/$MAX_JOBS jobs). Waiting..."
        sleep 30
        
        # Update progress
        NEW_COMPLETED=$(find "$OUTPUT_DIR" -name "*_result.json" -type f | wc -l)
        if [ $NEW_COMPLETED -gt $COMPLETED ]; then
            COMPLETED=$NEW_COMPLETED
            echo "Progress: $COMPLETED/$TOTAL_PDFS ($(( COMPLETED * 100 / TOTAL_PDFS ))%)"
        fi
    done
    
    # Submit job
    sbatch "$JOB_SCRIPT" "$pdf_path"
    SUBMITTED=$((SUBMITTED + 1))
    echo "Submitted $SUBMITTED: $(basename "$pdf_path")"
    sleep $SUBMIT_DELAY
done < "$PDF_LIST"

echo ""
echo "🎯 All jobs submitted!"
[ "$PILOT_MODE" = true ] && echo "🧪 Pilot run complete - check results before full batch"
echo "Monitor: squeue -u $USER"
echo "Outputs will be in: $OUTPUT_DIR"
echo ""