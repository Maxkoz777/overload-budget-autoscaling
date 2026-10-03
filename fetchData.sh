#!/bin/bash

set -euo pipefail

BASE_URL="https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces"

# DATA_DIR can be overridden so the pipeline can direct downloads straight
# into the per-hour work directory instead of the global data/ folder.
# Example: DATA_DIR=work/hour_000000/raw bash fetchData.sh start_date=0d0 end_date=0d1
DATA_DIR="${DATA_DIR:-data}"

prepare_dir() {
    mkdir -p "${DATA_DIR}/NodeMetrics" "${DATA_DIR}/MSMetrics" "${DATA_DIR}/MSRTMCR"
}

# $1 = start_day, $2 = end_day
# $3 = start_hour, $4 = end_hour
fetch_data() {
    local start_day="$1"
    local end_day="$2"
    local start_hour_arg="$3"
    local end_hour_arg="$4"

    # We intentionally skip CallGraph because it is huge and not needed
    # for the current autoscaling experiment.
    #
    # Index mapping:
    # 0 = MSMetrics    ~= MSResource
    # 1 = NodeMetrics  ~= Node
    # 2 = MSRTMCR      ~= response-time / call-rate metrics
    declare -a file_names=(
        "${DATA_DIR}/MSMetrics/MSMetrics"
        "${DATA_DIR}/NodeMetrics/NodeMetrics"
        "${DATA_DIR}/MSRTMCR/MSRTMCR"
    )

    declare -a remote_paths=(
        "MSMetricsUpdate/MSMetricsUpdate"
        "NodeMetricsUpdate/NodeMetricsUpdate"
        "MCRRTUpdate/MCRRTUpdate"
    )

    # File granularity in minutes:
    # MSMetrics:   30 minutes per file
    # NodeMetrics: 720 minutes = 12 hours per file
    # MSRTMCR:     3 minutes per file
    declare -a ratios=(30 720 3)

    local start_minute=$((start_day * 24 * 60 + start_hour_arg * 60))
    local end_minute=$((end_day * 24 * 60 + end_hour_arg * 60))

    for i in $(seq 0 2); do
        local ratio="${ratios[$i]}"
        local start_idx=$((start_minute / ratio))
        local end_idx=$((end_minute / ratio - 1))

        # If the interval end is not aligned to the file boundary,
        # include the partially overlapping final file.
        if [[ $((end_minute % ratio)) -ne 0 ]]; then
            end_idx=$((end_idx + 1))
        fi

        for idx in $(seq "$start_idx" "$end_idx"); do
            local file_name="${file_names[$i]}_${idx}.tar.gz"
            local remote_path="${remote_paths[$i]}_${idx}.tar.gz"
            local url="${BASE_URL}/${remote_path}"

            # If the file already exists (e.g. pre-populated from the
            # NodeMetrics cache by the Python pipeline), skip the download.
            # This avoids re-downloading the same 12-hour NodeMetrics archive
            # for every hour in the same 12-hour block.
            if [[ -f "$file_name" ]]; then
                local existing_size
                existing_size=$(wc -c < "$file_name")
                echo "Skipping (exists, ${existing_size} bytes): $file_name"
                continue
            fi

            echo "Downloading: $url"
            echo "Saving to:    $file_name"

            wget -c \
                --retry-connrefused \
                --tries=0 \
                --timeout=50 \
                -O "$file_name" \
                "$url"
        done
    done
}

for ARGUMENT in "$@"; do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"
    export "$KEY"="$VALUE"
done

if [[ -z "${start_date:-}" || -z "${end_date:-}" ]]; then
    echo "Usage: bash fetchData.sh start_date=0d0 end_date=0d1"
    exit 1
fi

start_day=$(( $(echo "$start_date" | cut -f1 -dd) + 0 ))
start_hour=$(( $(echo "$start_date" | cut -f2 -dd) + 0 ))
end_day=$(( $(echo "$end_date" | cut -f1 -dd) + 0 ))
end_hour=$(( $(echo "$end_date" | cut -f2 -dd) + 0 ))

prepare_dir
fetch_data "$start_day" "$end_day" "$start_hour" "$end_hour"