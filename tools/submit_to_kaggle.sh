#!/usr/bin/env bash
set -euo pipefail

competition="${KAGGLE_COMPETITION:-orbit-wars}"
archive="${SUBMISSION_FILE:-submission.tar.gz}"
build_only=0

if [[ "${1:-}" == "--build-only" ]]; then
  build_only=1
  shift
fi

message="${*:-main.py native agent}"

required_paths=(
  "main.py"
  "agent_common.py"
  "src/orbit_native/bindings.cpp"
  "src/orbit_native/orbit_core.cpp"
  "src/orbit_native/orbit_core.hpp"
  "third_party/pybind11_include/pybind11/pybind11.h"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required submission input: $path" >&2
    exit 1
  fi
done

python_agents=()
while IFS= read -r path; do
  python_agents+=("$path")
done < <(find . -maxdepth 1 -type f \( -name 'agent_*.py' -o -name 'agent_common.py' \) -print | sed 's#^\./##' | sort)

echo "Building $archive for $competition"
tar -czf "$archive" \
  main.py \
  "${python_agents[@]}" \
  src/orbit_native \
  third_party/pybind11_include

echo "Archive contents:"
tar -tzf "$archive" | sed -n '1,40p'
echo

if [[ "$build_only" == "1" ]]; then
  echo "Built $archive; skipping Kaggle submit because --build-only was set."
  exit 0
fi

echo "Submitting $archive"
kaggle competitions submit "$competition" -f "$archive" -m "$message"
