#!/usr/bin/env bash
set -euo pipefail

URL="https://business.yelp.com/external-assets/files/Yelp-JSON.zip"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
ZIP_PATH="${RAW_DIR}/Yelp-JSON.zip"

mkdir -p "${RAW_DIR}"

echo "Downloading Yelp dataset zip..."
curl -L \
  -H "User-Agent: Mozilla/5.0" \
  -H "Referer: https://business.yelp.com/data/resources/open-dataset/" \
  "${URL}" \
  -o "${ZIP_PATH}"

echo "Extracting zip..."
unzip -o "${ZIP_PATH}" -d "${RAW_DIR}" >/dev/null

rename_if_present() {
  local source="$1"
  local target="$2"
  if [[ -f "${source}" ]]; then
    mv -f "${source}" "${target}"
    echo "Renamed $(basename "${source}") -> $(basename "${target}")"
  fi
}

# Handle common zip layouts (files at root or nested under Yelp-JSON/).
rename_if_present "${RAW_DIR}/yelp_academic_dataset_business.json" "${RAW_DIR}/business.json"
rename_if_present "${RAW_DIR}/yelp_academic_dataset_review.json" "${RAW_DIR}/review.json"
rename_if_present "${RAW_DIR}/yelp_academic_dataset_user.json" "${RAW_DIR}/user.json"
rename_if_present "${RAW_DIR}/Yelp-JSON/yelp_academic_dataset_business.json" "${RAW_DIR}/business.json"
rename_if_present "${RAW_DIR}/Yelp-JSON/yelp_academic_dataset_review.json" "${RAW_DIR}/review.json"
rename_if_present "${RAW_DIR}/Yelp-JSON/yelp_academic_dataset_user.json" "${RAW_DIR}/user.json"

for required_file in business.json review.json user.json; do
  if [[ ! -f "${RAW_DIR}/${required_file}" ]]; then
    echo "Missing expected file after extraction: ${RAW_DIR}/${required_file}" >&2
    exit 1
  fi
done

echo "Done. Files available in ${RAW_DIR}:"
echo " - business.json"
echo " - review.json"
echo " - user.json"
