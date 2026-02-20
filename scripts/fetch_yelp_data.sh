#!/usr/bin/env bash
set -euo pipefail

URL="https://business.yelp.com/external-assets/files/Yelp-JSON.zip"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
ZIP_PATH="${RAW_DIR}/Yelp-JSON.zip"

mkdir -p "${RAW_DIR}"

if [[ -f "${ZIP_PATH}" ]]; then
  echo "Found existing archive at ${ZIP_PATH}, skipping download."
else
  echo "Downloading Yelp dataset zip..."
  curl -L \
    -H "User-Agent: Mozilla/5.0" \
    -H "Referer: https://business.yelp.com/data/resources/open-dataset/" \
    "${URL}" \
    -o "${ZIP_PATH}"
fi

echo "Extracting zip..."
unzip -o "${ZIP_PATH}" -d "${RAW_DIR}" >/dev/null

extract_tar_if_present() {
  local tar_path="$1"
  if [[ -f "${tar_path}" ]]; then
    echo "Extracting $(basename "${tar_path}")..."
    tar -xf "${tar_path}" -C "${RAW_DIR}"
  fi
}

rename_if_present() {
  local source="$1"
  local target="$2"
  if [[ -f "${source}" ]]; then
    mv -f "${source}" "${target}"
    echo "Renamed $(basename "${source}") -> $(basename "${target}")"
  fi
}

# Yelp currently ships a tar inside the zip; support both old and new layouts.
extract_tar_if_present "${RAW_DIR}/yelp_dataset.tar"
extract_tar_if_present "${RAW_DIR}/Yelp JSON/yelp_dataset.tar"
extract_tar_if_present "${RAW_DIR}/Yelp-JSON/yelp_dataset.tar"

# Handle common extracted layouts (root, Yelp-JSON/, or Yelp JSON/).
rename_if_present "${RAW_DIR}/yelp_academic_dataset_business.json" "${RAW_DIR}/business.json"
rename_if_present "${RAW_DIR}/yelp_academic_dataset_review.json" "${RAW_DIR}/review.json"
rename_if_present "${RAW_DIR}/yelp_academic_dataset_user.json" "${RAW_DIR}/user.json"
rename_if_present "${RAW_DIR}/Yelp-JSON/yelp_academic_dataset_business.json" "${RAW_DIR}/business.json"
rename_if_present "${RAW_DIR}/Yelp-JSON/yelp_academic_dataset_review.json" "${RAW_DIR}/review.json"
rename_if_present "${RAW_DIR}/Yelp-JSON/yelp_academic_dataset_user.json" "${RAW_DIR}/user.json"
rename_if_present "${RAW_DIR}/Yelp JSON/yelp_academic_dataset_business.json" "${RAW_DIR}/business.json"
rename_if_present "${RAW_DIR}/Yelp JSON/yelp_academic_dataset_review.json" "${RAW_DIR}/review.json"
rename_if_present "${RAW_DIR}/Yelp JSON/yelp_academic_dataset_user.json" "${RAW_DIR}/user.json"

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
