
REPO_DIR=path/to/APO/repo
DATA_DIR="${REPO_DIR}/data/hh-split"

python3 ${REPO_DIR}/tools/apo_data_converter.py \
	--golden_data_path ${DATA_DIR}/rm_data/hh_split_rm.golden.json \
	--sample_data_path ${DATA_DIR}/rm_data/hh_split_rm_alpaca_v0.sample.json \
	--output_dir ${DATA_DIR}/apo_data \
	--apo_data_name "rm_apo_data_v0"
