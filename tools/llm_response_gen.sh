#! /bin/bash

REPO_DIR=<path_to_APO_repo>
export PYTHONPATH=${REPO_DIR}


MODEL_DIR="chavinlo/alpaca-native"
MODEL_NAME="alpaca"

#TASK_TYPE="testing"
TASK_TYPE="sampling"

DATA_DIR=${REPO_DIR}/data/hh-split
if [[ "${TASK_TYPE}" == "testing" ]]; then    
    DATA_PATH=${DATA_DIR}/eval_data/hh_cleaned_origin.test.json
    DATA_NAME="hh_test"   
    DATA_TYPE="comparison_pair"   
else    
    DATA_DIR=${REPO_DIR}/data/hh-split
    DATA_PATH=${DATA_DIR}/llm_data/hh_split_llm.train.json
    DATA_NAME="hh_llm_train"   
    DATA_TYPE="comparison_pair"    
fi

OUTPUT_DIR=${DATA_DIR}/sample_data
mkdir -p $OUTPUT_DIR


EVAL_MICRO_BATCH_SIZE=1
MAX_INPUT_LENGTH=512

torchrun --nproc_per_node 8 --master_port 6000 ${REPO_DIR}/tools/inference_llm.py \
	  --model_name_or_path $MODEL_DIR \
	  --model_prefix ${MODEL_NAME} \
          --data_path $DATA_PATH \
          --output_dir $OUTPUT_DIR \
          --per_device_eval_batch_size $EVAL_MICRO_BATCH_SIZE \
	  --task_type ${TASK_TYPE} \
	  --data_suffix ${DATA_NAME} \
	  --max_length ${MAX_INPUT_LENGTH} \
	  --data_type ${DATA_TYPE}
