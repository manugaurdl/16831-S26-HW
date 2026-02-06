rameter sweep: varying num_agent_train_steps_per_iter
# Testing how training steps affect BC performance on Ant-v2

# Array of training step values to test
TRAINING_STEPS=(500 1000 2000 5000 10000 15000 20000)

# Common parameters
EXPERT_POLICY="rob831/policies/experts/Ant.pkl"
ENV_NAME="Ant-v2"
EXPERT_DATA="rob831/expert_data/expert_data_Ant-v2.pkl"
EVAL_BATCH_SIZE=5000
TRAIN_BATCH_SIZE=1000

echo "Starting hyperparameter sweep for num_agent_train_steps_per_iter"
echo "Testing values: ${TRAINING_STEPS[@]}"
echo "=================================================="

# Loop through each training step value
for steps in "${TRAINING_STEPS[@]}"
do
    echo ""
    echo "Running experiment with $steps training steps..."
    echo "=================================================="
    
    python -m rob831.scripts.run_hw1 \
        --expert_policy_file $EXPERT_POLICY \
        --env_name $ENV_NAME \
        --exp_name bc_ant_steps_${steps} \
        --n_iter 1 \
        --expert_data $EXPERT_DATA \
        --video_log_freq -1 \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --num_agent_train_steps_per_iter $steps \
        --train_batch_size $TRAIN_BATCH_SIZE
    
    echo ""
    echo "Completed experiment with $steps training steps"
    echo "=================================================="
    sleep 2  # Brief pause between experiments
done

echo ""
echo "All experiments completed!"
echo "Results saved in: hw1/data/"
