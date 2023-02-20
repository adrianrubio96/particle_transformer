#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_DarkMachines}
[[ -z $DATADIR ]] && DATADIR='./datasets/EventClass'

# set a comment via `COMMENT`
suffix=${COMMENT}

# Get batch size and lr
batchsize=$( echo "${@:3}" | cut -d ' ' -f2 )
lr=$( echo "${@:3}" | cut -d ' ' -f4 )
optimizer=adamW

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=1
samples_per_epoch=$((10 * 32 / $NGPUS))
samples_per_epoch_val=$((10 * 32))
dataopts="--num-workers 2 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp"
    batchopts="--batch-size ${batchsize} --start-lr ${lr}" # --lr-finder 1e-3,1e-9,200  --lr-finder-plot
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    batchopts="--batch-size ${batchsize} --start-lr ${lr}" # --lr-finder 1e-3,1e-9,200"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    batchopts="--batch-size 32 --start-lr 2e-2"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "full"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="full"

if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

# Define RUNCODE
# From string modelopts, get the architecture name
architecture=$(echo $modelopts | cut -d' ' -f1 | cut -d'/' -f2 | cut -d'.' -f1)

# Get string with time and date
RUNCODE=$(date +"%Y%m%d-%H%M%S")
RUNCODE+="_${architecture}_${optimiser}_lr${lr}_batch${batchsize}"
echo "RUNCODE=${RUNCODE}"

# Run training

#$CMD \
#    --data-train \
#    "signal:${DATADIR}/train/signal_*.root" \
#    "background:${DATADIR}/train/background_*.root" \
#    --data-val "${DATADIR}/val/*.root" \
#    --data-test \
#    "signal:${DATADIR}/test/signal_*.root" \
#    "background:${DATADIR}/test/background_*.root" \
#    --data-config data/DarkMachines/EventClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
#    --model-prefix training/DarkMachines/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
#    $dataopts $batchopts \
#    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
#    --optimizer ranger --log logs/DarkMachines_${FEATURE_TYPE}_${model}_{auto}${suffix}.log --predict-output pred.root \
#    --tensorboard DarkMachines_${FEATURE_TYPE}_${model}${suffix} \
#    "${@:3}"


# Training with no prediction on test set

$CMD \
    --data-train \
    "background:${DATADIR}/train/background_*.root" \
    "signal:${DATADIR}/train/signal_*.root" \
    --data-val "${DATADIR}/val/*.root" \
    --data-config data/DarkMachines/EventClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/DarkMachines/${FEATURE_TYPE}/${model}/${RUNCODE}${suffix}/net \
    $dataopts $batchopts \
    --num-epochs $epochs --gpus 0 \
    --optimizer $optimizer --log logs/DarkMachines_${FEATURE_TYPE}_${model}_${RUNCODE}${suffix}.log --predict-output pred.root \
    --tensorboard DarkMachines_${FEATURE_TYPE}_${model}${suffix} 

# Create script to run prediction on test set

echo "gpurun $CMD \
    --predict --data-test \
    "background:${DATADIR}/test/background_*.root" \
    "signal:${DATADIR}/test/signal_*.root" \
    --data-config data/DarkMachines/EventClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/DarkMachines/${FEATURE_TYPE}/${model}/${RUNCODE}${suffix}/net \
    --predict-output pred.root \
    $dataopts $batchopts" > pred_scripts/pred_${RUNCODE}${suffix}.sh

# Create script to run plots
PLOTDIR=${PLOTDIR}

## Scores
echo "#!/bin/bash" > ${PLOTDIR}/scores_scripts/scores_${RUNCODE}${suffix}.sh
echo "cd ${PLOTDIR}" >> ${PLOTDIR}/scores_scripts/scores_${RUNCODE}${suffix}.sh
echo "source setup.sh" >> ${PLOTDIR}/scores_scripts/scores_${RUNCODE}${suffix}.sh
echo "python scores.py -r ${RUNCODE}${suffix} " >> ${PLOTDIR}/scores_scripts/scores_${RUNCODE}${suffix}.sh

## ROC curve and AUC
echo "#!/bin/bash" > ${PLOTDIR}/roc_scripts/roc_${RUNCODE}${suffix}.sh
echo "source ~/.env_py3p7_torch/bin/activate" >> ${PLOTDIR}/roc_scripts/roc_${RUNCODE}${suffix}.sh
echo "cd ${PLOTDIR}" >> ${PLOTDIR}/roc_scripts/roc_${RUNCODE}${suffix}.sh
echo "python roc.py -r ${RUNCODE}${suffix}" >> ${PLOTDIR}/roc_scripts/roc_${RUNCODE}${suffix}.sh
echo "deactivate" >> ${PLOTDIR}/roc_scripts/roc_${RUNCODE}${suffix}.sh

## Loss curve
### Find tensorboard log directory
cd runs/
TBLOGDIR=$(ls -td -- */ | head -n 1 | grep DarkMachines_${FEATURE_TYPE}_${model}${suffix})
echo "#!/bin/bash" > ${PLOTDIR}/Loss_scripts/Loss_${RUNCODE}${suffix}.sh
echo "source ~/.env_py3p7_torch/bin/activate" >> ${PLOTDIR}/Loss_scripts/Loss_${RUNCODE}${suffix}.sh
echo "cd ${PLOTDIR}" >> ${PLOTDIR}/Loss_scripts/Loss_${RUNCODE}${suffix}.sh
echo "python Loss.py -r ${RUNCODE}${suffix} -t ${TBLOGDIR} " >> ${PLOTDIR}/Loss_scripts/Loss_${RUNCODE}${suffix}.sh
echo "deactivate" >> ${PLOTDIR}/Loss_scripts/Loss_${RUNCODE}${suffix}.sh
cd ..

#echo "training/DarkMachines/${FEATURE_TYPE}/${model}/20230118-181428_example_ParticleTransformer_ranger_lr0.001_batch20${suffix}/net_best_epoch_state.pt"
#echo "${@:3}"