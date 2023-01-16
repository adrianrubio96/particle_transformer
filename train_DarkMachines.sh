#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_DarkMachines}
[[ -z $DATADIR ]] && DATADIR='./datasets/EventClass'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=15
samples_per_epoch=$((10 * 32 / $NGPUS))
samples_per_epoch_val=$((10 * 32))
dataopts="--num-workers 2 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp"
    batchopts="--batch-size 20 --start-lr 1e-3"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    batchopts="--batch-size 20 --start-lr 1e-2"
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

# currently only Pythia
#SAMPLE_TYPE=

$CMD \
    --data-train \
    "ttbar:${DATADIR}/train/ttbar_*.root" \
    "Zjets:${DATADIR}/train/z_jets_*.root" \
    "wtop:${DATADIR}/train/wtop_*.root" \
    --data-val "${DATADIR}/val/*.root" \
    --data-test \
    "ttbar:${DATADIR}/test/ttbar_*.root" \
    "Zjets:${DATADIR}/test/z_jets_*.root" \
    "wtop:${DATADIR}/test/wtop_*.root" \
    --data-config data/DarkMachines/EventClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/DarkMachines/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
    --optimizer ranger --log logs/DarkMachines_${FEATURE_TYPE}_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard DarkMachines_${FEATURE_TYPE}_${model}${suffix} \
    "${@:3}"
