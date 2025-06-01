while getopts ":b:o:e:s:t:a:" option;do
    case "${option}" in
    e)  e=${OPTARG}
        env=$e
        echo env=$env
    ;;
    b)  b=${OPTARG}
        bind=$b
        echo bind=$bind
    ;;
    s)  s=${OPTARG}
        src=$s
        echo src=$src
    ;;
    o)  o=${OPTARG}
        outdir=$o
        echo outdir=$outdir
    ;;
    t)  t=${OPTARG}
        tmpdir=$t
        echo tmpdir=$tmpdir
    ;;
    a)  a=${OPTARG}
        cluster_address=$a
        echo cluster_address=$cluster_address
    ;;
    *)  echo "Did not supply the correct arguments"
    ;;
    esac
done

############################## UTILITY FUNCTIONS ##############################

# to support running distributed training on nodes
# in non-exclusive mode, we cannot assume that each 
# node has the same number of GPUs/CPUs
get_node_resources() {
    local node="$1"
    gpus=$(srun --nodes=1 --ntasks=1 -w "$node" bash -lc 'echo ${SLURM_GPUS_ON_NODE:-0}')
    cpus=$(srun --nodes=1 --ntasks=1 -w "$node" bash -lc 'echo ${SLURM_CPUS_ON_NODE:-0}')
    echo "${gpus} ${cpus}"
}

############################## ADD WORKER NODES ##############################

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# number of worker nodes
worker_num=$(( SLURM_JOB_NUM_NODES - 1 ))

for ((i = 0; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    read -r node_gpus node_cpus < <(get_node_resources "$node_i")
    mkdir -p $outdir/ray_worker_${i}
    echo "Starting worker node $i for Ray cluster..."
    srun --nodes=1 \
        --ntasks=1 \
        --gpus=$node_gpus \
        --cpus-per-task=$node_cpus \
        -w "$node_i" \
        --job-name="ray_worker_${i}" \
        --output="$outdir/ray_worker_${i}/ray_worker_${i}.out" \
        --error="$outdir/ray_worker_${i}/ray_worker_${i}.err" \
        --export=ALL \
        apptainer exec --userns --nv \
        --bind  $outdir/ray_worker_${i}:$tmpdir \
        --bind $bind \
        --bind ${src}:/workspace/segmentation/src/segmentation \
        --env PYTHONPATH=/workspace/segmentation/src \
        $env $src/clusters/ray_start_worker.sh \
        -t "$tmpdir" \
        -a "$cluster_address" \
        -c "$node_cpus" \
        -g "$node_gpus" &
done

echo "All worker nodes started."

sleep infinity