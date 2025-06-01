export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export NCCL_P2P_LEVEL=NVL

while getopts ":s:b:o:e:w:W:t:" option;do
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
    w)  w=${OPTARG}
        worker_opts=$w
        echo worker_opts=$worker_opts
    ;;
    W)  W=${OPTARG}
        worker_wrap=$W
        echo worker_wrap=$worker_wrap
    ;;
    t)  t=${OPTARG}
        task=$t
        echo task=$task
    ;;
    *)  echo "Did not supply the correct arguments"
    ;;
    esac
done

# create symlink between output directory on host
# and temporary directory in the container
tmpdir=/tmp/symlink_$(uuidgen | cut -d "-" -f5)
echo "Create symlink: $outdir -> $tmpdir"

############################## UTILITY FUNCTIONS ##############################

get_node_resources() {
    local node="$1"
    gpus=$(srun --nodes=1 --ntasks=1 -w "$node" bash -lc 'echo ${SLURM_GPUS_ON_NODE:-0}')
    cpus=$(srun --nodes=1 --ntasks=1 -w "$node" bash -lc 'echo ${SLURM_CPUS_ON_NODE:-0}')
    echo "${gpus} ${cpus}"
}

#bias to selection of higher range ports
function getfreeport()
{
    CHECK="do while"
    while [[ ! -z $CHECK ]]; do
        port=$(( ( RANDOM % 40000 )  + 20000 ))
        CHECK=$(netstat -a | grep $port)
    done
    echo $port
}

############################## SETUP PORTS ##############################

# for debugging
set -x

port=$(getfreeport)
echo "Head node will use port: $port"
export port

dashboard_port=$(getfreeport)
echo "Dashboard will use port: $dashboard_port"
export dashboard_port

export RAY_GRAFANA_HOST=${port}:3000
export RAY_PROMETHEUS_HOST=${port}:9090

############################## START HEAD NODE ##############################

# NOTE: We allocate head node with only 4 CPUs      
#       to separate driver process from Ray worker processes
#       this will be beneficial later when we run the workload
#       since Ray will expect N gpus + Nxcpus_per_worker CPUs
#       + 1 cpu for the driver process and we want to avoid having
#       to allocate max_possible_cpus_per_worker - 1 just to 
#       ensure that the total number of CPUs is not exceeded

head_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_nodes_array=($head_nodes)
head_node=${head_nodes_array[0]}

read -r head_node_gpus head_node_cpus < <(get_node_resources "$head_node")

# we reserve 1 CPU for the training driver process
head_node_cpus=$(( head_node_cpus - 1 ))

mkdir -p $outdir/ray_head_node

srun --nodes=1 \
      --ntasks=1 \
      --gpus=$head_node_gpus \
      --cpus-per-task=$head_node_cpus \
      --job-name="Head_Node" \
      --output="$outdir/ray_head_node.out" \
      --error="$outdir/ray_head_node.err" \
      apptainer exec --userns --nv \
      --bind $outdir/ray_head_node:$tmpdir \
      --bind $bind \
      --bind ${src}:/workspace/segmentation/src/segmentation \
      --env PYTHONPATH=/workspace/segmentation/src \
      $env \
      $src/clusters/ray_start_cluster.sh \
      -p $port \
      -d $dashboard_port \
      -c $head_node_cpus \
      -g $head_node_gpus \
      -t $tmpdir &

sleep 15

############################## ADD WORKER NODES ##############################

# we allocate total workload for all worker nodes and 
# then allocate work per worker in the workload script
# this is done to ensure that we support both 
# allocation with exclusive and non-exclusive nodes

head_node_ip=$(hostname --ip-address)
cluster_address="$head_node_ip:$port"

export head_node 
export head_node_ip
export cluster_address

echo "Starting worker nodes for Ray cluster..."
worker_wrap+=" -a ${cluster_address} -t ${tmpdir}"
read -r -a opts_array <<<"$worker_opts"
jobid=$(sbatch --parsable "${opts_array[@]}" --wrap="${worker_wrap}")

workers_num_nodes=$(squeue -j $jobid -h -o "%D")

############################## CHECK RAY CLUSTER STATUS ##############################

echo "Checking Ray cluster status..."

apptainer exec --userns --nv \
      --bind $outdir/ray_head_node:$tmpdir \
      --bind $bind \
      $env $src/clusters/ray_check_status.sh \
      -a $cluster_address \
      -r $(( workers_num_nodes + 1 ))

############################## RUN WORKLOAD ##############################

echo "Starting workload: $workload"
# NOTE: this works because our training script calls ray.init() with the os.environ["head_node_ip"] 
#       which allows the worker nodes to connect to the head node for distributed training
#       across all worker nodes. Ray Actors will be created as needed for all gpu workers.
# NOTE: currently, we set the path to the repository directly while still
#      in development. In the future, we may include any necessary dependencies in the
#      directly in the container.

apptainer exec --userns --nv \
      --bind ${outdir}/ray_head_node:${tmpdir} \
      --bind ${bind} \
      --bind ${src}:/workspace/segmentation/src/segmentation \
      --env PYTHONPATH=/workspace/segmentation/src \
      ${env} \
      bash -c "${task}"

############################## CLEANUP ##############################

echo "Tearing down Ray cluster..."

apptainer exec --userns --nv \
   --bind $bind \
   --bind $outdir/ray_head_node:$tmpdir \
   $env ray stop --force


readarray -t worker_job_ids < <( squeue -j "$jobid" -h -o "%A" )

echo "Cancelling worker jobs: ${worker_job_ids[*]}"
scancel "${worker_job_ids[@]}"

echo "Cancelling head node job: $SLURM_JOB_ID"
scancel $SLURM_JOB_ID

######################################################################