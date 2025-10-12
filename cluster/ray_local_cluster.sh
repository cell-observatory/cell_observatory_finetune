#!/usr/bin/env bash

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export RAY_DEDUP_LOGS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# parse args from `args_parser.sh` getopts
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DIR/args_parser.sh"

tmpdir=/tmp/symlink_$(uuidgen | cut -d "-" -f5)
echo "Create symlink: $outdir -> $tmpdir"

############################## SETUP PORTS
# for debugging
set -x

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

port=$(getfreeport)
echo "Head node will use port: $port"
export port

dashboard_port=$(getfreeport)
echo "Dashboard will use port: $dashboard_port"
export dashboard_port

############################## FIND NODES/HOSTS

head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

############################## START HEAD NODE

apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind --bind /dev/shm:/dev/shm \
    --bind $outdir:$tmpdir $env /workspace/cell_observatory_finetune/cluster/ray_start_cluster.sh \
    -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $gpus -t $tmpdir -q $object_store_memory &
sleep 10

check_headnode="apptainer exec --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env ray status --address $head_node_ip:$port"
while ! $check_headnode; do
    echo "Waiting for head node..."
    sleep 3
done

rpids=$(pgrep -u $USER ray)
echo "Ray head node PID:"
echo $rpids

############################## CLEANUP

cleanup() {
    ec=$?
    echo "Running job cleanup (exit code: $ec)"
    head_pid=$(cat "$outdir/cleanup_head.pid" 2>/dev/null || true)
    if [[ -n "$head_pid" ]]; then
        kill -TERM "$head_pid" 2>/dev/null || true
        for _ in {1..120}; do
            kill -0 "$head_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$head_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' SIGTERM SIGINT

############################## CHECK STATUS

echo apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind \
    --bind $outdir:$tmpdir $env /workspace/cell_observatory_finetune/cluster/ray_check_status.sh \
    -a $cluster_address -r 1

############################## RUN WORKLOAD

echo "Running user tasks"
echo $tasks
apptainer exec --userns --nv --bind $storage_server --bind $workspace --bind $bind --bind $outdir:$tmpdir $env $tasks