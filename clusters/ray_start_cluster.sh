export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export NCCL_P2P_LEVEL=NVL

while getopts ":p:d:c:g:t:" option;do
    case "${option}" in
    p)  p=${OPTARG}
        port=$p
        echo port=$port
    ;;
    d)  d=${OPTARG}
        dashboard_port=$d
        echo dashboard_port=$dashboard_port
    ;;
    c)  c=${OPTARG}
        cpus=$c
        echo cpus=$cpus
    ;;
    g)  g=${OPTARG}
        gpus=$g
        echo gpus=$gpus
    ;;
    t)  t=${OPTARG}
        tmpdir=$t
        echo tmpdir=$tmpdir
    ;;
    *)  echo "Did not supply the correct arguments"
    ;;
    esac
done

ip=$(hostname --ip-address)

cluster_address="$ip:$port"

echo "Starting ray head node @ $(hostname) => $cluster_address with CPUs[$cpus] & GPUs [$gpus]"

ray start --head --node-ip-address=$ip \
    --port=$port \
    --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0 \
    --min-worker-port 18999 --max-worker-port 19999 \
    --temp-dir=$tmpdir \
    --num-cpus=$cpus \
    --num-gpus=$gpus &

sleep 15

echo "Starting metrics server for Ray cluster..."

ray metrics launch-prometheus &

sleep infinity