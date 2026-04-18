We want to benchmark dataiku against my local computer and an AWS instance. The goal is to identify potential bottlenecks in the dataiku server we have, especially when training a machine learning model. 
Write one or multiple benchmarks, targetting every relevant hardware components, such that we can identify the bottlenecks, and conclude on what hardware to scale or change. 
You may use uv, especially uv add to install packages and uv run to run commands.  
You may use pytest to test your implementation. 


`uv run .\src\benchmark\pipeline\train.py --model-size medium --bench-iters 100 --amp --precision fp16 --export data/train_local.json`

# Notes

## Data

Switch to memory-mapped dataset: LMDB, HDF5
Set pin_memory=True and prefetch_factor=4 on DataLoader
Dataiku: confirm storage is NVMe-local, not NFS-mounted 

## Host -> dev
Verify PCIe width: `nvidia-smi -q | grep 'Bus'` — must be x16, not x8
Dataiku shared nodes: PCIe lane may be shared — request dedicated GPU allocation