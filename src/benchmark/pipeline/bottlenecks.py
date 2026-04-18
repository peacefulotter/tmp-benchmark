import json

aws = json.load(open("results/aws_g5xlarge.json"))
dataiku = json.load(open("results/dataiku_node.json"))

for stage in ["data_load", "forward_pass", "backward_pass", "host_to_device"]:
    a = next(b for b in aws["bottlenecks"] if b["stage"] == stage)
    d = next(b for b in dataiku["bottlenecks"] if b["stage"] == stage)
    delta_pct = (d["mean_ms"] - a["mean_ms"]) / a["mean_ms"] * 100
    print(
        f"{stage:<22}  AWS={a['mean_ms']:>7.1f}ms  Dataiku={d['mean_ms']:>7.1f}ms  Δ={delta_pct:+.1f}%  [{a['severity']} / {d['severity']}]"
    )

print()
print(
    f"Training throughput  AWS={aws['throughput_train']:.0f} samp/s   Dataiku={dataiku['throughput_train']:.0f} samp/s"
)
print(
    f"Throttle detected    AWS={aws['throttle_detected']}   Dataiku={dataiku['throttle_detected']}"
)
