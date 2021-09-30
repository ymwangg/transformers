import subprocess
import argparse
import re
"""
Matches results print from HF benchmarks script:
--------------------------------------------------------------------------------
          Model Name             Batch Size     Seq Length     Time in s   
--------------------------------------------------------------------------------
      bert-base-uncased              4              128            0.062     
--------------------------------------------------------------------------------
"""
RESULTS_REGEX = "(?<=-{80}\n) *\S+ *\S+ *\S+ *\S+ *(?=\n-{80})"


def fetch_latency(output):
    match = re.search(RESULTS_REGEX, output)
    if match:
        return match.group(0).split()[-1]
    return "N/A"


def run_command(cmd):
    output = subprocess.run(cmd,
                            stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE,
                            shell=True)
    return output.stdout.decode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--use_xla", action='store_true')
    parser.add_argument('-m',
                        '--models',
                        nargs='+',
                        default=[
                            "bert-base-uncased", "bert-large-uncased",
                            "/pytorch/xla/hopper/test/files/bart-base.json",
                            "roberta-base", "gpt2"
                        ])
    parser.add_argument('-s',
                        '--sequence_lengths',
                        nargs='+',
                        type=int,
                        default=[128, 512])
    parser.add_argument(
        '-b',
        '--batch_sizes',
        nargs='+',
        type=int,
        default=[1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 96, 128])
    parser.add_argument('-d',
                        '--transformers-dir',
                        default="/opt/ml/code/transformers")
    args = parser.parse_args()

    for model in args.models:
        for seq_len in args.sequence_lengths:
            for batch_size in args.batch_sizes:
                # print("running {} batch_size={} sequence_length={}".format(
                #     model, batch_size, seq_len))
                cmd = "python3 {}/examples/pytorch/benchmarking/run_benchmark.py --models {} --training yes --batch_sizes {} --sequence_lengths {} --inference no --memory false --fp16".format(
                    args.transformers_dir, model, batch_size, seq_len)

                cmd_xla_sgd = cmd + " --tpu true --optim SGD"
                cmd_xla_adam = cmd + " --tpu true --optim Adam"
                cmd_xla_syncfree_sgd = cmd + " --tpu true --optim SGD --syncfree"
                cmd_xla_syncfree_adam = cmd + " --tpu true --optim Adam --syncfree"
                cmd_sgd = cmd + " --tpu false --optim SGD"
                cmd_adam = cmd + " --tpu false --optim Adam"
                cmds = [cmd_sgd, cmd_xla_sgd, cmd_xla_syncfree_sgd, cmd_adam, cmd_xla_adam, cmd_xla_syncfree_adam]
                latencies = []
                for cmd in cmds:
                    print(cmd)
                    latencies.append(fetch_latency(run_command(cmd)))
                model_name = model
                if model_name.endswith(".json"):
                    model_name = model_name.split("/")[-1][:-5]
                print(
                    "Columns: model, seq_len, batch_size, native_sgd, xla_sgd, syncfree_sgd, native_adam, xla_adam, syncfree_adam"
                )
                print(
                    f"{model_name},{seq_len},{batch_size}," + ",".join(map(str,latencies))
                )
