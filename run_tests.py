import argparse
from pathlib import Path
import subprocess
import filecmp
import datetime
import time
import re
import os


def run_tests(break_on_fail, local):   
    now = datetime.datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    print(f"Starting tests at {formatted_time}")
    Path("outputs").mkdir(parents=True, exist_ok=True)

    for solution in Path(".").iterdir():
        if solution.is_dir() and solution.name not in ["tests", "outputs"]:
            make = subprocess.run("make", cwd=solution.name, capture_output=True, timeout=300)
            if make.returncode != 0:
                print(f"{solution.name}: FAILED (make)")
                if break_on_fail:
                    print(make.stdout.decode())
                    exit(1)
                continue

            print(f"Solution: {solution.name}")
            for test in Path("tests").iterdir():
                workers = int(test.name[test.name.rfind("_") + 1:])
                
                match = re.search(r"_n_(\d+)_k_", test.name)
                if match:
                    n_value = int(match.group(1))
                    edges = n_value * 16
                else:
                    n_value = 0
                    edges = 0

                for f in Path("outputs").iterdir():
                    f.unlink()

                command = "mpiexec" if local else "srun"
                start_time = time.time()
                execution = subprocess.run(
                    [command, "-n", str(workers), "./test_command.sh", solution.name, test.name],
                    capture_output=True,
                    timeout=300
                )
                end_time = time.time()
                duration = end_time - start_time

                if execution.returncode != 0:
                    print(f"    {test.name}: FAILED (srun) [Took {duration:.3f} sec, Edges: {edges}]")
                    if break_on_fail:
                        print(execution.stdout.decode())
                        exit(1)
                    continue

                failed = False
                for i in range(workers):
                    expected = f"tests/{test.name}/{i}.out"
                    actual = f"outputs/{i}.out"
                    try:
                        if not filecmp.cmp(expected, actual, shallow=False):
                            print(f"    {test.name}: FAILED (outputs differ on rank {i}) [Took {duration:.3f} sec, Edges: {edges}]")
                            failed = True
                            if break_on_fail:
                                exit(1)
                            break
                    except FileNotFoundError as e:
                        print(f"    {test.name}: FAILED (missing file on rank {i})")
                        print(f"        Missing file: {e.filename}")
                        failed = True
                        if break_on_fail:
                            exit(1)
                        break

                if not failed:
                    print(f"    {test.name}: PASSED [Took {duration:.3f} sec, Edges: {edges}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test runner')
    parser.add_argument('-b', '--breakonfail', action='store_true', help='break and print stdout on fail')
    parser.add_argument('-l', '--local', action='store_true', help='run tests locally (without slurm)')

    args = parser.parse_args()
    run_tests(args.breakonfail, args.local)
