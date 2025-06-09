import os
import subprocess
from pathlib import Path
import argparse

def generate_outputs(test_dir: Path, use_local: bool):
    for test_case in test_dir.iterdir():
        if not test_case.is_dir():
            continue

        workers = len(list(test_case.glob("*.in")))
        print(f"Generating outputs for test case: {test_case.name} with {workers} workers")

        # Clean any previous outputs
        for i in range(workers):
            output_file = test_case / f"{i}.out"
            if output_file.exists():
                output_file.unlink()

        # Command to run sssp on this test
        command = ["mpiexec" if use_local else "srun", "-n", str(workers), "./generate_command.sh", test_case.name]

        result = subprocess.run(command, capture_output=True)
        if result.returncode != 0:
            print(f"FAILED to generate for {test_case.name}")
            print(result.stderr.decode())
        else:
            print(f"Generated successfully for {test_case.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate outputs for new_tests.")
    parser.add_argument('--dir', default="new_tests", help='Directory containing test cases')
    parser.add_argument('--local', action='store_true', help='Use mpiexec locally instead of srun')

    args = parser.parse_args()

    generate_outputs(Path(args.dir), args.local)
