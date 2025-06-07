To verify your solution follow the steps:
1. Place the contents of this archive on the Okeanos
2. Place the ZIP file with your solution in the same directory (the one holding run_test.py and other files of this checker)
3. Unpack your solution
4. Execute:
``` sh
sbatch sbatch_run_tests.sh
```
5. You can monitor the progress of the scheduled task using the `squeue` command (you can filter the output by `squeue | grep <your_login>`)
6. After the task finishes (is no longer vibile in the `squeue` command output) the content of the `output.txt` file should be as follows (assuming ab123456 is your students login):

Solution: ab123456
    path_20_4: PASSED
    random_20_4: PASSED
