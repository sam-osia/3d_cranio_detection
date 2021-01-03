import os
from subprocess import call
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-S', '--script', required=True)
parser.add_argument('-R', '--runs', default=None, required=True)
parser.add_argument('-T', '--tag', default=None)

inputs = parser.parse_args()

for i in range(int(inputs.runs)):
    command = f'./submit_script.sh {inputs.script}'
    args = f'--run_id {i}'
    if inputs.tag is not None:
        args = f'{args} --tag {inputs.tag}'
    print(f'{command} "{args}"')
    call(f'{command} "{args}"', shell=True)
    print()
