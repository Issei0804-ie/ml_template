import argparse
import os

from fabric import Connection

import conf

HOSTNAME = "ie-gpu"
host = Connection(HOSTNAME)

p = argparse.ArgumentParser(description="deploy to ie-gpu")
p.add_argument("branch_name")
args = p.parse_args()
BRANCH_NAME = args.branch_name
RSYNC_FILES = conf.RSYNC_FILES
IMAGE_SOURCE = os.path.join("~", conf.WORK_DIR, "torch.sif")

host.run(f"mkdir -p {os.path.join(conf.WORK_DIR, conf.REPOSITORY_NAME)}")
with host.cd(os.path.join(conf.WORK_DIR, conf.REPOSITORY_NAME)):
    result = host.run("ls")
    dirs = result.stdout.split("\n")
    print(dirs)
    if not BRANCH_NAME in dirs:
        host.run(f"git clone {conf.GIT_LINK} -b {BRANCH_NAME} {BRANCH_NAME}")
        host.run(f"cp {IMAGE_SOURCE} {BRANCH_NAME}")
    with host.cd(BRANCH_NAME):
        result = host.run(f"git pull")
        print(result)
        for file in RSYNC_FILES:
            os.system(
                f"rsync -avhz {file} {HOSTNAME}:{os.path.join('~', conf.WORK_DIR, conf.REPOSITORY_NAME, BRANCH_NAME)}"
            )
        host.run(f"make slurm-run")
