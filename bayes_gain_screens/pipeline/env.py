import os
import logging
logger = logging.getLogger(__name__)

class Env(object):
    def __init__(self, *args, **kwargs):
        pass

    def compose(self, cmd):
        return "bash -c '{cmd}'".format(cmd=cmd)


class SingularityEnv(Env):
    def __init__(self, image, bind_dirs):
        super(SingularityEnv, self).__init__()
        self.image = image
        self.bind_dirs = bind_dirs

    def compose(self, cmd):
        exec_cmd = "singularity exec -B /tmp,/dev/shm,$HOME,{bind_dirs} {image} \\\n{cmd}".format(
            bind_dirs=self.bind_dirs, image=self.image, cmd=cmd)
        return exec_cmd


class CondaEnv(Env):
    def __init__(self, conda_env):
        super(CondaEnv, self).__init__()
        self.conda_env = conda_env

    def compose(self, cmd):
        exec_cmd = "bash -c 'source $HOME/.bashrc; conda activate {conda_env}; export PYTHONPATH=; {cmd}'".format(
            conda_env=self.conda_env, cmd=cmd)
        return exec_cmd


def create_qsub_script(working_dir, name, cmd):
    """
    Create a qsub script.
    Args:
        working_dir: working dir
        name: name of job
        cmd: command to run

    Returns: path to submit script.
    """
    submit_script = os.path.join(working_dir, 'submit_script.sh')
    logger.info("Creating qsub submit script: {}".format(submit_script))
    with open(submit_script, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#PBS -N {}\n'.format(name))
        f.write('#PBS -q main\n')
        f.write('#PBS -v\n')
        f.write('#PBS -w e\n')
        f.write('#PBS -l nodes=1:ppn=32\n')
        f.write('#PBS -l walltime=168:00:00\n')
        f.write(cmd)
    return submit_script