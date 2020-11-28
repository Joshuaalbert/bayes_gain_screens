import glob
import os
import shutil
import subprocess
import logging
logger = logging.getLogger(__name__)

from bayes_gain_screens.pipeline.env import Env


class CMD(object):
    """
    Class to create and call command from a given environment.
    __call__ bubbles up the exit status.
    working_dir: the directory where command will run.
    script_dir: the directory where to find the script to run.
    shell: str, 'python' or 'bash'
    exec_env: and Env instance
    skip: bool, if true then skip running the command.
    """
    def __init__(self, working_dir, script_dir, script_name, shell='python', exec_env:Env=None, skip=False):
        self.skip = skip
        self.cmd = [shell, os.path.join(script_dir, script_name)]
        self.working_dir = working_dir
        if exec_env is None:
            exec_env = Env()
        self.exec_env = exec_env


    def add(self, name, value):
        """
        Adds a command line argument in --name=value format.
        Args:
            name: str, name of arg
            value: str, value of arg
        Return:
            self
        """
        self.cmd.append("--{}={}".format(name, value))
        return self

    def __call__(self):
        """
        Constructs a call from a designated command and environment.
        Returns: Exit code of the command bubbled from the shell.
        """
        if self.skip:
            return None
        proc_log = os.path.join(self.working_dir, 'state.log')
        os.chdir(self.working_dir)
        ###
        # this is the main command that will be run.
        run_cmd = ' \\\n\t'.join(self.cmd + ['2>&1 | tee -a {}; exit ${{PIPESTATUS[0]}}'.format(proc_log)])
        # This is the command that will execute the above command in the correct env
        exec_command = self.exec_env.compose(run_cmd)
        logger.info("Running:\n{}".format(exec_command))
        try:
            exit_status = subprocess.call(exec_command, shell=True)
        except KeyboardInterrupt:
            exit_status = 1
        logger.info("Finisihed:\n{}\nwith exit code {}".format(exec_command, exit_status))
        return exit_status


class Step(object):
    """
    Step class defines what a step is.
    Args:
        name: step name
        deps: list of other step names, or Step instances
        **cmd_kwargs: dict of keyword arguments to pass to the command
    """
    def __init__(self, name, deps, **cmd_kwargs):
        self.name = name
        self.deps = [dep.name if isinstance(dep, Step) else dep for dep in list(deps)]
        self.cmd_kwargs = cmd_kwargs
        self.working_dir = None
        self.flag = None
        self.cmd_args = []

    def add_cmd_arg(self, name, value):
        self.cmd_args.append((name, value))
        return self

    def build_cmd(self):
        """
        Build the command.
        """
        if self.flag is None:
            raise ValueError("Flag is none for Step {}".format(self.name))
        if self.flag > 0:
            self.cmd = CMD(self.working_dir, **self.cmd_kwargs)
        else:
            self.cmd = CMD(self.working_dir, skip=True, **self.cmd_kwargs)
        for (name, value) in self.cmd_args:
            self.cmd.add(name, value)

    def get_dask_task(self):
        """
        Gets the dask entry (cmd, *deps)
        """
        if self.cmd is None:
            raise ValueError("Cmd is not built for step {}".format(self.name))
        return (self.cmd,) + tuple(self.deps)

    def build_working_dir(self, root_working_dir):
        """
        Builds the working directory, if flag is appropriate.
        Args:
            root_working_dir: str, root of pipeline within which to build working directories.
        """
        self.working_dir = make_working_dir(root_working_dir, self.name, self.flag)


def make_working_dir(root_working_dir, name, do_flag):
    """
    Create the working directory based in `root_working_dir`.

    Args:
        root_working_dir: base directory to run pipeline in.
        name: name of working_dir to build.
        do_flag: int, method to get working directory
            0=return most recent working_dir or make fresh if none exist,
            1=clobber old working dirs of same prefix, and make fresh one,
            2=make a new directory with name name_{idx}.

    Returns:

    """
    previous_working_dirs = sorted(glob.glob(os.path.join(root_working_dir, "{}*".format(name))))
    if len(previous_working_dirs) == 0:
        working_dir = os.path.join(root_working_dir, name)
        most_recent = working_dir
    else:
        working_dir = os.path.join(root_working_dir, "{}_{}".format(name, len(previous_working_dirs)))
        most_recent = previous_working_dirs[-1]

    if do_flag == 0:
        os.makedirs(most_recent, exist_ok=True)
        return most_recent
    if do_flag == 1:
        for dir in previous_working_dirs:
            if os.path.isdir(dir):
                logger.info("Removing old working dir: {}".format(dir))
                shutil.rmtree(dir)
        working_dir = os.path.join(root_working_dir, name)
        os.makedirs(working_dir)
        logger.info("Made fresh working dir: {}".format(working_dir))
        return working_dir
    if do_flag == 2:
        os.makedirs(working_dir)
        logger.info("Made fresh working dir: {}".format(working_dir))
        return working_dir