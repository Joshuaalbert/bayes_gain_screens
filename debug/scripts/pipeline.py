import os
import subprocess
import sys
import shutil
import glob
import argparse
import datetime
from timeit import default_timer


def cmd_call(cmd):
    print("{}".format(cmd))
    exit_status = subprocess.call(cmd, shell=True)
    if exit_status:
        raise ValueError("Failed to  run: {}".format(cmd))


def str_(s):
    """
    In python 3 turns bytes to string. In python 2 just passes string.
    :param s:
    :return:
    """
    try:
        return s.decode()
    except:
        return s


def create_qsub_script(working_dir, name, cmd):
    submit_script = os.path.join(working_dir, 'submit_script.sh')
    print("Creating qsub submit script: {}".format(submit_script))
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


class CMD(object):
    def __init__(self, working_dir, script_dir, script_name, shell='python', exec_env=None):
        self.cmd = [shell, os.path.join(script_dir, script_name)]
        self.working_dir = working_dir
        if exec_env is None:
            exec_env = Env()
        self.exec_env = exec_env

    def add(self, name, value):
        self.cmd.append("--{}={}".format(name, value))
        return self

    def __call__(self):
        proc_log = os.path.join(self.working_dir, 'state.log')
        ###
        # this is the main command that will be run.
        run_cmd = ' \\\n\t'.join(self.cmd + ['2>&1 | tee -a {}; exit ${{PIPESTATUS[0]}}'.format(proc_log)])
        # This is the command that will execute the above command in the correct bash env
        exec_command = self.exec_env.compose(run_cmd)
        print("Running:\n{}".format(exec_command))

        exit_status = subprocess.call(exec_command, shell=True)
        print("Finisihed:\n{}\nwith exit code {}".format(exec_command, exit_status))
        return exit_status


def make_working_dir(root_working_dir, name, do_flag):
    '''If 0 then return most recent working_dir, if 1 then most recent if it exists else stop, if 2 then make a new directory.'''
    previous_working_dirs = sorted(glob.glob(os.path.join(root_working_dir, "{}*".format(name))))
    if len(previous_working_dirs) == 0:
        working_dir = os.path.join(root_working_dir, name)
        most_recent = working_dir
    else:
        working_dir = os.path.join(root_working_dir, "{}_{}".format(name, len(previous_working_dirs)))
        most_recent = previous_working_dirs[-1]

    if do_flag == 0:
        return most_recent
    if do_flag == 1:
        for dir in previous_working_dirs:
            if os.path.isdir(dir):
                print("Removing old working dir: {}".format(dir))
                shutil.rmtree(dir)
        working_dir = os.path.join(root_working_dir, name)
        os.makedirs(working_dir)
        print("Made working dir: {}".format(working_dir))
        return working_dir
    if do_flag == 2:
        # if os.path.isdir(working_dir):
        #     shutil.rmtree(working_dir)
        #     print("Removed pre-existing working dir: {}".format(working_dir))
        os.makedirs(working_dir)
        print("Made working dir: {}".format(working_dir))
        return working_dir


def iterative_topological_sort(graph, start):
    """
    Get Depth-first topology.

    :param graph: dependency dict (like a dask)
        {'a':['b','c'],
        'c':['b'],
        'b':[]}
    :param start: str
        the node you want to search from.
        This is equivalent to the node you want to compute.
    :return: list of str
        The order get from `start` to all ancestors in DFS.
    """
    seen = set()
    stack = []  # path variable is gone, stack and order are new
    order = []  # order will be in reverse order at first
    q = [start]
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v)  # no need to append to path any more
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]:  # new stuff here!
                order.append(stack.pop())
            stack.append(v)

    return stack + order[::-1]  # new return value!


def now():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def execute_dask(dsk, key, timing_file=None, state_file=None):
    """
    Go through the dask in topo order using DFS to reach `key`
    :param dsk: dict
        Dask graph
    :param key: str
        The node you want to arrive at.
    :param timing_file: str
        Where to store timing info
    :param state_file:
        Where to store pipeline state
    :return:
    """
    graph = {k: v[1:] for k, v in dsk.items()}
    topo_order = iterative_topological_sort(graph, key)[::-1]
    print("Execution order shall be:")
    for k in topo_order:
        print("\t{}".format(k))
    res = {}
    with open(state_file, 'w') as state:
        state.write("{} | START_PIPELINE\n".format(now()))
        state.flush()
        for k in topo_order:
            print("{} | Executing task {}".format(now(), k))
            state.write("{} | START | {}\n".format(now(), k))
            state.flush()
            t0 = default_timer()
            res[k] = dsk[k][0]()
            time_to_run = default_timer() - t0
            if res[k] is not None:
                print("Task {} took {:.2f} hours".format(k, time_to_run / 3600.))
                if res[k] == 0:
                    state.write("{} | END | {}\n".format(now(), k))
                    state.flush()
                    if timing_file is not None:
                        update_timing(timing_file, k, time_to_run)
                else:
                    state.write("{} | FAIL | {}\n".format(now(), k))
                    state.flush()
                    print("FAILURE at: {}".format(k))
                    state.write("{} | PIPELINE_FAILURE\n".format(now()))
                    state.flush()
                    exit(3)
            else:
                state.write("{} | END_WITHOUT_RUN {}\n".format(now(), k))
                print("{} skipped.".format(k))
        state.write("{} | PIPELINE_SUCCESS\n".format(now()))
        state.flush()
    return res


def update_timing(timing_file, name, time):
    timings = {}
    if os.path.isfile(timing_file):
        with open(timing_file, 'r+') as f:
            for line in f:
                if line.strip() == "" or "#" in line:
                    continue
                line = [a.strip() for a in line.strip().split(',')]
                timings[line[0]] = line[1:]
    if name not in timings.keys():
        timings[name] = ["{:.2f}".format(time)]
    else:
        timings[name].append("{:.2f}".format(time))
    with open(timing_file, 'w') as f:
        for k, t in timings.items():
            f.write("{},{}\n".format(k, ",".join(t)))


class Step(object):
    def __init__(self, name, deps, **cmd_kwargs):
        self.name = name
        self.deps = list(deps)
        self.cmd_kwargs = cmd_kwargs
        self.working_dir = None
        self.flag = None

    def build_cmd(self):
        if self.flag > 0:
            self.cmd = CMD(self.name, **self.cmd_kwargs)
        else:
            self.cmd = lambda *x: None

    def get_dask_task(self):
        if self.cmd is None:
            raise ValueError("Cmd is not built for step {}".format(self.name))
        return (self.cmd,) + tuple(self.deps)

    def build_working_dir(self, root_working_dir):
        self.working_dir = make_working_dir(root_working_dir, self.name, self.flag)


def main(archive_dir, root_working_dir, script_dir, obs_num, region_file, ncpu, ref_dir, ref_image_fits,
         block_size,
         deployment_type,
         no_subtract,
         bind_dirs,
         lofar_sksp_simg,
         lofar_gain_screens_simg,
         bayes_gain_screens_simg,
         bayes_gain_screens_conda_env,
         auto_resume,
         **do_kwargs):
    for key in do_kwargs.keys():
        if not key.startswith('do_'):
            raise KeyError("One of the 'do_<step_name>' args is invalid {}".format(key))

    root_working_dir = os.path.abspath(root_working_dir)
    script_dir = os.path.abspath(script_dir)
    try:
        os.makedirs(root_working_dir)
    except:
        pass
    root_working_dir = os.path.join(root_working_dir, 'L{obs_num}'.format(obs_num=obs_num))
    try:
        os.makedirs(root_working_dir)
    except:
        pass
    archive_dir = os.path.abspath(archive_dir)
    if not os.path.isdir(archive_dir):
        raise IOError("Archive dir doesn't exist {}".format(archive_dir))
    if ref_image_fits is None:
        ref_image_fits = os.path.join(archive_dir, 'image_full_ampphase_di_m.NS.app.restored.fits')
    timing_file = os.path.join(root_working_dir, 'timing.txt')

    print("Changing to {}".format(root_working_dir))
    os.chdir(root_working_dir)

    state_file = os.path.join(root_working_dir, 'STATE')

    if region_file is None:
        region_file = os.path.join(root_working_dir, 'bright_calibrators.reg')
        print("Region file is None, thus assuming region file is {}".format(region_file))
    else:
        region_file = os.path.abspath(region_file)
        if not os.path.isfile(region_file):
            raise IOError(
                "Region file {} doesn't exist, should leave as None if you want to auto select calibrators.".format(
                    region_file))
        do_kwargs['do_choose_calibrators'] = 0
        print("Using supplied region file for calibrators {}".format(region_file))
        if not os.path.isfile(os.path.join(root_working_dir, 'bright_calibrators.reg')):
            cmd_call("rsync -avP {} {}".format(region_file, os.path.join(root_working_dir, 'bright_calibrators.reg')))
        else:
            raise IOError("Region file already found. Not copying provided one.")
        region_file = os.path.join(root_working_dir, 'bright_calibrators.reg')

    print("Constructing run environments")
    if lofar_sksp_simg is not None:
        if not os.path.isfile(lofar_sksp_simg):
            print("Singularity image {} doesn't exist. Better have lofar tools sourced for ddf-pipeline work.".format(
                lofar_sksp_simg))
            lofar_sksp_env = Env()
        else:
            if bind_dirs is None:
                bind_dirs = './'  # redundant placeholder
            lofar_sksp_env = SingularityEnv(lofar_sksp_simg, bind_dirs=bind_dirs)
    else:
        print("Not using SKSP image, so lofar software better be sourced already that can do ddf pipeline work.")
        lofar_sksp_env = Env()

    if lofar_gain_screens_simg is not None:
        if not os.path.isfile(lofar_gain_screens_simg):
            print("Singularity image {} doesn't exist. Better have lofar tools sourced for screen imaging.".format(
                lofar_gain_screens_simg))
            lofar_gain_screens_env = Env()
        else:
            if bind_dirs is None:
                bind_dirs = './'  # redundant placeholder
            lofar_gain_screens_env = SingularityEnv(lofar_gain_screens_simg, bind_dirs=bind_dirs)
    else:
        print("Not using lofar gain screens image, so lofar software better be sourced already that can image screens.")
        lofar_gain_screens_env = Env()

    if bayes_gain_screens_simg is not None:
        if not os.path.isfile(bayes_gain_screens_simg):
            print(
                "Singularity image {} doesn't exist. Better have bayes gain screens sourced.".format(
                    bayes_gain_screens_simg))
            bayes_gain_screens_env = Env()
        else:
            if bind_dirs is None:
                bind_dirs = './'  # redundant placeholder
            bayes_gain_screens_env = SingularityEnv(bayes_gain_screens_simg, bind_dirs=bind_dirs)
    else:
        print("Not using bayes gain screen image, so bayes_gain_screens better be installed in conda env: {}".format(
            bayes_gain_screens_conda_env))
        bayes_gain_screens_env = CondaEnv(bayes_gain_screens_conda_env)

    step_list = [
        Step('download_archive', [], script_dir=script_dir, script_name='download_archive.py', exec_env=lofar_sksp_env),
        Step('choose_calibrators', ['choose_calibrators'], script_dir=script_dir, script_name='choose_calibrators.py',
             exec_env=lofar_sksp_env),
        Step('subtract', ['choose_calibrators', 'download_archive'], script_dir=script_dir,
             script_name='sub-sources-outside-region-mod.py', exec_env=lofar_sksp_env),
        Step('solve_dds4', ['subtract'], script_dir=script_dir, script_name='solve_on_subtracted.py',
             exec_env=lofar_sksp_env),
        Step('slow_solve_dds4', ['solve_dds4', 'smooth_dds4'], script_dir=script_dir,
             script_name='slow_solve_on_subtracted.py', exec_env=lofar_sksp_env),
        Step('smooth_dds4', ['solve_dds4'], script_dir=script_dir, script_name='smooth_dds4_simple.py',
             exec_env=bayes_gain_screens_env),
        Step('tec_inference', ['solve_dds4', 'smooth_dds4'], script_dir=script_dir,
             script_name='tec_inference_improved.py', exec_env=bayes_gain_screens_env),
        Step('infer_screen', ['smooth_dds4', 'tec_inference'], script_dir=script_dir, script_name='infer_screen.py',
             exec_env=bayes_gain_screens_env),
        Step('merge_slow', ['slow_solve_dds4', 'smooth_dds4', 'infer_screen'], script_dir=script_dir,
             script_name='merge_slow.py', exec_env=bayes_gain_screens_env),
        Step('image_subtract_dirty', ['subtract'], script_dir=script_dir, script_name='image.py',
             exec_env=lofar_sksp_env),
        Step('image_dds4', ['solve_dds4'], script_dir=script_dir, script_name='image.py', exec_env=lofar_sksp_env),
        Step('image_smooth', ['smooth_dds4'], script_dir=script_dir, script_name='image.py',
             exec_env=lofar_gain_screens_env),
        Step('image_smooth_slow', ['smooth_dds4', 'merge_slow'], script_dir=script_dir, script_name='image.py',
             exec_env=lofar_gain_screens_env),
        Step('image_screen', ['infer_screen'], script_dir=script_dir, script_name='image.py',
             exec_env=lofar_gain_screens_env),
        Step('image_screen_slow', ['infer_screen', 'merge_slow'], script_dir=script_dir, script_name='image.py',
             exec_env=lofar_gain_screens_env),
    ]

    # building step map
    steps = {}
    for step in step_list:
        if step.name not in STEPS:
            raise KeyError("Step.name {} not a valid step.".format(step.name))
        for dep in step.deps:
            if dep not in STEPS:
                raise ValueError("Step {} dep {} invalid.".format(step.name, dep))
        for do_arg in do_kwargs.keys():
            if do_arg == "do_{}".format(step.name):
                step.flag = do_kwargs[do_arg]
                print("User requested step do_{}={}".format(step.name, step.flag))
                break
        steps[step.name] = step
    # possibly auto resuming by setting flag
    if auto_resume:
        print("Attempting auto resume")
        if not os.path.isfile(state_file):
            print("No state file: {}".format(state_file))
            print("Resume not possible. Trusting your user requested pipeline steps.")
        else:
            with open(state_file, 'r') as f:
                for line in f.readlines():
                    split = line.split("|")
                    if len(split) == 3:
                        name = split[1].strip()
                        status = split[2].strip()
                    else:
                        continue
                    if name not in steps.keys():
                        raise ValueError("Could not find step {}".format(name))
                    if "END" in status:
                        print("Auto-resume infers {} should be skipped.".format(name))
                        steps[name].flag = 0
    # make required working directories (no deleting
    for k, step in steps.items():
        step.build_working_dir(root_working_dir)
        step.build_cmd()
        step.cmd.add('working_dir', step.working_dir)

    data_dir = steps['download_archive'].working_dir

    steps['download_archive'].cmd \
        .add('obs_num', obs_num) \
        .add('archive_dir', archive_dir) \

    steps['choose_calibrators'].cmd \
        .add('region_file', region_file) \
        .add('ref_image_fits', ref_image_fits) \
        .add('flux_limit', 0.30) \
        .add('min_spacing_arcmin', 6.)

    steps['subtract'].cmd \
        .add('region_file', region_file) \
        .add('ncpu', ncpu) \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir)

    steps['solve_dds4'].cmd \
        .add('region_file', region_file) \
        .add('ncpu', ncpu) \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir) \

    steps['smooth_dds4'].cmd \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir) \

    steps['tec_inference'].cmd \
        .add('obs_num', obs_num) \
        .add('ncpu', ncpu // 2) \
        .add('data_dir', data_dir) \
        .add('ref_dir', ref_dir)

    steps['slow_solve_dds4'].cmd \
        .add('ncpu', ncpu) \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir) \

    steps['infer_screen'].cmd \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir) \
        .add('ref_image_fits', ref_image_fits) \
        .add('block_size', block_size) \
        .add('max_N', 250) \
        .add('ncpu', ncpu) \
        .add('ref_dir', ref_dir) \
        .add('deployment_type', deployment_type)

    steps['merge_slow'].cmd \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir) \

    steps['image_subtract_dirty'].cmd \
        .add('image_type', 'image_subtract_dirty') \
        .add('ncpu', ncpu) \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir) \
        .add('script_dir', script_dir)

    steps['image_dds4'].cmd \
        .add('image_type', 'image_dds4') \
        .add('ncpu', ncpu) \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir) \
        .add('script_dir', script_dir) \
        .add('use_init_dico', True)

    steps['image_smooth'].cmd \
        .add('image_type', 'image_smoothed') \
        .add('ncpu', ncpu) \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir) \
        .add('script_dir', script_dir) \
        .add('use_init_dico', True)

    steps['image_smooth_slow'].cmd \
        .add('image_type', 'image_smoothed_slow') \
        .add('ncpu', ncpu) \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir) \
        .add('script_dir', script_dir) \
        .add('use_init_dico', True)

    steps['image_screen'].cmd \
        .add('image_type', 'image_screen') \
        .add('ncpu', ncpu) \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir) \
        .add('script_dir', script_dir) \
        .add('use_init_dico', True)

    steps['image_screen_slow'].cmd \
        .add('image_type', 'image_screen_slow') \
        .add('ncpu', ncpu) \
        .add('obs_num', obs_num) \
        .add('data_dir', data_dir) \
        .add('script_dir', script_dir) \
        .add('use_init_dico', True)

    dsk = {}
    for name in steps.keys():
        dsk[name] = steps[name].get_dask_task()

    dsk['endpoint'] = (lambda *x: None,) + tuple([k for k in dsk.keys()])
    execute_dask(dsk, 'endpoint', timing_file=timing_file, state_file=state_file)


STEPS = [
    "download_archive",
    "choose_calibrators",
    "subtract",
    "solve_dds4",
    "slow_solve_dds4",
    "smooth_dds4",
    "tec_inference",
    "merge_slow",
    "image_subtract_dirty",
    "image_smooth",
    "image_dds4",
    "image_smooth_slow",
    "infer_screen",
    "image_screen",
    "image_screen_slow"]


def add_args(parser):
    def string_or_none(s):
        if s.lower() == 'none':
            return None
        else:
            return s

    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.register('type', 'str_or_none', string_or_none)

    required = parser.add_argument_group('Required arguments')
    env_args = parser.add_argument_group('Execution environment arguments')
    optional = parser.add_argument_group('Optional arguments')
    step_args = parser.add_argument_group('Enable/Disable steps')

    optional.add_argument('--no_download',
                          help='Whether to move instead of copy the archive dir. Potentially unsafe. Requires env variable set SP_AUTH="1".',
                          default=False, type="bool", required=False)
    optional.add_argument('--region_file',
                          help='ds9 region file defining calbrators. If not provided, they will be automatically determined.',
                          required=False, type='str_or_none',
                          default=None)
    optional.add_argument('--ref_dir',
                          help='Which direction to reference from. If not provided, it is the first (usually brightest) direction.',
                          required=False, type=int, default=0)
    optional.add_argument('--ref_image_fits',
                          help='Reference image used to extract screen directions and auto select calibrators if region_file is None. If not provided, it will use the one in the archive directory.',
                          required=False, default=None, type='str_or_none')
    optional.add_argument('--auto_resume',
                          help='Whether or try to automatically resume operations based on the STATE file.',
                          required=False, default=True, type='bool')

    try:
        workers = os.cpu_count()
        if 'sched_getaffinity' in dir(os):
            workers = len(os.sched_getaffinity(0))
    except:
        import multiprocessing
        workers = multiprocessing.cpu_count()

    optional.add_argument('--ncpu',
                          help='Number of processes to use at most. If not then set to number of available physical cores.',
                          default=workers, type=int, required=False)
    required.add_argument('--obs_num', help='Obs number L*',
                          default=None, type=int, required=True)
    required.add_argument('--archive_dir', help='Where are the archives stored. Can also be networked locations, e.g. <user>@<host>:<path> but you must have authentication.',
                          default=None, type=str, required=True)
    required.add_argument('--root_working_dir', help='Where the root of all working dirs are.',
                          default=None, type=str, required=True)
    required.add_argument('--script_dir', help='Where the scripts are located.',
                          default=None, type=str, required=True)
    optional.add_argument('--block_size',
                          help='Number of blocks to infer screen at a time for screen. Large blocks give better S/N for inferred kernel hyper params, but then the ionosphere might change in this time.',
                          default=10, type=int, required=False)
    optional.add_argument('--deployment_type',
                          help='Which type of deployment [directional, non_integral, tomographic]. Currently only directional should be used.',
                          default='directional', type=str, required=False)
    env_args.add_argument('--bind_dirs', help='Which directories to bind to singularity.',
                          default=None, type='str_or_none', required=False)
    env_args.add_argument('--lofar_sksp_simg',
                          help='The lofar SKSP singularity image. If None or doesnt exist then uses local env.',
                          default=None, type='str_or_none', required=False)
    env_args.add_argument('--lofar_gain_screens_simg',
                          help='Point to the lofar gain screens branch singularity image. If None or doesnt exist then uses local env.',
                          default=None, type='str_or_none', required=False)
    env_args.add_argument('--bayes_gain_screens_simg',
                          help='Point to the bayes_gain_screens singularity image. If None or doesnt exist then uses conda env.',
                          default=None, type='str_or_none', required=False)
    env_args.add_argument('--bayes_gain_screens_conda_env',
                          help='The conda env to use if bayes_gain_screens_simg not provided.',
                          default='tf_py', type=str, required=False)

    for s in STEPS:
        step_args.add_argument('--do_{}'.format(s),
                               help='Do {}? (NO=0/YES_CLOBBER=1/YES_NO_CLOBBER=2)'.format(s),
                               default=0, type=int, required=False)


def test_main():
    main('/home/albert/store/lockman/archive',  # P126+65',
         '/home/albert/store/root_redo',
         '/home/albert/store/scripts',
         obs_num=342938,  # 664320,#667204,#664480,#562061,
         region_file='/home/albert/store/lockman/LHdeepbright.reg',
         ref_image_fits=None,  # '/home/albert/store/lockman/lotss_archive_deep_image.app.restored.fits',
         ncpu=32,
         ref_dir=0,
         block_size=50,
         deployment_type='directional',
         no_download=False,
         bind_dirs='/beegfs/lofar',
         lofar_sksp_simg='/home/albert/store/lofar_sksp_ddf.simg',
         lofar_gain_screens_simg='/home/albert/store/lofar_sksp_ddf_gainscreens_premerge.simg',
         bayes_gain_screens_simg=None,
         bayes_gain_screens_conda_env='tf_py',
         do_choose_calibrators=0,
         do_download_archive=0,
         do_subtract=0,
         do_image_subtract_dirty=0,
         do_solve_dds4=0,
         do_smooth_dds4=0,
         do_slow_solve_dds4=0,
         do_tec_inference=2,
         do_merge_slow=0,
         do_infer_screen=0,
         do_image_dds4=0,
         do_image_smooth=0,
         do_image_smooth_slow=0,
         do_image_screen=0,
         do_image_screen_slow=0)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        test_main()
        exit(0)
    parser = argparse.ArgumentParser(
        description='Runs full pipeline on a single obs_num.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))

    main(**vars(flags))
