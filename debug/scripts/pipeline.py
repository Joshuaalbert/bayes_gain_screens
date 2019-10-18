import os
import glob
import argparse
from timeit import default_timer


class CMD(object):
    def __init__(self, script_dir, script_name, shell='python'):
        self.cmd = ["{} {}".format(os.path.join(shell, script_dir, script_name))]

    def add(self, name, value):
        self.cmd.append("--{}={}".format(name, value))

    def __call__(self):
        cmd = ' \\n'.format(self.cmd)
        print("Running {}".format(cmd))
        t0 = default_timer()
        os.system(cmd)
        self.time_to_run = default_timer() - t0
        print("Finisihed {}".format(cmd))
        print("Took {} minutes".format(self.time_to_run / 60.))


def make_working_dir(root_working_dir, name, do_flag):
    working_dir = os.path.join(root_working_dir, name)
    if do_flag == 0:
        return None
    if do_flag == 1:
        if os.path.isdir(working_dir):
            raise IOError("{} working dir {} exists.".format(name, working_dir))
        os.makedirs(working_dir)
    if do_flag == 2:
        if os.path.isdir(working_dir):
            os.unlink(working_dir)
        os.makedirs(working_dir)
    return working_dir

def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--region_file', help='boxfile, required argument', required=True, type=str)
    parser.add_argument('--ncpu', help='number of cpu to use', default=32, type=int)
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--archive_dir', help='Where are the archives stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--root_working_dir', help='Where the root of all working dirs are.',
                        default=None, type=str, required=True)
    parser.add_argument('--script_dir', help='Where the scripts are located.',
                        default=None, type=str, required=True)

    parser.add_argument('--do_subtract', help='Do this task. 0=skip, 1=do if no working directory exists, 2=clobber old working directory and redo.',default=1, type=int, required=False)
    parser.add_argument('--do_image_subtract_dirty', help='Do this task. 0=skip, 1=do if no working directory exists, 2=clobber old working directory and redo.',default=1, type=int, required=False)
    parser.add_argument('--do_solve_dds4', help='Do this task. 0=skip, 1=do if no working directory exists, 2=clobber old working directory and redo.',default=1, type=int, required=False)
    parser.add_argument('--do_smooth_dds4', help='Do this task. 0=skip, 1=do if no working directory exists, 2=clobber old working directory and redo.',default=1, type=int, required=False)
    parser.add_argument('--do_slow_dds4', help='Do this task. 0=skip, 1=do if no working directory exists, 2=clobber old working directory and redo.',default=1, type=int, required=False)
    parser.add_argument('--do_image_smooth', help='Do this task. 0=skip, 1=do if no working directory exists, 2=clobber old working directory and redo.',default=1, type=int, required=False)
    parser.add_argument('--do_image_smooth_slow', help='Do this task. 0=skip, 1=do if no working directory exists, 2=clobber old working directory and redo.',default=1, type=int, required=False)
    parser.add_argument('--do_tec_inference', help='Do this task. 0=skip, 1=do if no working directory exists, 2=clobber old working directory and redo.',default=1, type=int, required=False)
    parser.add_argument('--do_infer_screen', help='Do this task. 0=skip, 1=do if no working directory exists, 2=clobber old working directory and redo.',default=1, type=int, required=False)
    parser.add_argument('--do_image_screen', help='Do this task. 0=skip, 1=do if no working directory exists, 2=clobber old working directory and redo.',default=1, type=int, required=False)
    parser.add_argument('--do_image_screen_slow', help='Do this task. 0=skip, 1=do if no working directory exists, 2=clobber old working directory and redo.',default=1, type=int, required=False)


def iterative_topological_sort(graph, start):
    seen = set()
    stack = []    # path variable is gone, stack and order are new
    order = []    # order will be in reverse order at first
    q = [start]
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v) # no need to append to path any more
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]: # new stuff here!
                order.append(stack.pop())
            stack.append(v)

    return stack + order[::-1]   # new return value!


def execute_dask(dsk, key):
    graph = {k: v[1:] for k,v in dsk.items()}
    topo_order = iterative_topological_sort(graph, key)[::-1]
    res = {}
    for k in topo_order:
        res[k] = dsk[k][0]()
    return res

def main(archive_dir, root_working_dir, script_dir, obs_num, region_file, ncpu,
         do_subtract,
         do_image_subtract_dirty,
         do_solve_dds4,
         do_smooth_dds4,
         do_slow_dds4,
         do_image_smooth,
         do_image_smooth_slow,
         do_tec_inference,
         do_infer_screen,
         do_image_screen,
         do_image_screen_slow):
    archive_dir = os.path.abspath(archive_dir)
    root_working_dir = os.path.abspath(root_working_dir)
    script_dir = os.path.abspath(script_dir)
    region_file = os.path.abspath(region_file)
    try:
        os.makedirs(root_working_dir)
    except:
        pass
    working_dir = os.path.join(root_working_dir, 'L{obs_num}'.format(obs_num=obs_num))
    try:
        os.makedirs(working_dir)
    except:
        pass
    print("Changing to {}".format(working_dir))
    os.chdir(working_dir)
    subtract_working_dir = make_working_dir(root_working_dir, 'subtract', do_subtract)
    image_subtract_dirty_working_dir = make_working_dir(root_working_dir, 'image subtract', do_image_subtract_dirty)
    solve_dds4_working_dir = make_working_dir(root_working_dir, 'solve dds4', do_solve_dds4)
    smooth_dds4_working_dir = make_working_dir(root_working_dir, 'smooth dds4', do_smooth_dds4)
    slow_dds4_working_dir = make_working_dir(root_working_dir, 'slow dds4', do_slow_dds4)
    image_smooth_working_dir = make_working_dir(root_working_dir, 'image smooth', do_image_smooth)
    image_smooth_slow_working_dir = make_working_dir(root_working_dir, 'image smooth slow', do_image_smooth_slow)
    tec_inference_working_dir = make_working_dir(root_working_dir, 'tec inference', do_tec_inference)
    infer_screen_working_dir = make_working_dir(root_working_dir, 'infer screen', do_infer_screen)
    image_screen_working_dir = make_working_dir(root_working_dir, 'image screen', do_image_screen)
    image_screen_slow_working_dir = make_working_dir(root_working_dir, 'image screen slow', do_image_screen_slow)

    dsk = {}
    if subtract_working_dir is not None:
        cmd = CMD(script_dir, 'sub-sources-outside-region-mod.py')
        cmd.add('region_file', region_file)
        cmd.add('ncpu', ncpu)
        cmd.add('obs_num', obs_num)
        cmd.add('data_dir', archive_dir)
        cmd.add('working_dir', subtract_working_dir)
        dsk['subtract'] = (cmd,)
    else:
        dsk['subtract'] = (lambda *x: None,)

    if image_subtract_dirty_working_dir is not None:
        cmd = CMD(script_dir, 'image.py')
        cmd.add('image_type', 'image_subtract_dirty')
        cmd.add('ncpu', ncpu)
        cmd.add('obs_num', obs_num)
        cmd.add('data_dir', subtract_working_dir)
        cmd.add('working_dir', image_subtract_dirty_working_dir)
        dsk['image_subtract_dirty'] = (cmd, 'subtract')
    else:
        dsk['image_subtract_dirty'] = (lambda *x: None, 'subtract')

    if solve_dds4_working_dir is not None:
        cmd = CMD(script_dir, 'solve_on_subtracted.py')
        cmd.add('region_file', region_file)
        cmd.add('ncpu', ncpu)
        cmd.add('obs_num', obs_num)
        cmd.add('data_dir', subtract_working_dir)
        cmd.add('working_dir', solve_dds4_working_dir)
        dsk['solve_dds4'] = (cmd, 'subtract')
    else:
        dsk['solve_dds4'] = (lambda *x: None, 'subtract')

    if smooth_dds4_working_dir is not None:
        cmd = CMD(script_dir, 'smooth_dds4.sh', 'bash')
        cmd.add('conda_env', 'tf_py')
        cmd.add('script_dir', script_dir)
        cmd.add('obs_num', obs_num)
        cmd.add('data_dir', subtract_working_dir)
        cmd.add('working_dir', smooth_dds4_working_dir)
        dsk['smooth_dds4'] = (cmd, 'subtract')
    else:
        dsk['smooth_dds4'] = (lambda *x: None, 'subtract')

    if slow_dds4_working_dir is not None:
        cmd = CMD(script_dir, 'slow_solve_on_subtracted.py')
        cmd.add('ncpu', ncpu)
        cmd.add('obs_num', obs_num)
        cmd.add('data_dir', subtract_working_dir)
        cmd.add('working_dir', slow_dds4_working_dir)
        dsk['slow_solve_dds4'] = (cmd, 'smooth_dds4')
    else:
        dsk['slow_solve_dds4'] = (lambda *x: None, 'smooth_dds4')

    if image_smooth_working_dir is not None:
        cmd = CMD(script_dir, 'image.py')
        cmd.add('image_type', 'image_smoothed')
        cmd.add('ncpu', ncpu)
        cmd.add('obs_num', obs_num)
        cmd.add('data_dir', subtract_working_dir)
        cmd.add('working_dir', image_smooth_working_dir)
        dsk['image_smooth'] = (cmd, 'smooth_dds4')
    else:
        dsk['image_smooth'] = (lambda *x: None, 'smooth_dds4')

    if image_smooth_slow_working_dir is not None:
        cmd = CMD(script_dir, 'image.py')
        cmd.add('image_type', 'image_smoothed_slow')
        cmd.add('ncpu', ncpu)
        cmd.add('obs_num', obs_num)
        cmd.add('data_dir', subtract_working_dir)
        cmd.add('working_dir', image_smooth_slow_working_dir)
        dsk['image_smooth_slow'] = (cmd, 'infer_screen', 'smooth_dds4', 'slow_solve_dds4')
    else:
        dsk['image_smooth_slow'] = (lambda *x: None, 'infer_screen', 'smooth_dds4', 'slow_solve_dds4')

    if tec_inference_working_dir is not None:
        cmd = CMD(script_dir, 'tec_inference.sh', 'bash')
        cmd.add('conda_env', 'tf_py')
        cmd.add('script_dir', script_dir)
        cmd.add('obs_num', obs_num)
        cmd.add('data_dir', subtract_working_dir)
        cmd.add('working_dir', tec_inference_working_dir)
        dsk['tec_inference'] = (cmd, 'smooth_dds4', 'solve_dds4')
    else:
        dsk['tec_inference'] = (lambda *x: None, 'smooth_dds4', 'solve_dds4')

    if infer_screen_working_dir is not None:
        cmd = CMD(script_dir, 'infer_screen.sh', 'bash')
        cmd.add('conda_env', 'tf_py')
        cmd.add('script_dir', script_dir)
        cmd.add('obs_num', obs_num)
        cmd.add('data_dir', subtract_working_dir)
        cmd.add('working_dir', infer_screen_working_dir)
        dsk['infer_screen'] = (cmd, 'tec_inference')
    else:
        dsk['infer_screen'] = (lambda *x: None, 'tec_inference')

    if image_screen_working_dir is not None:
        cmd = CMD(script_dir, 'image.py')
        cmd.add('image_type', 'image_screen')
        cmd.add('ncpu', ncpu)
        cmd.add('obs_num', obs_num)
        cmd.add('data_dir', subtract_working_dir)
        cmd.add('working_dir', image_screen_working_dir)
        dsk['image_screen'] = (cmd, 'infer_screen')
    else:
        dsk['image_screen'] = (lambda *x: None, 'infer_screen')

    if image_screen_slow_working_dir is not None:
        cmd = CMD(script_dir, 'image.py')
        cmd.add('image_type', 'image_screen_slow')
        cmd.add('ncpu', ncpu)
        cmd.add('obs_num', obs_num)
        cmd.add('data_dir', subtract_working_dir)
        cmd.add('working_dir', image_screen_slow_working_dir)
        dsk['image_screen_slow'] = (cmd, 'infer_screen', 'slow_solve_dds4')
    else:
        dsk['image_screen_slow'] = (lambda *x: None, 'infer_screen', 'slow_solve_dds4')


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--region_file', help='boxfile, required argument', required=True, type=str)
    parser.add_argument('--ncpu', help='number of cpu to use', default=32, type=int)
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--archive_dir', help='Where are the archives stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--root_working_dir', help='Where the root of all working dirs are.',
                        default=None, type=str, required=True)
    parser.add_argument('--script_dir', help='Where the scripts are located.',
                        default=None, type=str, required=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs full pipeline on a single obs_num.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))
