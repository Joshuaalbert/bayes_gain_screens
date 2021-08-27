import os
import subprocess
import sys
import argparse

from bayes_gain_screens.pipeline.env import Env, SingularityEnv, CondaEnv
from bayes_gain_screens.pipeline.pipeline import Pipeline
from bayes_gain_screens.pipeline.step import Step

import logging

logger = logging.getLogger(__name__)


def cmd_call(cmd):
    logger.info("{}".format(cmd))
    exit_status = subprocess.call(cmd, shell=True)
    if exit_status:
        raise ValueError("Failed to  run: {}".format(cmd))


def main(archive_dir, script_dir, root_working_dir, obs_num, region_file, ncpu, ref_image_fits,
         retry_task_on_fail,
         no_download,
         bind_dirs,
         lofar_sksp_simg,
         lofar_gain_screens_simg,
         bayes_gain_screens_simg,
         bayes_gain_screens_conda_env,
         auto_resume,
         **do_kwargs):
    if script_dir is None:
        script_dir = os.path.join(os.path.dirname(sys.modules["bayes_gain_screens"].__file__),'steps')

    for key in do_kwargs.keys():
        if not key.startswith('do_'):
            raise KeyError("One of the 'do_<step_name>' args is invalid {}".format(key))

    root_working_dir = os.path.abspath(root_working_dir)
    script_dir = os.path.abspath(script_dir)
    os.makedirs(root_working_dir, exist_ok=True)
    root_working_dir = os.path.join(root_working_dir, 'L{obs_num}'.format(obs_num=obs_num))
    os.makedirs(root_working_dir, exist_ok=True)
    logger.info("Changing to {}".format(root_working_dir))
    os.chdir(root_working_dir)

    timing_file = os.path.join(root_working_dir, 'timing.txt')
    state_file = os.path.join(root_working_dir, 'STATE')
    archive_dir = os.path.abspath(archive_dir)
    if not os.path.isdir(archive_dir):
        raise IOError("Archive dir doesn't exist {}".format(archive_dir))
    if ref_image_fits is None:
        ref_image_fits = os.path.join(archive_dir, 'image_full_ampphase_di_m.NS.app.restored.fits')
    if region_file is None:
        region_file = os.path.join(root_working_dir, 'bright_calibrators.reg')
        logger.info("Region file is None, thus assuming region file is {}".format(region_file))
    else:
        region_file = os.path.abspath(region_file)
        if not os.path.isfile(region_file):
            raise IOError(
                "Region file {} doesn't exist, should leave as None if you want to auto select calibrators.".format(
                    region_file))
        do_kwargs['do_choose_calibrators'] = 0
        logger.info("Using supplied region file for calibrators {}".format(region_file))
        if not os.path.isfile(os.path.join(root_working_dir, 'bright_calibrators.reg')):
            cmd_call("rsync -avP {} {}".format(region_file, os.path.join(root_working_dir, 'bright_calibrators.reg')))
        else:
            if do_kwargs['do_choose_calibrators'] > 0:
                raise IOError("Region file already found. Not copying provided one.")
        region_file = os.path.join(root_working_dir, 'bright_calibrators.reg')

    logger.info("Constructing run environments")
    if lofar_sksp_simg is not None:
        if not os.path.isfile(lofar_sksp_simg):
            logger.info(
                "Singularity image {} doesn't exist. Better have lofar tools sourced for ddf-pipeline work.".format(
                    lofar_sksp_simg))
            lofar_sksp_env = Env()
        else:
            if bind_dirs is None:
                bind_dirs = root_working_dir  # redundant placeholder
            lofar_sksp_env = SingularityEnv(lofar_sksp_simg, bind_dirs=bind_dirs)
    else:
        logger.info("Not using SKSP image, so lofar software better be sourced already that can do ddf pipeline work.")
        lofar_sksp_env = Env()

    if lofar_gain_screens_simg is not None:
        if not os.path.isfile(lofar_gain_screens_simg):
            logger.info(
                "Singularity image {} doesn't exist. Better have lofar tools sourced for screen imaging.".format(
                    lofar_gain_screens_simg))
            lofar_gain_screens_env = Env()
        else:
            if bind_dirs is None:
                bind_dirs = root_working_dir  # redundant placeholder
            lofar_gain_screens_env = SingularityEnv(lofar_gain_screens_simg, bind_dirs=bind_dirs)
    else:
        logger.info(
            "Not using lofar gain screens image, so lofar software better be sourced already that can image screens.")
        lofar_gain_screens_env = Env()

    if bayes_gain_screens_simg is not None:
        if not os.path.isfile(bayes_gain_screens_simg):
            logger.info(
                "Singularity image {} doesn't exist. Better have bayes gain screens sourced.".format(
                    bayes_gain_screens_simg))
            bayes_gain_screens_env = Env()
        else:
            if bind_dirs is None:
                bind_dirs = root_working_dir  # redundant placeholder
            bayes_gain_screens_env = SingularityEnv(bayes_gain_screens_simg, bind_dirs=bind_dirs)
    else:
        logger.info(
            "Not using bayes gain screen image, so bayes_gain_screens better be installed in conda env: {}".format(
                bayes_gain_screens_conda_env))
        bayes_gain_screens_env = CondaEnv(bayes_gain_screens_conda_env)

    step_list = [
        Step('download_archive', [], script_dir=script_dir, script_name='download_archive.py', exec_env=lofar_sksp_env),
        Step('choose_calibrators', ['choose_calibrators'], script_dir=script_dir, script_name='choose_calibrators.py',
             exec_env=lofar_sksp_env),
        Step('subtract', ['choose_calibrators', 'download_archive'], script_dir=script_dir,
             script_name='sub-sources-outside-region-mod.py', exec_env=lofar_sksp_env),
        Step('subtract_outside_pb', ['choose_calibrators', 'download_archive'], script_dir=script_dir,
             script_name='sub-sources-outside-pb.py', exec_env=lofar_sksp_env),
        Step('solve_dds4', ['subtract'], script_dir=script_dir, script_name='solve_on_subtracted.py',
             exec_env=lofar_sksp_env),
        Step('neural_gain_flagger', ['solve_dds4'], script_dir=script_dir, script_name='neural_gain_flagger.py',
             exec_env=bayes_gain_screens_env),
        Step('slow_solve_dds4', ['solve_dds4', 'tec_inference_and_smooth', 'infer_screen'], script_dir=script_dir,
             script_name='slow_solve_on_subtracted.py', exec_env=lofar_sksp_env),
        Step('tec_inference_and_smooth', ['solve_dds4','neural_gain_flagger'], script_dir=script_dir,
             script_name='tec_inference_and_smooth.py', exec_env=bayes_gain_screens_env),
        Step('infer_screen', ['tec_inference_and_smooth'], script_dir=script_dir,
             script_name='infer_screen.py',
             exec_env=bayes_gain_screens_env),
        Step('merge_slow', ['slow_solve_dds4', 'infer_screen', 'tec_inference_and_smooth'],
             script_dir=script_dir,
             script_name='merge_slow.py', exec_env=bayes_gain_screens_env),
        Step('flag_visibilities', ['infer_screen'],
             script_dir=script_dir,
             script_name='flag_visibilities.py', exec_env=lofar_gain_screens_env),
        Step('image_subtract_dirty', ['subtract'], script_dir=script_dir, script_name='image.py',
             exec_env=lofar_sksp_env),
        Step('image_subtract_dds4', ['tec_inference_and_smooth'], script_dir=script_dir,
             script_name='image.py', exec_env=lofar_sksp_env),
        Step('image_dds4', ['solve_dds4'], script_dir=script_dir, script_name='image.py',
             exec_env=lofar_sksp_env),
        Step('image_smooth', ['tec_inference_and_smooth'], script_dir=script_dir,
             script_name='image.py',
             exec_env=lofar_gain_screens_env),
        Step('image_smooth_slow', ['merge_slow'],
             script_dir=script_dir,
             script_name='image.py',
             exec_env=lofar_gain_screens_env),
        Step('image_screen', ['infer_screen'], script_dir=script_dir,
             script_name='image.py',
             exec_env=lofar_gain_screens_env),
        Step('image_screen_slow', ['merge_slow'],
             script_dir=script_dir,
             script_name='image.py',
             exec_env=lofar_gain_screens_env),
        Step('image_screen_slow_restricted',
             ['merge_slow', 'subtract_outside_pb', 'flag_visibilities'],
             script_dir=script_dir,
             script_name='image.py',
             exec_env=lofar_gain_screens_env),
        Step('image_smooth_slow_restricted',
             ['merge_slow', 'subtract_outside_pb', 'flag_visibilities'],
             script_dir=script_dir,
             script_name='image.py',
             exec_env=lofar_gain_screens_env)
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
                logger.info("User requested step do_{}={}".format(step.name, step.flag))
                break
        steps[step.name] = step

    pipeline = Pipeline(auto_resume, root_working_dir, state_file, timing_file, steps)

    data_dir = steps['download_archive'].working_dir
    logger.info(data_dir)

    steps['download_archive'] \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('archive_dir', archive_dir) \
        .add_cmd_arg('no_download', no_download)

    steps['choose_calibrators'] \
        .add_cmd_arg('region_file', region_file) \
        .add_cmd_arg('ref_image_fits', ref_image_fits) \
        .add_cmd_arg('flux_limit', 0.15) \
        .add_cmd_arg('min_spacing_arcmin', 6.) \
        .add_cmd_arg('max_N', 45)

    steps['subtract'] \
        .add_cmd_arg('region_file', region_file) \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('predict_column', 'PREDICT_SUB') \
        .add_cmd_arg('sub_column', 'DATA_SUB')

    steps['subtract_outside_pb'] \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('predict_column', 'PREDICT_SUB') \
        .add_cmd_arg('sub_column', 'DATA_RESTRICTED')

    steps['solve_dds4'] \
        .add_cmd_arg('region_file', region_file) \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir)

    steps['neural_gain_flagger'] \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('plot_results', True)

    steps['tec_inference_and_smooth'] \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('plot_results', True)

    steps['slow_solve_dds4'] \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir)

    steps['infer_screen'] \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('ref_image_fits', ref_image_fits) \
        .add_cmd_arg('max_N', 250) \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('plot_results', True)

    steps['merge_slow'] \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir)

    steps['flag_visibilities'] \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('new_weights_col', 'OUTLIER_FLAGS')

    steps['image_subtract_dirty'] \
        .add_cmd_arg('image_type', 'dirty:subtracted') \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('script_dir', script_dir)

    steps['image_dds4'] \
        .add_cmd_arg('image_type', 'dds4:data') \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('script_dir', script_dir) \
        .add_cmd_arg('use_init_dico', True)

    steps['image_subtract_dds4'] \
        .add_cmd_arg('image_type', 'dds4:subtracted') \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('script_dir', script_dir) \
        .add_cmd_arg('use_init_dico', True) \
        .add_cmd_arg('init_dico', os.path.join(data_dir, 'image_full_ampphase_di_m.NS.DATA_SUB.DicoModel'))

    steps['image_smooth'] \
        .add_cmd_arg('image_type', 'smoothed:data') \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('script_dir', script_dir) \
        .add_cmd_arg('use_init_dico', True)

    steps['image_smooth_slow'] \
        .add_cmd_arg('image_type', 'smoothed_slow:data:outliers_flagged') \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('script_dir', script_dir) \
        .add_cmd_arg('use_init_dico', True)

    steps['image_screen'] \
        .add_cmd_arg('image_type', 'screen:restricted') \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('script_dir', script_dir) \
        .add_cmd_arg('use_init_dico', True)

    steps['image_screen_slow'] \
        .add_cmd_arg('image_type', 'screen_slow:data:outliers_flagged') \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('script_dir', script_dir) \
        .add_cmd_arg('use_init_dico', True)

    steps['image_smooth_slow_restricted'] \
        .add_cmd_arg('image_type', 'smoothed_slow:restricted:outliers_flagged') \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('script_dir', script_dir) \
        .add_cmd_arg('use_init_dico', True) \
        .add_cmd_arg('init_dico', os.path.join(data_dir,
                                               'image_full_ampphase_di_m.NS.DATA_RESTRICTED.DicoModel'))

    steps['image_screen_slow_restricted'] \
        .add_cmd_arg('image_type', 'screen_slow:restricted:outliers_flagged') \
        .add_cmd_arg('ncpu', ncpu) \
        .add_cmd_arg('obs_num', obs_num) \
        .add_cmd_arg('data_dir', data_dir) \
        .add_cmd_arg('script_dir', script_dir) \
        .add_cmd_arg('use_init_dico', True) \
        .add_cmd_arg('init_dico', os.path.join(data_dir,
                                               'image_full_ampphase_di_m.NS.DATA_RESTRICTED.DicoModel'))

    pipeline.build()
    pipeline.run(retry_task_on_fail=retry_task_on_fail)


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

    optional.add_argument('--retry_task_on_fail',
                          help='Retry on failed task this many times. Useful for non-determininistic bugs. 0 means off.',
                          default=0, type=int, required=False)
    optional.add_argument('--no_download',
                          help='Whether to move instead of copy the archive dir. Potentially unsafe. Requires env variable set SP_AUTH="1".',
                          default=False, type="bool", required=False)
    optional.add_argument('--region_file',
                          help='ds9 region file defining calbrators. If not provided, they will be automatically determined.',
                          required=False, type='str_or_none',
                          default=None)
    optional.add_argument('--ref_image_fits',
                          help='Reference image used to extract screen directions and auto select calibrators if region_file is None. If not provided, it will use the one in the archive directory.',
                          required=False, default=None, type='str_or_none')
    optional.add_argument('--auto_resume',
                          help='Int flag indicating whether or try to automatically resume operations based on the STATE file. Flags (-1) 0/1/2 with usual meaning (see do_*). If negative then forces resumes. Otherwise assert that previous run finished successfully first.',
                          required=False, default=2, type=int)
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
    optional.add_argument('--script_dir',
                          help='Where the scripts are located, by default uses those installed with package.',
                          default=None, type='str_or_none', required=False)

    required.add_argument('--obs_num', help='Obs number L*',
                          default=None, type=int, required=True)
    required.add_argument('--archive_dir',
                          help='Where are the archives stored. Can also be networked locations, e.g. <user>@<host>:<path> but you must have authentication.',
                          default=None, type=str, required=True)
    required.add_argument('--root_working_dir', help='Where the root of all working dirs are.',
                          default=None, type=str, required=True)

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
                          default='bayes_gain_screens_py', type=str, required=False)

    for s in STEPS:
        step_args.add_argument('--do_{}'.format(s),
                               help='Do {}? (NO=0/YES_CLOBBER=1/YES_NO_CLOBBER=2)'.format(s),
                               default=0, type=int, required=False)


def debug_main():
    main(archive_dir='/home/albert/store/lockman/archive',
         root_working_dir='/home/albert/store/root',
         script_dir=None,
         obs_num=664480, #342938,
         region_file=None,
         ncpu=32,
         ref_image_fits=None,
         no_download=False,
         bind_dirs='/beegfs/lofar',
         lofar_sksp_simg='/home/albert/store/lofar_sksp_ddf.simg',
         lofar_gain_screens_simg='/home/albert/store/lofar_sksp_ddf_gainscreens_premerge.simg',
         bayes_gain_screens_simg=None,
         bayes_gain_screens_conda_env='bayes_gain_screens_py',
         auto_resume=0,
         do_choose_calibrators=0,
         do_download_archive=0,
         do_subtract=0,
         do_subtract_outside_pb=0,
         do_image_subtract_dirty=0,
         do_solve_dds4=0,
         do_neural_gain_flagger=0,
         do_tec_inference_and_smooth=1,
         do_slow_solve_dds4=1,
         do_merge_slow=1,
         do_flag_visibilities=1,
         do_infer_screen=1,
         do_image_dds4=0,
         do_image_subtract_dds4=0,
         do_image_smooth=0,
         do_image_smooth_slow=0,
         do_image_screen=0,
         do_image_screen_slow=0,
         do_image_smooth_slow_restricted=1,
         do_image_screen_slow_restricted=1,
         retry_task_on_fail=0)


STEPS = [
    "download_archive",
    "choose_calibrators",
    "subtract",
    "subtract_outside_pb",
    "solve_dds4",
    "neural_gain_flagger",
    "slow_solve_dds4",
    "tec_inference_and_smooth",
    "infer_screen",
    "merge_slow",
    "flag_visibilities",
    "image_subtract_dirty",
    "image_subtract_dds4",
    "image_dds4",
    "image_smooth",
    "image_smooth_slow",
    "image_smooth_slow_restricted",
    "image_screen",
    "image_screen_slow",
    "image_screen_slow_restricted"]

if __name__ == '__main__':
    if len(sys.argv) == 1:
        debug_main()
        exit(0)
    parser = argparse.ArgumentParser(
        description='Runs full pipeline on a single obs_num from archive.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    logger.info("Running with:")
    for option, value in vars(flags).items():
        logger.info("    {} -> {}".format(option, value))

    main(**vars(flags))
