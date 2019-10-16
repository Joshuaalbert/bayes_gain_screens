import os
import sys
import glob
import argparse


def main(obs_num, solsdir, sol_name, suffix):
#    data_dir = os.path.abspath(data_dir)
#    solsdir = os.path.join(data_dir, 'SOLSDIR')
    sol_folders = sorted(glob.glob(os.path.join(solsdir, "L{}*.ms".format(obs_num))))
    if len(sol_folders) == 0:
        raise ValueError("Invalid obs num {}".format(obs_num))
    sols = []
    for f in sol_folders:
        sol = glob.glob(os.path.join(f, '*{}.sols.npz'.format(sol_name)))
        if len(sol) == 0:
            print("Can't find {} in {}".format(sol_name, f))
            continue
        sols.append(os.path.abspath(sol[0]))
    h5files = []
    for s in sols:
        cmd = 'killMS2H5parm.py --nofulljones {h5_file} {npz_file} '.format(npz_file=s, h5_file=s.replace('.npz','.h5'))
        h5files.append(s.replace('.npz','.h5'))
        print(cmd)
        os.system(cmd)
    if suffix is not None:
        outh5parm= 'L{}_{}_{}.h5'.format(obs_num, sol_name, suffix)
    else:
        outh5parm= 'L{}_{}_merged.h5'.format(obs_num, sol_name)

    cmd = 'H5parm_collector.py --outh5parm={outh5parm} [{h5files}]'.format(outh5parm=outh5parm, h5files=','.join(h5files))
    print(cmd)
    os.system(cmd)

def add_args(parser):
    def bool_parse(s):
        return s.strip().lower() == 'true'
    parser.register('type', 'bool', bool_parse)
    def dir_parse(s):
        s = os.path.abspath(s)
        if not os.path.isdir(s):
            raise ValueError("{} doesn't exist".format(s))
        return s

    parser.register('type', 'bool', bool_parse)
    parser.register('type', 'dir', dir_parse)

    optional = parser._action_groups.pop()
    parser._action_groups.append(optional)
    required = parser.add_argument_group('Required arguments')
    
    required.add_argument('--obs_num', type=int, help="""The observation number.""", required=True)
    required.add_argument('--solsdir', type='dir', help="""The SOLSDIR directory.""", required=True)
    required.add_argument('--sol_name', type=str, help="""The solution name.""", required=True)
    optional.add_argument('--suffix', type=str, default='merged', help="""Optional suffix of output (before .h5).""", required=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("\t{} -> {}".format(option, value))
    main(**vars(flags))
#    main(obs_num, '/home/albert/store/lockman/data_gain_analysis', 'DDS3_full_smoothed')
