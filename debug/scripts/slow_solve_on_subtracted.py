import os
import glob
import pyregion
import numpy as np
import sys
import tables
import argparse

def prepare_kms_sols(data_dir, obs_num, sol_name):
    merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, sol_name))
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.sols.npz'.format(obs_num, sol_name))
    smooth_merged_sol = os.path.join(data_dir, 'L{}_{}_merged_smooth.sols.npz'.format(obs_num, sol_name))
    with tables.open_file(merged_h5parm) as t:
        #Nt, Nf, Na, Nd, Npol
        phase = t.root.smoothed000.phase000.val[:].T
        amp = t.root.smoothed000.amplitude000.val[:].T
    kms = np.load(merged_sol)
    if phase[:, :, :, :, 0].shape != kms['Sols']['G'][:, :, :, :, 0, 0].shape:
        raise ValueError("Shapes are not correct in kms solutions {} {}".format(kms['Sols']['G'].shape, phase.shape))

    kms['Sols']['G'][:, :, :, :, 0, 0] = amp[:, :, :, :, 0] * np.cos(phase[:, :, :, :, 0]) + 1j * amp[:, :, :, :, 0] * np.sin(
        phase[:, :, :, :, 0])  # XX
    kms['Sols']['G'][:, :, :, :, 1, 1] = amp[:, :, :, :, 0] * np.cos(phase[:, :, :, :, 0]) + 1j * amp[:, :, :, :, 0] * np.sin(
        phase[:, :, :, :, 3])  # YY

    np.savez(smooth_merged_sol, ModelName=kms['ModelName'], MaskedSols=kms['MaskedSols'], \
             FreqDomains=kms['FreqDomains'], StationNames=kms['StationNames'], BeamTimes=kms['BeamTimes'], \
             SourceCatSub=kms['SourceCatSub'], ClusterCat=kms['ClusterCat'], MSName=kms['MSName'], Sols=kms['Sols'], \
             SkyModel=kms['SkyModel'])



def solve(masked_dico_model, obs_num, clustercat, working_dir, data_dir, ncpu, sol_name):
    ddfcache = glob.glob(os.path.join(working_dir, '*.ddfcache'))
    for d in ddfcache:
        os.unlink(d)

    mslist = sorted(glob.glob(os.path.join(data_dir,'L{}*.ms'.format(obs_num))))
    if len(mslist) == 0:
        raise IOError("MS list  empty")

    if not os.path.isfile(clustercat):
        raise IOError("Clustercat doesn't exist {}".format(clustercat))

    solsdir = os.path.join(data_dir, 'SOLSDIR')

    for ms in mslist:
        cmd = 'kMS.py --MSName {ms} --SolverType KAFCA '.format(ms=ms)
        cmd = cmd + '--PolMode Scalar --BaseImageName image_full_ampphase_di_m.NS '
        cmd = cmd + '--dt 43.630000 --NIterKF 6 --CovQ 0.1 --LambdaKF=0.5 --NCPU {ncpu}'.format(ncpu=ncpu)
        cmd = cmd + '--OutSolsName {out_sols} --NChanSols 1 --PowerSmooth=0.0 '.format(out_sols=sol_name)
        cmd = cmd + '--InCol DATA_SUB --Weighting Natural --UVMinMax=0.500000,5000.000000 '
        cmd = cmd + '--SolsDir={solsdir}  --BeamMode LOFAR --LOFARBeamMode=A --DDFCacheDir=. '.format(solsdir=solsdir)
        cmd = cmd + '--NodesFile {clustercat} --DicoModel {masked_dico_model}'.format(clustercat=clustercat, masked_dico_model=masked_dico_model)

        cmd = cmd.replace(' --', ' \\n--')
        with open(os.path.join(working_dir, 'instruct.sh'), 'w') as f:
            f.write(cmd)
        print(cmd)
        os.system(cmd)

def make_merged_h5parm(obs_num, sol_name, data_dir, working_dir):
    solsdir = os.path.join(data_dir, 'SOLSDIR')
    merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, sol_name))
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
        cmd = 'killMS2H5parm.py --nofulljones {h5_file} {npz_file} '.format(npz_file=s,
                                                                            h5_file=s.replace('.npz', '.h5'))
        h5files.append(s.replace('.npz', '.h5'))
        print(cmd)
        os.system(cmd)

    cmd = 'H5parm_collector.py --outh5parm={merged_h5parm} [{h5files}]'.format(merged_h5parm=merged_h5parm,
                                                                               h5files=','.join(h5files))
    print(cmd)
    os.system(cmd)

    solsfile = os.path.join(working_dir, 'solslist.txt')
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.sols.npz'.format(obs_num, sol_name))
    with open(solsfile, 'w') as f:
        for s in sols:
            f.write("{}\n".format(s))
    os.system('MergeSols.py --SolsFilesIn={} --SolFileOut={}'.format(solsfile, merged_sol))


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--ncpu', help='Number of cpu to use, default=34', default=32, type=int)
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)

def main(obs_num, data_dir, working_dir, ncpu):
    clustercat = os.path.join(data_dir, 'subtract.ClusterCat.npy')
    masked_dico_model = os.path.join(data_dir, 'image_full_ampphase_di_m.NS.masked.DicoModel')
    os.chdir(working_dir)
    prepare_kms_sols(data_dir, obs_num, 'DSS4_full')
    solve(masked_dico_model,obs_num, clustercat, working_dir,data_dir, ncpu,'DSS4_full_slow')
    make_merged_h5parm(obs_num, 'DSS4_full_slow', data_dir, working_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep soures inside box region, subtract everything else and create new ms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))