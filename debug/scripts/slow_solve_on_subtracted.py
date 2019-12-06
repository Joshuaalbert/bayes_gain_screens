import os
import glob
import numpy as np
import tables
import argparse

def prepare_kms_sols(data_dir, obs_num):
    sol_name = 'DDS4_full'
    merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, sol_name))
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.sols.npz'.format(obs_num, sol_name))
    smooth_merged_sol = os.path.join(data_dir, 'L{}_{}_smoothed_merged.sols.npz'.format(obs_num, sol_name))
    with tables.open_file(merged_h5parm) as t:
        #Nt, Nf, Na, Nd, Npol
        phase = t.root.smoothed000.phase000.val[...].T
        amp = t.root.smoothed000.amplitude000.val[...].T
    kms = np.load(merged_sol)
    if phase[:, :, :, :, 0].shape != kms['Sols']['G'][:, :, :, :, 0, 0].shape:
        raise ValueError("Shapes are not correct in kms solutions {} {}".format(kms['Sols']['G'].shape, phase.shape))

    kms['Sols']['G'][:, :, :, :, 0, 0] = amp[:, :, :, :, 0] * np.cos(phase[:, :, :, :, 0]) + \
                                         1j * amp[:, :, :, :, 0] * np.sin(phase[:, :, :, :, 0])  # XX
    kms['Sols']['G'][:, :, :, :, 1, 1] = amp[:, :, :, :, 0] * np.cos(phase[:, :, :, :, 0]) + \
                                         1j * amp[:, :, :, :, 0] * np.sin(phase[:, :, :, :, 0])  # YY

    np.savez(smooth_merged_sol, ModelName=kms['ModelName'], MaskedSols=kms['MaskedSols'],
             FreqDomains=kms['FreqDomains'], StationNames=kms['StationNames'], BeamTimes=kms['BeamTimes'],
             SourceCatSub=kms['SourceCatSub'], ClusterCat=kms['ClusterCat'], MSName=kms['MSName'], Sols=kms['Sols'],
             SkyModel=kms['SkyModel'])

def solve(masked_dico_model, obs_num, clustercat, working_dir, data_dir, ncpu):
    sol_name='DDS4_full'
    pre_apply_sol_name='{}_smoothed'.format(sol_name)
    out_sol_name='{}_slow'.format(sol_name)

    mslist = sorted(glob.glob(os.path.join(data_dir,'L{}*.ms'.format(obs_num))))
    if len(mslist) == 0:
        raise IOError("MS list  empty")

    if not os.path.isfile(clustercat):
        raise IOError("Clustercat doesn't exist {}".format(clustercat))

    solsdir = os.path.join(data_dir, 'SOLSDIR')

    for ms in mslist:
        cmd = ['kMS.py',
               '--MSName={ms}'.format(ms=ms),
               '--SolverType=KAFCA',
               '--PolMode=Scalar',
               '--BaseImageName=image_full_ampphase_di_m.NS',
               '--dt=43.630000',
               '--NIterKF=6',
               '--CovQ=0.1',
               '--LambdaKF=0.5',
               '--NCPU={ncpu}'.format(ncpu=ncpu),
               '--OutSolsName={out_sol_name}'.format(out_sol_name=out_sol_name),
               '--NChanSols=1',
               '--PowerSmooth=0.0',
               '--PreApplySols=[{pre_apply_sol_name}]'.format(pre_apply_sol_name=pre_apply_sol_name),
               '--InCol=DATA_SUB',
               '--Weighting=Natural',
               '--UVMinMax=0.500000,5000.000000',
               '--SolsDir={solsdir}'.format(solsdir=solsdir),
               '--BeamMode=LOFAR',
               '--LOFARBeamMode=A',
               '--DDFCacheDir=.',
               '--NodesFile={clustercat}'.format(clustercat=clustercat),
               '--DicoModel={masked_dico_model}'.format(masked_dico_model=masked_dico_model)]

        cmd = ' \\\n\t'.join(cmd)

        with open(os.path.join(working_dir, 'instruct.sh'), 'w') as f:
            f.write(cmd)
        print(cmd)
        os.system(cmd)

def make_merged_h5parm(obs_num, data_dir, working_dir):
    slow_sol = 'DDS4_full_slow'
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.sols.npz'.format(obs_num, slow_sol))
    merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, slow_sol))
    solsdir = os.path.join(data_dir, 'SOLSDIR')
    sol_folders = sorted(glob.glob(os.path.join(solsdir, "L{}*.ms".format(obs_num))))
    if len(sol_folders) == 0:
        raise ValueError("Invalid obs num {}".format(obs_num))
    sols = []
    for f in sol_folders:
        sol = glob.glob(os.path.join(f, '*{}.sols.npz'.format(slow_sol)))
        if len(sol) == 0:
            print("Can't find {} in {}".format(slow_sol, f))
            continue
        sols.append(os.path.abspath(sol[0]))
    solsfile = os.path.join(working_dir, 'solslist_dds4_slow.txt')
    with open(solsfile, 'w') as f:
        for s in sols:
            f.write("{}\n".format(s))
    os.system('MergeSols.py --SolsFilesIn={} --SolFileOut={}'.format(solsfile, merged_sol))
    os.system('killMS2H5parm.py --nofulljones {h5_file} {npz_file} '.format(npz_file=merged_sol,
                                                                            h5_file=merged_h5parm))

def make_symlinks(data_dir, obs_num):
    print("Creating symbolic links")
    smooth_merged_sol = os.path.join(data_dir, 'L{}_DDS4_full_smoothed_merged.sols.npz'.format(obs_num))
    solsdir = os.path.join(data_dir, 'SOLSDIR')
    sol_folders = glob.glob(os.path.join(solsdir, 'L{obs_num}*.ms'.format(obs_num=obs_num)))
    for f in sol_folders:
        src = smooth_merged_sol
        dst = os.path.join(f, 'killMS.DDS4_full_smoothed.sols.npz')
        if os.path.islink(dst):
            os.unlink(dst)
        print("Linking {} -> {}".format(src,dst))
        os.symlink(src,dst)

def main(obs_num, data_dir, working_dir, ncpu):
    clustercat = os.path.join(data_dir, 'subtract.ClusterCat.npy')
    masked_dico_model = os.path.join(data_dir, 'image_full_ampphase_di_m.NS.masked.DicoModel')
    os.chdir(working_dir)
    prepare_kms_sols(data_dir, obs_num)
    make_symlinks(data_dir, obs_num)
    solve(masked_dico_model,obs_num, clustercat, working_dir,data_dir, ncpu)
    make_merged_h5parm(obs_num, data_dir, working_dir)

def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--ncpu', help='Number of cpu to use', default=32, type=int, required=True)
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Solve slow AP solutions on 43 minutes timescale to solve holes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("\t{} -> {}".format(option, value))
    main(**vars(flags))



