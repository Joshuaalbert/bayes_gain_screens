import pyrap.tables as pt
import glob
import numpy as np
import os

def main(obs_num=667218):
    mslist = glob.glob('./L{}*.ms'.format(obs_num))
    if len(mslist) == 0:
        raise ValueError("mslist empty")

    sim_column = 'SIM_DATA'
    pred_column = 'DD_PREDICT'
    for ms in mslist:
        with pt.table(ms, readonly=False, ack=True) as t:
            print('Working on {}'.format(ms))
            if pred_column not in t.colnames():
                continue
            if sim_column not in t.colnames():
                print("Making SIM_DATA")
                desc = t.getcoldesc(pred_column)
                newdesc = pt.makecoldesc(sim_column, desc)
                newdmi = t.getdminfo(pred_column)
                newdmi['NAME'] = 'Dysco' + sim_column
                t.addcols(newdesc, newdmi)
                print("Adding perturbation to {}".format(pred_column))
                for row in range(0, t.nrows(), 3000000):
                    sim_data = t.getcol(pred_column, startrow=row, nrow=3000000, rowincr=1)
                    sim_data += 2.*(np.random.normal(size=sim_data.shape) + 1j*np.random.normal(size=sim_data.shape))
                    t.putcol(sim_column, sim_data, startrow=row, nrow=3000000, rowincr=1)
    print("Solving")
    kMS(dicomodel = "image_full_ampphase_di_m.NS.DicoModel", data_column = 'SIM_DATA', obs_num=obs_num, clustercat=None)

                
def kMS(dicomodel = "image_full_ampphase_di_m.NS.DicoModel", data_column = 'SIM_DATA', obs_num=None, clustercat=None):
    
    os.system('rm -rf *.ddfcache')

    #mslist = sorted(glob.glob('cP126+65BEAM_1_chan*-*.ms'))
    if obs_num is None:
        mslist = sorted(glob.glob('L*.ms'))
    else:
        mslist = sorted(glob.glob('L{}*.ms'.format(obs_num)))
    if len(mslist)==0:
        raise ValueError("MS list  empty")

    if clustercat is None:
        clustercat = glob.glob("*ClusterCat.npy")
        if len(clustercat)==0:
            raise ValueError("No Cluster Cat passed or found")
        clustercat = clustercat[0]
    print("Using cluster cat {}".format(clustercat))

    outsolsname = 'sim_data_smooth_slow_full'

    for ms in mslist:

      cmd = 'kMS.py --MSName '+ ms + ' ' + '--SolverType KAFCA '
      cmd = cmd + '--PolMode Scalar --BaseImageName image_full_ampphase_di_m.NS ' 
      cmd = cmd + '--dt 0.5 --NIterKF 6 --CovQ 0.1 --LambdaKF=0.5 --NCPU 32 ' 
      cmd = cmd + '--OutSolsName ' + outsolsname + ' --NChanSols 1 --PowerSmooth=0.0 ' 
      cmd = cmd + '--InCol ' + data_column + ' --Weighting Natural --UVMinMax=0.100000,5000.000000 ' 
      cmd = cmd + '--SolsDir=SOLSDIR  --BeamMode LOFAR --LOFARBeamMode=A --DDFCacheDir=. ' 
      cmd = cmd + '--NodesFile ' + clustercat + ' --DicoModel ' + dicomodel
    
      print cmd
      os.system(cmd)


if __name__ == '__main__':
    main()
