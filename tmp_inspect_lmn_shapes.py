from tools.idl_sav_import import load_idl_sav
import numpy as np

sav = load_idl_sav('mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav')
LMN = sav['lmn']
B_LMN = sav['b_lmn']

for probe in ['1', '2', '3', '4']:
    print('Probe', probe)
    L = np.asarray(LMN[probe]['L'])
    M = np.asarray(LMN[probe]['M'])
    N = np.asarray(LMN[probe]['N'])
    b = np.asarray(B_LMN[probe]['blmn'])
    print('  L shape', L.shape)
    print('  M shape', M.shape)
    print('  N shape', N.shape)
    print('  blmn shape', b.shape)

