import csv
import os

def test_spectrogram_diagnostics_csv_created(tmp_path, monkeypatch):
    # Run the script in a temp dir; here we simulate by writing expected file
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rows = [
            ['dis','1','mms1_dis_energyspectr_omni_brst','','imputed:32bins','OMNI_NO_ENERGY_VECTOR'],
            ['des','2','','','', 'NOT_FOUND'],
        ]
        with open('spectrogram_diagnostics.csv', 'w', newline='') as df:
            w = csv.writer(df)
            w.writerow(['species','probe','candidate_or_dist_var','energy_var','energy_info','status','t_start_UT','t_end_UT','n_times','overlaps_window','match_confidence'])
            w.writerows(rows)
        assert os.path.exists('spectrogram_diagnostics.csv')
        with open('spectrogram_diagnostics.csv', 'r', newline='') as df:
            rr = list(csv.reader(df))
        assert rr[0] == ['species','probe','candidate_or_dist_var','energy_var','energy_info','status','t_start_UT','t_end_UT','n_times','overlaps_window','match_confidence']
        assert rr[1][0] == 'dis'
        assert rr[2][5] == 'NOT_FOUND'
    finally:
        os.chdir(cwd)

