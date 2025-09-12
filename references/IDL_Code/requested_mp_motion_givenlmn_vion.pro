pro Requested_MP_Motion_GivenLMN_Vion
;created October 2024
;first event for Sarah vines, String of pearl MMS event on jan. 25th, 2019 ~0400UT 
;given .sav structure wtih time range, LMN and Vion in GSM

restore, '~kllera/Downloads/mp_lmn_systems_20190127_1215-1255_mp-ver2.sav'  ;event presented at MMS 2023 firstlook and MMS2025 
;restore, '~kllera/Downloads/mp_lmn_system_20190125_0400-0430.sav'  ;SVK event
;restore, '~kllera/Downloads/mp_lmn_system_20190125_0400-0430.sav',/verbose
  ;% RESTORE: Save file written by svines@svines-ml, Tue Oct 29 16:31:54 2024.
  ;% RESTORE: IDL version 9.0.0 (darwin, arm64).
  ;% RESTORE: Restored variable: TRANGE_FULL.         Trange of event
  ;% RESTORE: Restored variable: TRANGE_FULL_STR.
  ;% RESTORE: Restored variable: TRANGE_LMN.          Trange for obtained LMN
  ;% RESTORE: Restored variable: TRANGE_LMN_STR.
  ;% RESTORE: Restored variable: LHAT.                unit vectors from MVAB method (B in GSM input)
  ;% RESTORE: Restored variable: MHAT.
  ;% RESTORE: Restored variable: NHAT.
  ;% RESTORE: Restored variable: VI_LMN1.             structure FPI's ion velocity in LMN per MMS s/c  .x is UT time "in Unix time" 
  ;% RESTORE: Restored variable: VI_LMN2.                                                              .y[time intg, (VL, VM,VN)]
  ;% RESTORE: Restored variable: VI_LMN3.
  ;% RESTORE: Restored variable: VI_LMN4.
;TRANGE_FULL,TRANGE_FULL_STR,TRANGE_LMN,TRANGE_LMN_STR,LHAT,MHAT,NHAT,VI_LMN1,VI_LMN2,VI_LMN3,VI_LMN4
;
;in jullian date  $julianDay = $unixTimeStamp / 86400. + 2440587.5 + centering timestep (half of 30 ms or 7.5 ms if doing)
jul_trange_full=trange_full/ 86400. + 2440587.5
jul_Vi1=Vi_lmn1.x[*]/ 86400. + 2440587.5 
jul_Vi2=Vi_lmn2.x[*]/ 86400. + 2440587.5 
jul_Vi3=Vi_lmn3.x[*]/ 86400. + 2440587.5 
jul_Vi4=Vi_lmn4.x[*]/ 86400. + 2440587.5 


Vn_min= -100;-450 ;km/s y_min for ploting
Vn_max= 160;450

!x.style=1
!y.style=1
!p.font=12
!P.charsize= 1           ;Size of letters, etc
!x.margin=[14,7]
options, '*', 'thick', 3 ; plot line thickness +1 to 3
time_stamp, /off





dt= ceil((jul_trange_full[1] - jul_trange_full[0])/60.)
timespan, jul_trange_full[0], dt,/minute
!X.range=jul_trange_full
dummy = LABEL_DATE(DATE_FORMAT=['%H:%I!C%D/%N/%Y'], offset =0)
!X.tickformat = 'label_date'

;check range looks right, data read properly
plot,jul_Vi1,Vi_lmn1.y[*,2], xtickunit='minute', xtickinterval=5

;stop


!X.tickformat = '' ;to clear the previous label-date/Date_format that will not work with unix time
;Spedas and tplots use unix time
dt= ceil((time_double(trange_full[1] - trange_full[0]))/60.)
timespan, trange_full[0], dt,/minute

yzero = replicate(0, n_elements(trange_full))
store_data,'y0_line', data=[{x:trange_full, y:yzero}, {x:trange_full, y:yzero},{x:trange_full, y:yzero},{x:trange_full, y:yzero}] ;gives all MMS labels
ylim, 'y0_line', Vn_min, Vn_max 
options, 'y0_line', 'thick', 1
options, 'y0_line', 'ytitle', 'V!diN!N!c!c[km/s]'
options, 'y0_line', labels=['mms1','mms2','mms3','mms4']
options,'y0_line',colors=[0,6,4,2]
options, 'y0_line','labflag',-1

store_data, 'mms1_ViN', data={x:Vi_lmn1.x, y:Vi_lmn1.y[*,2]}
;store_data, sc_id+'Vi', data={x:jul_Vi1, y:Vi_lmn1.y} ; time doesn't match tplots
store_data, 'mms2_ViN', data={x:Vi_lmn2.x, y:Vi_lmn2.y[*,2]}
store_data, 'mms3_ViN', data={x:Vi_lmn3.x, y:Vi_lmn3.y[*,2]}
store_data, 'mms4_ViN', data={x:Vi_lmn4.x, y:Vi_lmn4.y[*,2]} 

options, ['mms1_ViN','mms2_ViN','mms3_ViN','mms4_ViN'], overplot=['y0_line'], /noerase
ylim, ['mms1_ViN','mms2_ViN','mms3_ViN','mms4_ViN'], Vn_min, Vn_max ;these zeros gives auto scale;

store_data, 'VIN_multi', data = ['mms1_ViN','mms2_ViN','mms3_ViN','mms4_ViN']
options, 'VIN_multi', 'ytitle', 'V!diN!N!c!c[km/s]'
options, 'VIN_multi', labels=['mms1','mms2','mms3','mms4']
options,'VIN_multi',colors=[0,6,4,2]
options, 'VIN_multi','labflag',-1
options, 'VIN_multi','ysubtitle',''
options,'mms1_ViN',colors=0
options,'mms2_ViN',colors=6
options,'mms3_ViN',colors=4
options,'mms4_ViN',colors=2

;=============setup
output_folder='/users/kllera/MMS_plots/Requested_MP_Motion/'
allMMS4e= 0;1   ;0 once MMS 4 electron went down (Skip L2 mms4 e; get QL level)
high_res=0    ; fast/survy 30 ms is 0; for 7.5 ms burst data make 1


;Load FPI to confirm boundary timing
IF allMMS4e then begin
  if high_res eq 0 then begin
    mms_load_fpi, probes=[1,2,3,4],datatype=['dis-moms','des-moms'], level='l2',data_rate='fast', varformat=['*energyspectr_omni*'], /time_clip ; can add other feilds; '*bulkv*'
    ;mms_load_fpi, probes=[1,2,3,4],datatype='dis-moms', level='l2',data_rate='fast', varformat=['*energyspectr_omni*'], /time_clip
    
  endif
  if high_res eq 1 then begin
    mms_load_fpi,probes=[1,2,3,4], datatype='des-qmoms', level='l2', data_rate='brst', /time_clip 
    mms_load_fpi,probes=[1,2,3,4], datatype='dis-qmoms', level='l2', data_rate='brst', /time_clip
    
  endif
endIF else begin
  if high_res eq 0 then begin ;No MMS4 elec, switch elec to QL
    mms_load_fpi, probes=[1,2,3],datatype='des-moms', level='l2',data_rate='fast', varformat=['*energyspectr_omni*'], /time_clip 
    mms_load_fpi, probes=[4],datatype='des', level='ql',data_rate='fast', varformat=['*energyspectr_omni*'], /time_clip
    mms_load_fpi, probes=[1,2,3,4],datatype='dis-moms', level='l2',data_rate='fast', varformat=['*energyspectr_omni*'], /time_clip
    
  endif
  if high_res eq 1 then begin ;No MMS4 elec, skipping
    mms_load_fpi,probes=[1,2,3], datatype='des-qmoms', level='l2', data_rate='brst', /time_clip  
     mms_load_fpi, probes=[4],datatype='des', level='ql',data_rate='brst', varformat=['*energyspectr_omni*'], /time_clip 
    mms_load_fpi,probes=[1,2,3,4], datatype='dis-qmoms', level='l2', data_rate='brst', /time_clip
  endif
endelse

;fpi version 3 (for version 2, load names differ; e.g. energyspectr_omni_avg not omni_fast)
ylim,'*_dis_energyspectr_omni_fast',10,30000,1
   for ii=1, 4 do options,'mms'+string(ii,f='(I0)')+'_dis_energyspectr_omni_fast', 'ytitle', 'Ion!CMMS'+string(ii,f='(I0)') ;+'!C [eV]'
ylim,'*_des_energyspectr_omni_fast',10,30000,1
   for ii=1, 4 do options,'mms'+string(ii,f='(I0)')+'_des_energyspectr_omni_fast', 'ytitle', 'Elec!CMMS'+string(ii,f='(I0)'); +'!C [eV]'
   
   
   
   mms_load_fgm, probes=[1,2,3,4], level='l2', data_rate = 'srvy', varformat='*gse*', /time_clip ;srvy is the fast in mag
   
   for sc = 1,4 do begin
     str=string(sc)
     sc_id='mms'+string(str,format='(I1)')
     ;split the magnetic fields up into vector components
     split_vec, sc_id+'_fgm_b_gse_srvy_l2'                             ;srvy is the fast in mag
     copy_data, sc_id+'_fgm_b_gse_fast_l2_0', sc_id+'_Bx'
     copy_data, sc_id+'_fgm_b_gse_fast_l2_1', sc_id+'_By'
     copy_data, sc_id+'_fgm_b_gse_fast_l2_2', sc_id+'_Bz'
   endfor
     
     ;Store the corresponding GSE components from the 4 craft into one variable
     store_data, 'Bx_multi', data = ['mms1_Bx','mms2_Bx','mms3_Bx','mms4_Bx']
     options,'Bx_multi',colors=[0,6,4,2]
     options, 'Bx_multi', 'ytitle', 'B!dX!N!c!c[nT]'
     options, 'Bx_multi', labels=['mms1','mms2','mms3','mms4']
     options, 'Bx_multi','labflag',-1
     options, 'Bx_multi','ysubtitle',''

     store_data, 'By_multi', data = ['mms1_By','mms2_By','mms3_By','mms4_By']
     options,'By_multi',colors=[0,6,4,2]
     options, 'By_multi', 'ytitle', 'B!dY!N!c!c[nT]'
     options, 'By_multi', labels=['mms1','mms2','mms3','mms4']
     options, 'By_multi','labflag',-1
     options, 'By_multi','ysubtitle',''

     store_data, 'Bz_multi', data = ['mms1_Bz','mms2_Bz','mms3_Bz','mms4_Bz']
     options,'Bz_multi',colors=[0,6,4,2]
     options, 'Bz_multi', 'ytitle', 'B!dZ!N!c!c[nT]'
     options, 'Bz_multi', labels=['mms1','mms2','mms3','mms4']
     options, 'Bz_multi','labflag',-1
     options, 'Bz_multi','ysubtitle',''

     ;Rotate into LMN
     ;====================================================================
     evec =fltarr(3,3)
     
     ;Guy's LMN at 12:21UT  MVAB
     evec[*,0]= [-0.16832965,0.92102875,0.35124232]
     evec[*,1]= [0.26106316,-0.30194845,0.9168823]
     evec[*,2]= [0.95053204,0.24603491,-0.1896198]

       for sc=1,4 do begin
         str=string(sc)
         sc_id='mms'+string(str,format='(I1)')

         ;Rotate B-field from GSE to LMN
         get_data, sc_id+'_fgm_b_gse_srvy_l2_bvec', data=bdata ;srvy  and "bvec" is the fast here
         blmn = fltarr(n_elements(bdata.x),3)
         for i=0, n_elements(bdata.x)-1 do blmn(i,*) = evec##bdata.y(i,0:2)
         store_data, sc_id+'_B_LMN', data={x:bdata.x, y:blmn}
         split_vec, sc_id+'_B_LMN'
         copy_data, sc_id+'_B_LMN_x', sc_id+'_BL'
         copy_data, sc_id+'_B_LMN_y', sc_id+'_BM'
         copy_data, sc_id+'_B_LMN_z', sc_id+'_BN'
         options, sc_id+'_BL', colors=2
         options, sc_id+'_BM', colors=4
         options, sc_id+'_BN', colors=6
         
         endfor
         ;Store LMN values into their new tplot variables
         store_data, 'BL_multi', data = ['mms1_BL','mms2_BL','mms3_BL','mms4_BL']
         options,'BL_multi',colors=[0,6,4,2]
         options, 'BL_multi', 'ytitle', 'B!dL!N!c!c[nT]'
         options, 'BL_multi', labels=['mms1','mms2','mms3','mms4']
         options, 'BL_multi','labflag',-1
         options, 'BL_multi','ysubtitle',''

   
;have start and stop times in these vt (coded for up to 5 intervals per s/c)
;vt_mms2= time_double(['2019-01-25/04:11:00', '2019-01-25/04:03:25',  $ ;continuous front in MMS2 (right to left)
;                      '2019-01-25/04:06:27', '2019-01-25/04:03:20',  $ ;front section right to left
;                      '2019-01-25/04:06:59', '2019-01-25/04:11:00',  $
;                      '2019-01-25/04:13:30', '2019-01-25/04:16:11',  $
;                      '2019-01-25/04:16:11', '2019-01-25/04:13:30',  $  ; right to left
;                      ''], tformat='YYYY-MM-DD/hh:mm:ss.fffff')
;vt_mms1=time_double([ '2019-01-25/04:06:45', '2019-01-25/04:03:23',  $ ;front section right to left
;                      '2019-01-25/04:07:00', '2019-01-25/04:10:57',  $
;                      '2019-01-25/04:13:30', '2019-01-25/04:16:17',  $
;                      '2019-01-25/04:16:17', '2019-01-25/04:13:30',  $ ; right to left
;                      ''], tformat='YYYY-MM-DD/hh:mm:ss.fffff')
;
;vt_mms4=time_double([ '2019-01-25/04:06:39', '2019-01-25/04:03:23',  $ ;front section right to left
;                      '2019-01-25/04:07:02', '2019-01-25/04:11:00',  $
;                      '2019-01-25/04:13:30', '2019-01-25/04:16:24',  $
;                      '2019-01-25/04:16:24', '2019-01-25/04:13:30',   $; right to left
;                      ''], tformat='YYYY-MM-DD/hh:mm:ss.fffff')
;                      
;vt_mms3=time_double([ '2019-01-25/04:06:37', '2019-01-25/04:03:23',  $ ;front section right to left
;                      '2019-01-25/04:07:02', '2019-01-25/04:11:00',  $
;                      '2019-01-25/04:13:34', '2019-01-25/04:16:32',  $
;                      '2019-01-25/04:16:32', '2019-01-25/04:13:34',  $  ;right to left
;                      ''], tformat='YYYY-MM-DD/hh:mm:ss.fffff')
;;///////
;;finalized segments ;sarah's 01/25 event times `
If time_string(trange_full[0]) eq '2019-01-25/04:00:00' then begin
 vt_mms2= time_double(['2019-01-25/04:10:59', '2019-01-25/04:03:25',  $ ;continuous front in MMS2 (right to left)
                      '2019-01-25/04:13:32', '2019-01-25/04:15:25', $  ; piecewise seg
                      '2019-01-25/04:16:13', '2019-01-25/04:15:25',  $  ; right to left piecewise segment
                      ''], tformat='YYYY-MM-DD/hh:mm:ss.fffff')
 vt_mms1=time_double(['2019-01-25/04:06:33', '2019-01-25/04:03:23',  $ ;front section right to left
                      '2019-01-25/04:06:59', '2019-01-25/04:10:53',  $
                      '2019-01-25/04:10:56', '2019-01-25/04:10:53',   $ ;right to left segment
                      '2019-01-25/04:13:30', '2019-01-25/04:15:30',  $
                      '2019-01-25/04:16:17', '2019-01-25/04:15:30',  $ ; right to left segment 
                      ''], tformat='YYYY-MM-DD/hh:mm:ss.fffff')
 vt_mms4=time_double([ '2019-01-25/04:06:36', '2019-01-25/04:03:23',  $ ;front section right to left
                      '2019-01-25/04:07:04', '2019-01-25/04:10:48',  $
                      '2019-01-25/04:11:00', '2019-01-25/04:10:49',  $
                      '2019-01-25/04:13:30', '2019-01-25/04:15:40',  $
                      '2019-01-25/04:16:24', '2019-01-25/04:15:40',   $; right to left
                      ''], tformat='YYYY-MM-DD/hh:mm:ss.fffff')
 vt_mms3=time_double([ '2019-01-25/04:06:45', '2019-01-25/04:03:30',  $ ;front section right to left
                      '2019-01-25/04:07:01', '2019-01-25/04:10:56',  $
                      '2019-01-25/04:11:01', '2019-01-25/04:10:57',   $ ;piece wise, right to left
                      '2019-01-25/04:13:31', '2019-01-25/04:15:43',  $
                      '2019-01-25/04:16:31', '2019-01-25/04:15:44',  $  ;right to left
                      ''], tformat='YYYY-MM-DD/hh:mm:ss.fffff')                      
endif else print, 'expecting 2019-01-25/04:00:00'  
;    

;my 01/27/2019
If time_string(trange_full[0]) eq '2019-01-27/04:00:00' then begin
  vt_mms2= time_double(['2019-01-27/12:21:19', '2019-01-27/12:43:21',  $ ;Full MMS2 interval, then in reverse T_0... actually midpt from both starting point
    '2019-01-27/12:43:21','2019-01-27/12:30:00'], tformat='YYYY-MM-DD/hh:mm:ss.fffff')
  ;'2019-01-27/12:21:25', '2019-01-27/12:43:21'; full window for mms2
  vt_mms1= time_double(['2019-01-27/12:30:31','2019-01-27/12:21:10', $  ;skimes boundary mid-way; and reverse
                        '2019-01-27/12:43:16','2019-01-27/12:30:51'], $
                        tformat='YYYY-MM-DD/hh:mm:ss.fffff')
                        ;['2019-01-27/12:21:19', '2019-01-27/12:31:17', $ ;mms2
;  vt_mms1= time_double(['2019-01-27/12:21:10', '2019-01-27/12:43:16','2019-01-27/12:30:31','2019-01-27/12:21:10', $  ;skimes boundary mid-way; and reverse
;    '2019-01-27/12:43:16','2019-01-27/12:30:51'], tformat='YYYY-MM-DD/hh:mm:ss.fffff')
  ;'2019-01-27/12:21:31', '2019-01-27/12:30:31' ;full 1st half for mms1
  ;'2019-01-27/12:30:51', '2019-01-27/12:43:16', second half from ~12:31 formms1 overshoots in current LMN from 1245UT
  
    vt_mms4= time_double(['2019-01-27/12:21:24', '2019-01-27/12:30:28', $
                      '2019-01-27/12:30:28', '2019-01-27/12:25:16', $  ;elec-edge some mixing seen in ions too
                      '2019-01-27/12:30:47','2019-01-27/12:30:56', $
                      '2019-01-27/12:43:16','2019-01-27/12:30:56', $
                      '2019-01-27/12:43:17', '2019-01-27/12:42:00'], tformat='YYYY-MM-DD/hh:mm:ss.fffff')
  vt_mms3= time_double(['2019-01-27/12:21:38', '2019-01-27/12:27:04', $ 
                        '2019-01-27/12:30:12', '2019-01-27/12:27:04', $  ;elec-edge some mixing seen in ions too
                        '2019-01-27/12:31:04','2019-01-27/12:31:21', $
                        '2019-01-27/12:43:00', '2019-01-27/12:31:06'], tformat='YYYY-MM-DD/hh:mm:ss.fffff') ;disconnected second half
  ;'2019-01-27/12:21:38', '2019-01-27/12:30:12' fulll 1st half for mms3 ;'2019-01-27/12:31:10','2019-01-27/12:33:20'    
endif else print, 'expecting 2019-01-27/12:21:24'
  ;
  ;          
  ;          
  ;          
;plotting ion 
;IF hasMMS4e then begin if electrons needed
  tplot,['mms2_dis_energyspectr_omni_fast','mms1_dis_energyspectr_omni_fast','mms4_dis_energyspectr_omni_fast','mms3_dis_energyspectr_omni_fast', 'y0_line', 'VIN_multi']
  timebar,[vt_mms2],color=!d.n_colors-2,linestyle=2 ;vertical lines indicated calulated intervals
  
  wait,1

;=========getting MP Distance 
;
;stop ;check intervals before calc ;check each vt_mms# to confirm ranges

For sc_i=0,3 do begin
  case sc_i of
    0: begin
      vt=vt_mms1
      get_data,'mms1_ViN', data=d
      sc_str='mms1'

    end
    1: begin
      vt=vt_mms2
      get_data,'mms2_ViN', data=d
      sc_str='mms2'
    end
    2: begin
      vt=vt_mms3
      get_data,'mms3_ViN', data=d
      sc_str='mms3'
    end
    3: begin
      vt=vt_mms4
      get_data,'mms4_ViN', data=d
      sc_str='mms4'
    end  
  endcase

  if vt[0]lt vt[1] then begin    ;moving away from boundary with time
    get_data,sc_str +'_ViN', data=d
    ;make sure the we pass the start/stop interval, bins ~4sec apart
    t_roi=where(d.x ge (vt[0] -3.989999) and d.x le (vt[1]+ 3.989999),/null, cnt)  ;vt[1],/null, cnt)
    print, 'leaving boundary'
  endif else begin
    if vt[0] gt vt[1] then begin ;moving towards a boundary with time
      get_data,sc_str +'_ViN', data=d
      t_roi=where(d.x ge (vt[1] -3.989999) and d.x le (vt[0]+ 3.989999),/null, cnt)
      t_roi=reverse(t_roi)
      print,'approaching boundary'
    endif
  endelse

  get_data,sc_str+'_ViN', data=d
  roi=time_string(d.x[t_roi[0]], tformat='hhmmss')+ 'UT' ;closest data point ahead time for signature
  dn_roi=make_array(cnt ,value=!VALUES.D_NAN,/double)
  dn_roi[0]=0.0

  get_data,sc_str+'_ViN', data=d
  for i=1L, cnt -1L DO dn_roi[i]=tsum(d.x[t_roi[0:i]],d.y[t_roi[0:i]]) ;this VN is only time and data values;
  dn_raw=dn_roi
  dn_abs_roi=abs(dn_roi)

  store_data,sc_str +'DN_raw',data={x:d.x[t_roi],y:dn_raw}, $  ;(abs applied)  
    dlim={colors:6, ytitle:'Distance to MP!C!C[km]', labels:'| V!Dn!Ndt |', labflag:-1, thick:2, psym:10}

  store_data,sc_str +'DN_ROI1',data={x:d.x[t_roi],y:dn_abs_roi}, $  ;(abs applied)    
    dlim={colors:6, ytitle:'Distance to MP!C!C[km]', labels:'| V!Dn!Ndt |', labflag:-1, thick:2, psym:10}

  DN=create_struct(sc_str +'DN_ROI1',{x:d.x[t_roi],y:dn_abs_roi})
  ;;;;;;;;;;;;;;;;;;;;;;;;
  ; for multiple near boundaries
  ;;;;;;;;;;;;;;;;;;;;;;;;
  if n_elements(vt) gt 3 then begin ;with an empty entry ensure proper vt_mmm# syntax, gt3 instead or 2 (which would be one start & stop and the '' entries

    get_data,sc_str+'_ViN', data=d
    if vt[2]lt vt[3] then t_roi2=where(d.x ge (vt[2] -3.989999) and d.x le (vt[3]+ 3.989999),/null, cnt2) else begin
      t_roi2=where(d.x ge (vt[3] -3.989999) and d.x le (vt[2]+ 3.989999),/null, cnt2)
      t_roi2=reverse(t_roi2)
      print,'approaching boundary senario'
    endelse
    roi=roi+ '_' + time_string(d.x[t_roi2[0]], tformat='hhmmss')+ 'UT'

    dn_roi2=make_array(cnt2 ,value=!VALUES.D_NAN,/double)
    dn_roi2[0]=0.0
    for i=1L, cnt2 -1L DO dn_roi2[i]=tsum(d.x[t_roi2[0:i]],d.y[t_roi2[0:i]])
    dn_raw2=dn_roi2
    dn_roi2=abs(dn_roi2)

    store_data,sc_str +'dn_raw2',data={x:d.x[t_roi2],y:dn_raw2},dlim={colors:3, labflag:-1, thick:2, psym:10}

    store_data,sc_str +'DN_ROI2',data={x:d.x[t_roi2],y:dn_roi2}, $
      dlim={colors:6, labflag:-1, thick:2, psym:10}

    store_data, sc_str +'DN', data=[sc_str +'DN_ROI1',sc_str +'DN_ROI2']
    ;   ylim, 'DN',10,10000,1 ;1=log
    ;ylim, 'DN',0,8000,0 ;linear =0
    
     DN=create_struct(DN, sc_str +'DN_ROI2',{x:d.x[t_roi2],y:dn_roi2})
  endif
  if n_elements(vt) gt 5 then begin
    get_data,sc_str+'_ViN', data=d
    if vt[4]lt vt[5] then t_roi3=where(d.x ge (vt[4] -3.989999) and d.x le (vt[5]+ 3.989999),/null, cnt3) else begin
      t_roi3=where(d.x ge (vt[5] -3.989999) and d.x le (vt[4]+ 3.989999),/null, cnt3)
      t_roi3=reverse(t_roi3)
      print,'approaching boundary senario'
    endelse
    get_data,sc_str+'_ViN', data=d
    roi=roi+ '_' + time_string(d.x[t_roi3[0]], tformat='hhmmss')+ 'UT'

    dn_roi3=make_array(cnt3 ,value=!VALUES.D_NAN,/double)
    dn_roi3[0]=0.0
    for i=1L, cnt3 -1L DO dn_roi3[i]=tsum(d.x[t_roi3[0:i]],d.y[t_roi3[0:i]]) ;n-component, Vn
    dn_raw3=dn_roi3
    dn_roi3=abs(dn_roi3)
    ;          dmax3=max(dn_roi3) ;max(abs(dn_roi))
    ;          tmax3=where(dn_roi3 eq dmax3)
    ;         if vt[4] gt vt[5] then dn_roi3= dmax3 -dn_roi3

    store_data,sc_str +'DN_ROI3',data={x:d.x[t_roi3],y:dn_roi3},dlim={colors:6, labflag:-1, thick:2, psym:10}
    store_data,sc_str +'DN_raw3',data={x:d.x[t_roi3],y:dn_raw3},dlim={colors:6, labflag:-1, thick:2, psym:10}

    store_data,sc_str + 'DN', data=[sc_str +'DN_ROI1',sc_str +'DN_ROI2',sc_str +'DN_ROI3']
    ;   ylim, 'DN',10,10000,1 ;1=log
    ;ylim, 'DN',0,8000,0 ;linear =0
    
    DN=create_struct(DN, sc_str +'DN_ROI3',{x:d.x[t_roi3],y:dn_roi3})

    store_data, sc_str +'DN_beta', data=sc_str +'DN_raw3', $
      dlim={colors:6, ytitle:'Distance to MP!C!C[km]', labels:'| V!Dn!Ndt |', labflag:-1, thick:2, psym:10}
    ;   ylim, 'DN',10,10000,1 ;1=log
    ;ylim, 'DN',0,8000,0 ;linear =0
  endif
  if n_elements(vt) gt 7 then begin
    get_data,sc_str+'_ViN', data=d
    if vt[6]lt vt[7] then t_roi4=where(d.x ge (vt[6] -3.989999) and d.x le (vt[7]+ 3.989999),/null, cnt4) else begin
      t_roi4=where(d.x ge (vt[7] -3.989999) and d.x le (vt[6]+ 3.989999),/null, cnt4)
      t_roi4=reverse(t_roi4)
      print,'approaching boundary senario'
    endelse
    get_data,sc_str+'_ViN', data=d
    roi=roi+ '_' + time_string(d.x[t_roi4[0]], tformat='hhmmss')+ 'UT'

    dn_roi4=make_array(cnt4 ,value=!VALUES.D_NAN,/double)
    dn_roi4[0]=0.0
    get_data,sc_str+'_ViN', data=d
    for i=1L, cnt4 -1L DO dn_roi4[i]=tsum(d.x[t_roi4[0:i]],d.y[t_roi4[0:i]]) ;n-component, Vn
    dn_raw4=dn_roi4
    dn_roi4=abs(dn_roi4)
    ;      dmax4=max(dn_roi4) ;max(abs(dn_roi))
    ;      tmax4=where(dn_roi4 eq dmax4)
    ;      if vt[6] gt vt[7] then dn_roi4= dmax4 -dn_roi4


    store_data,sc_str +'DN_ROI4',data={x:d.x[t_roi4],y:dn_roi4},dlim={colors:6, labflag:-1, thick:2, psym:10}
    store_data,sc_str +'DN_raw4',data={x:d.x[t_roi4],y:dn_raw4},dlim={colors:1, labflag:-1, thick:2, psym:10}

    store_data, sc_str +'DN', data=[sc_str +'DN_ROI1',sc_str +'DN_ROI2',sc_str +'DN_ROI3', sc_str +'DN_ROI4']
    ;   ylim, 'DN',10,10000,1 ;1=log
    ;ylim, 'DN',0,8000,0 ;linear =0
    ;options, 'DN', 'panel_size',2

    DN=create_struct(DN, sc_str +'DN_ROI4',{x:d.x[t_roi4],y:dn_roi4})

    store_data, sc_str +'DN_beta', data=[sc_str +'DN_raw3', sc_str +'DN_raw4']
    options,sc_str +'DN_beta','ytitle','!CDistance Relative to!CBoundary crossing !C[km]'
    ;   ylim, 'DN',10,10000,1 ;1=log
    ;ylim, 'DN',0,8000,0 ;linear =0

  endif
  if n_elements(vt) gt 9 then begin
    get_data,sc_str+'_ViN', data=d
    for ii=0, floor((n_elements(vt)-10)/2) DO begin ;empty entry

      if vt[8+(2*ii)]lt vt[9+(2*ii)] then t_roii=where(d.x ge (vt[8 +(2*ii)] -3.989999) and d.x le (vt[9+(2*ii)]+ 3.989999),/null, cntii) else begin
        t_roii=where(d.x ge (vt[9+(2*ii)] -3.989999) and d.x le (vt[8+(2*ii)]+ 3.989999),/null, cntii)
        t_roii=reverse(t_roii)
        print,'approaching boundary senario'
      endelse
      roi=roi+ '_' + time_string(d.x[t_roii[0]], tformat='hhmmss')+ 'UT'

      dn_roii=make_array(cntii ,value=!VALUES.D_NAN,/double)
      dn_roii[0]=0.0
      get_data,sc_str+'_ViN', data=d
      for i=1L, cntii -1L DO dn_roii[i]=tsum(d.x[t_roii[0:i]],d.y[t_roii[0:i]]) ;n-component, Vn
      dn_rawi=dn_roii

      case ii of
        0: begin
          store_data,sc_str +'DN_ROI5',data={x:d.x[t_roii],y:abs(dn_roii)},dlim={colors:6, labflag:-1, thick:2, psym:10}
          store_data,sc_str +'DN_raw5',data={x:d.x[t_roii],y:dn_rawi},dlim={colors:1, labflag:-1, thick:2, psym:10}

          store_data, sc_str +'DN', data=[sc_str +'DN_ROI1',sc_str +'DN_ROI2',sc_str +'DN_ROI3', sc_str +'DN_ROI4', sc_str +'DN_ROI5']
          DN=create_struct(DN, sc_str +'DN_ROI5',{x:d.x[t_roii],y:dn_roii})
        end
        1: begin
          store_data,sc_str +'DN_ROI6',data={x:d.x[t_roii],y:abs(dn_roii)},dlim={colors:6, labflag:-1, thick:2, psym:10}
          store_data, sc_str +'DN_raw6',data={x:d.x[t_roii],y:dn_rawi},dlim={colors:1, labflag:-1, thick:2, psym:10}

          store_data, sc_str +'DN', data=[sc_str +'DN_ROI1',sc_str +'DN_ROI2',sc_str +'DN_ROI3', sc_str +'DN_ROI4',sc_str + 'DN_ROI5',sc_str + 'DN_ROI6']
          DN=create_struct(DN, sc_str +'DN_ROI6',{x:d.x[t_roii],y:dn_roii})
        end
        2: begin
          store_data,sc_str +'DN_ROI7',data={x:d.x[t_roii],y:abs(dn_roii)},dlim={colors:6, labflag:-1, thick:2, psym:10}
          store_data, sc_str +'DN_raw7',data={x:d.x[t_roii],y:dn_rawi},dlim={colors:1, labflag:-1, thick:2, psym:10}

          store_data, sc_str +'DN', data=[sc_str +'DN_ROI1',sc_str +'DN_ROI2',sc_str +'DN_ROI3', sc_str +'DN_ROI4', sc_str +'DN_ROI5',sc_str + 'DN_ROI6', sc_str +'DN_ROI7']
          DN=create_struct(DN, sc_str +'DN_ROI7',{x:d.x[t_roii],y:dn_roii})
        end
        3: begin
          store_data,sc_str +'DN_ROI8',data={x:d.x[t_roii],y:abs(dn_roii)},dlim={colors:6, labflag:-1, thick:2, psym:10}
          store_data,sc_str +'DN_raw8',data={x:d.x[t_roii],y:dn_rawi},dlim={colors:1, labflag:-1, thick:2, psym:10}

          store_data, sc_str +'DN', data=[sc_str +'DN_ROI1',sc_str +'DN_ROI2',sc_str +'DN_ROI3', sc_str +'DN_ROI4',sc_str + 'DN_ROI5', sc_str +'DN_ROI6', sc_str +'DN_ROI7', sc_str +'DN_ROI8']
          DN=create_struct(DN, sc_str +'DN_ROI8',{x:d.x[t_roii],y:dn_roii})
        end
        4: begin
          store_data, sc_str +'DN_ROI9',data={x:d.x[t_roii],y:abs(dn_roii)},dlim={colors:6, labflag:-1, thick:2, psym:10}
          store_data,sc_str +'DN_raw9',data={x:d.x[t_roii],y:dn_rawi},dlim={colors:1, labflag:-1, thick:2, psym:10}

          store_data, sc_str +'DN', data=[sc_str +'DN_ROI1',sc_str +'DN_ROI2',sc_str +'DN_ROI3', sc_str +'DN_ROI4',sc_str + 'DN_ROI5', sc_str +'DN_ROI6', sc_str +'DN_ROI7', sc_str +'DN_ROI8', sc_str +'DN_ROI9']
          DN=create_struct(DN, sc_str +'DN_ROI9',{x:d.x[t_roii],y:dn_roii})
        end
        5: begin
          store_data, sc_str +'DN_ROI10',data={x:d.x[t_roii],y:abs(dn_roii)},dlim={colors:6, labflag:-1, thick:2, psym:10}
          store_data,sc_str +'DN_raw10',data={x:d.x[t_roii],y:dn_rawi},dlim={colors:1, labflag:-1, thick:2, psym:10}

          store_data, sc_str +'DN', data=[sc_str +'DN_ROI1',sc_str +'DN_ROI2',sc_str +'DN_ROI3', sc_str +'DN_ROI4',sc_str + 'DN_ROI5', sc_str +'DN_ROI6', sc_str +'DN_ROI7', sc_str +'DN_ROI8', sc_str +'DN_ROI9',sc_str +'DN_ROI10']
          DN=create_struct(DN, sc_str +'DN_ROI10',{x:d.x[t_roii],y:dn_roii})
        end
      endcase
      
    endfor
  endif
  case sc_i of
    0: mms1DN = DN
    1: mms2DN = DN
    2: mms3DN = DN
    3: mms4DN = DN
  endcase; second set for structure save
  
Endfor

store_data, 'multi_mmsDN', data = ['mms1DN','mms2DN','mms3DN','mms4DN']
options, 'multi_mmsDN', 'ytitle', 'GSE_x[km]'
options, 'multi_mmsDN', labels=['mms1DN','mms2DN','mms3DN','mms4DN']
options,'multi_mmsDN',colors=[0,6,4,2]
options, 'multi_mmsDN','labflag',-1
options, 'multi_mmsDN','ysubtitle',''



;====distance test
;MEC file times  every 30 secs
If time_string(trange_full[0]) eq '2019-01-27/04:00:00' then begin
nfile=file_search('~kllera/Downloads/May8_email/mms[1-4]_GSE_LMN_20190127_1200_1300.csv',count=cnt) ;downloaded gmail guy's Auto4.py

if cnt eq 0 then print, 'files not found, check path' $ ;not robust, assuming mms1-4 files are all present
  else begin
    dmms1=read_csv(nfile[0], /N_TABLE_HEADER, TABLE_HEADER= fields)
    dmms2=read_csv(nfile[1], /N_TABLE_HEADER, TABLE_HEADER= fields)
    dmms3=read_csv(nfile[2], /N_TABLE_HEADER, TABLE_HEADER= fields)
    dmms4=read_csv(nfile[3], /N_TABLE_HEADER, TABLE_HEADER= fields)
 ;  MMS> help, dmms1
;    ** Structure <96833408>, 7 tags, length=7744, data length=7744, refs=1:
;    FIELD1          STRING    Array[121]
;    FIELD2          DOUBLE    Array[121]
;    FIELD3          DOUBLE    Array[121]
;    FIELD4          DOUBLE    Array[121]
;    FIELD5          DOUBLE    Array[121]
;    FIELD6          DOUBLE    Array[121]
;    FIELD7          DOUBLE    Array[121]   
; print, fields
;iso_time,GSE_X_km,GSE_Y_km,GSE_Z_km,L_km,M_km,N_km    
;colors=[0,6,4,2]    
   
    ;sc_str= 'mms' +string(jj,format='(I0)') +'_'
    store_data,'mms1_GSE_x_km',data={x:time_double(dmms1.(0)),y:dmms1.(1)},  $
      dlim={colors:0, ytitle:'MMS1 GSE_x[km]', labflag:-1, thick:2, psym:10}
    store_data,'mms2_GSE_x_km',data={x:time_double(dmms2.(0)),y:dmms2.(1)},  $
      dlim={colors:6, ytitle:'MMS2 GSE_x[km]', labflag:-1, thick:2, psym:10}
    store_data,'mms3_GSE_x_km',data={x:time_double(dmms3.(0)),y:dmms3.(1)},  $
      dlim={colors:4, ytitle:'MMS4 GSE_x[km]', labflag:-1, thick:2, psym:10} 
    store_data,'mms4_GSE_x_km',data={x:time_double(dmms4.(0)),y:dmms4.(1)},  $
      dlim={colors:2, ytitle:'MMS4 GSE_x[km]', labflag:-1, thick:2, psym:10}
    
    
    ;relative to adjacent s/c
    store_data,'mms1n2',data={x:time_double(dmms1.(0)),y:(dmms1.(1)- dmms2.(1))},  $
      dlim={colors:0, ytitle:'s/c dist !N GSE_x[km]', label: 'mms1n2', labflag:-1, thick:2, psym:10}  
    store_data,'mms4n1',data={x:time_double(dmms4.(0)),y:(dmms4.(1)- dmms1.(1))},  $
      dlim={colors:6, ytitle:'s/c dist !N GSE_x[km]', label: 'mms1n2', labflag:-1, thick:2, psym:10}
    store_data,'mms3n4',data={x:time_double(dmms3.(0)),y:(dmms3.(1)- dmms4.(1))},  $
      dlim={colors:4, ytitle:'s/c dist !N GSE_x[km]', label: 'mms1n2', labflag:-1, thick:2, psym:10}   
    store_data,'mms2n3',data={x:time_double(dmms2.(0)),y:-(dmms2.(1)- dmms3.(1))},  $
      dlim={colors:2, ytitle:'s/c dist !N GSE_x[km]', label: 'mms1n2', labflag:-1, thick:2, psym:10}
    
    store_data, 'dist_multi', data = ['mms1n2','mms4n1','mms3n4','mms2n3']
    options, 'dist_multi', 'ytitle', 'GSE_x[km]'
    options, 'dist_multi', labels=['mms1n2','mms4n1','mms3n4','-mms2n3']
    options,'dist_multi',colors=[0,6,4,2]
    options, 'dist_multi','labflag',-1
    options, 'dist_multi','ysubtitle',''
    options,'mms1n2',colors=0
    options,'mms4n1',colors=6
    options,'mms3n4',colors=4
    options,'mms2n3',colors=2
    
    ;relative Normal distance between sc (guy's rolling averge LMN)
    store_data,'N_mms1n2',data={x:time_double(dmms1.(0)),y:(dmms1.(6)- dmms2.(6))},  $
      dlim={colors:0, ytitle:'s/c dist !N N_t[km]', label: 'N_(mms1n2)', labflag:-1, thick:2, psym:10}
    store_data,'N_mms4n1',data={x:time_double(dmms1.(0)),y:(dmms4.(6)- dmms1.(6))},  $
      dlim={colors:6, ytitle:'s/c dist !N N_t[km]', label: 'N_(mms4n1)', labflag:-1, thick:2, psym:10}
    store_data,'N_mms3n4',data={x:time_double(dmms1.(0)),y:(dmms3.(6)- dmms4.(6))},  $
      dlim={colors:4, ytitle:'s/c dist !N N_t[km]', label: 'N_(mms3n4)', labflag:-1, thick:2, psym:10}
    store_data,'N_mms2n3',data={x:time_double(dmms1.(0)),y:-(dmms2.(6)- dmms3.(6))},  $
      dlim={colors:0, ytitle:'s/c dist !N N_t[km]', label: '-N_(mms2n3)', labflag:-1, thick:2, psym:10}
    
    store_data, 'N_dist_multi', data = ['N_mms1n2','N_mms4n1','N_mms3n4','N_mms2n3']
    options, 'N_dist_multi', 'ytitle', 's/c dist !N N_t[km]
    options, 'N_dist_multi', labels=['N_mms1n2','N_mms4n1','N_mms3n4','-N_mms2n3']
    options,'N_dist_multi',colors=[0,6,4,2]
    options, 'N_dist_multi','labflag',-1
    options, 'N_dist_multi','ysubtitle',''
    options,'N_mms1n2',colors=0
    options,'N_mms4n1',colors=6
    options,'N_mms3n4',colors=4
    options,'N_mms2n3',colors=2  
    
;stop
   endelse
   endif

;====Plotting our distance to MP from the various segments
;all ion spectra with piecewised MP distance and all s/c V_n at end
tplot,['mms2_dis_energyspectr_omni_fast','mms2DN','mms1_dis_energyspectr_omni_fast','mms1DN','mms4_dis_energyspectr_omni_fast','mms4DN','mms3_dis_energyspectr_omni_fast','mms3DN', 'y0_line', 'VIN_multi','mms3_des_energyspectr_omni_fast'], window=0
timebar,[vt_mms4],color=!d.n_colors-2,linestyle=2 ;vertical lines indicated calulated intervals

window,1
;tplot,['mms3_dis_energyspectr_omni_fast','mms3DN','y0_line', 'mms3_ViN','mms3_des_energyspectr_omni_fast'],window=1 
;timebar,[vt_mms3],color=!d.n_colors-2,linestyle=2 ;vertical lines indicated calulated intervals

;tplot,['mms4_dis_energyspectr_omni_fast','mms4DN','y0_line', 'mms4_ViN','mms4_des_energyspectr_omni_fast'],window=1 
;timebar,[vt_mms4],color=!d.n_colors-2,linestyle=2
tplot,['mms2_dis_energyspectr_omni_fast','mms2DN','y0_line', 'mms2_ViN'],window=1 ;mms is black
timebar,[vt_mms2],color=!d.n_colors-2,linestyle=2 ;vertical lines indicated calulated intervals

wait,.2

window,2
 tplot,['dist_multi','N_dist_multi'],window=2
 
 ylim,['mms1DN','mms2DN','mms3DN','mms4DN'],0,10000
 tplot,['multi_mmsDN','mms1DN','mms2DN','mms3DN','mms4DN'],window=2


;;; zoomed jan 27
;ts= '2019-01-27/12:19:59'
;dt = 24;5;24; 18; 28;min       (28 for full; 24 for slightly)
;timespan, ts, dt,/minute
;fname=time_string(ts, tformat='YYYY_MM_DD_hhmmss')+'MP_Motion_givenLMN_Vion'+ systime(); strmid(timestamp(),0,19) ;timestamp() nolonger working?!
;FNAME2='/users/kllera/MMS_plots/' +fname+ '_dist2MP_pagewidth.ps'
;
; t0=time_double(['2019-01-27/12:30:50'], tformat='YYYY-MM-DD/hh:mm:ss.fffff')
; tsz='2019-01-27/12:27:59'
; dtz=5 
; timespan, tsz, dtz,/minute
; fnamez=time_string(tsz, tformat='YYYY_MM_DD_hhmmss')+'MP_Motion_givenLMN_Vion'+ systime(); strmid(timestamp(),0,19) ;timestamp() nolonger working?!
; FNAME2z='/users/kllera/MMS_plots/' +fnamez+ '_dist2MP_pagewidth.ps'
;POPEN, FNAME2, xsize=6.5,ysize=5.5, units=inches
;tplot,['BL_multi','mms2_dis_energyspectr_omni_fast','mms2_des_energyspectr_omni_fast','mms1_dis_energyspectr_omni_fast','mms1_des_energyspectr_omni_fast','mms3_dis_energyspectr_omni_fast','mms3_des_energyspectr_omni_fast']
;timebar,[t0],color=!d.n_colors-2,linestyle=2
;pclose

stop 
;now create end product
;save,TRANGE_FULL, TRANGE_FULL_STR, TRANGE_LMN, TRANGE_LMN_STR, LHAT, MHAT, NHAT, VI_LMN1, VI_LMN2, VI_LMN3, VI_LMN4, mms1DN, mms2DN, mms3DN, mms4DN, filename='Distance_mp_lmn_system_20190125_0400-0430.sav', description='Added the boundary-normal distance to MP for each s/c, mms#DN. Contains 5-plot segements each s/c except for MMS2 which has 3. Ex: MMS2DN.MMS2DN_ROI3.x to get x, UnixTime. '
;restore,'~/Distance_mp_lmn_system_20190125_0400-0430.sav',/verbose
print, 'done'

END
;mms2DN:       2019-01-25/04:15:24      3031. and mms2DN:       2019-01-25/04:10:53      775.5 
;ctime, vt,vy  ;curser clicks for vertical line, v. time and v. y ; CLick boundary w/cold accleration 1st
;ctime, vt,vy, npoints=10
;timebar,[vt],color=!d.n_colors-2,linestyle=2
;str_vt=time_string(vt) ;insert string-time segments to vt
;vt= time_double(str_vt, tformat='YYYY-MM-DD/hh:mm:ss.fffff')
;

;for all pannel snapshot
;tplot,['mms2_dis_energyspectr_omni_fast','mms2_des_energyspectr_omni_fast','mms1_dis_energyspectr_omni_fast','mms1_des_energyspectr_omni_fast','mms4_dis_energyspectr_omni_fast','mms4_des_energyspectr_omni_fast','mms3_dis_energyspectr_omni_fast','mms3_des_energyspectr_omni_fast', 'y0_line', 'VIN_multi'], window=0
