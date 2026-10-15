"""车辆遥测清洗。仅依赖 pandas / numpy；默认输出到本脚本旁 results/。

python clean_telemetry.py --input "原始.csv"
不删除原始记录，不学习全样本统计填补参数，不训练模型。
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import time
import numpy as np
import pandas as pd

CORE = ['Vehicle speed', 'ENGINE RPM', 'ACCELERATION X', 'ACCELERATION Y', 'ACCELERATION Z']
BINARY = ['IDLING', 'IGNITION_STATUS', 'TOWING']
CODES = ['FUEL SYSTEM STATUS', 'FUEL TYPE', 'HARDWARE STATUS', 'MONITOR STATUS SINCE DTCS CLEARED',
         'MONITOR STATUS THIS DRIVE CYCLE', 'OBD STANDARDS THIS VEHICLE CONFORMS TO', 'OXYGEN SENSORS PRESENT IN 2 BANKS']
COUNTERS = ['DISTANCE TRAVELED SINCE CODES CLEARED','DISTANCE TRAVELED WITH MIL ON',
            'RUN TIME SINCE ENGINE START','WARM UPS SINCE CODES CLEARED']
FAST = ['ABSOLUTE THROTTLE POSITION B','ABSOLUTE THROTTLE POSITION D','ABSOLUTE THROTTLE POSITION E',
        'RELATIVE THROTTLE POSITION','COMMANDED THROTTLE ACTUATOR','CALCULATED ENGINE LOAD','TIMING ADVANCE']
PARAMS = dict(segment_gap_seconds=300, window_seconds=10, local_minus_utc_seconds=19800,
              core_ttl_seconds=10, fast_ttl_seconds=20, other_ttl_seconds=60, gps_ttl_seconds=10,
              gps_diagnostic_max_gap_seconds=10, gps_speed_review_kmh=200,
              altitude_review_below=-100, altitude_review_above=500)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_csv(df, path):
    df.to_csv(path,index=False,encoding='utf-8-sig',date_format='%Y-%m-%d %H:%M:%S.%f')

def canonical(v):
    return re.sub(r'[^a-z0-9]+','_',v.lower()).strip('_')

def prepare(raw):
    required=['deviceId','timeMili','timestamp','value','variable','alarmClass']
    assert list(raw.columns)==required, 'Unexpected source columns'
    d=raw.copy()
    d.insert(0,'source_row_id',np.arange(1,len(d)+1))
    d['vehicle']=d.deviceId.map({v:f'V{i+1}' for i,v in enumerate(d.deviceId.unique())})
    d['time_local']=pd.to_datetime(d.timestamp,errors='raise')
    d['time_utc_inferred']=d.time_local-pd.Timedelta(seconds=PARAMS['local_minus_utc_seconds'])
    ms=pd.to_numeric(d.timeMili.replace('',np.nan),errors='raise')
    inferred=((d.time_utc_inferred-pd.Timestamp('1970-01-01'))/pd.Timedelta(milliseconds=1)).round().astype('int64')
    assert inferred[ms.notna()].eq(ms[ms.notna()]).all(), 'Inconsistent observed UTC offset'
    d['time_mili_was_missing']=ms.isna().astype('int8')
    d['time_mili_clean']=inferred
    d['alarm_class']=pd.to_numeric(d.alarmClass,errors='raise').astype('int8')
    assert d.alarm_class.isin(range(6)).all()
    d['raw_time_backstep']=d.groupby('deviceId',sort=False).time_local.diff().dt.total_seconds().lt(0).astype('int8')
    assert not d.duplicated(['deviceId','time_local','variable']).any(), 'Unexpected duplicate key: resolve explicitly'
    d=d.sort_values(['vehicle','time_local','source_row_id'],kind='stable').reset_index(drop=True)
    delta=d.groupby('vehicle').time_local.diff().dt.total_seconds()
    block=(delta.gt(PARAMS['segment_gap_seconds'])|delta.isna()).groupby(d.vehicle).cumsum().astype(int)
    d['segment_id']=d.vehicle+'_'+block.astype(str).str.zfill(3)
    d['variable_key']=d.variable.map(canonical)
    assert d[['variable','variable_key']].drop_duplicates().variable_key.is_unique
    d['value_num']=pd.to_numeric(d.value,errors='coerce')
    d['value_type']='continuous'
    d.loc[d.variable.isin(BINARY),'value_type']='binary'
    d.loc[d.variable.isin(CODES),'value_type']='code'
    d.loc[d.variable.isin(COUNTERS),'value_type']='counter'
    d.loc[d.variable.eq('POSITION'),'value_type']='position'
    d.loc[d.variable.eq('PIDNameNotAvailable'),'value_type']='unknown'
    d['unknown_variable']=d.variable.eq('PIDNameNotAvailable').astype('int8')
    d['numeric_parse_failure']=(d.value_num.isna()&~d.variable.eq('POSITION')&~d.unknown_variable.astype(bool)).astype('int8')
    assert d.numeric_parse_failure.sum()==0
    for v,g in d[d.variable.isin(BINARY)].groupby('variable'):
        assert g.value_num.isin([0,1]).all(), v
    d['position_format_invalid']=0
    for field in ['latitude','longitude','altitude']:
        d[field+'_raw']=np.nan
        d[field]=np.nan
    mask=d.variable.eq('POSITION')
    parts=d.loc[mask,'value'].str.split(',',expand=True)
    assert parts.shape[1]==3
    for i,field in enumerate(['latitude','longitude','altitude']):
        parsed=pd.to_numeric(parts[i],errors='coerce')
        d.loc[mask,field+'_raw']=parsed
        d.loc[mask,field]=parsed
    invalid=mask&(d[['latitude','longitude','altitude']].isna().any(axis=1)|d.latitude.abs().gt(90)|d.longitude.abs().gt(180))
    d.loc[invalid,'position_format_invalid']=1
    zero=mask&d.latitude.eq(0)&d.longitude.eq(0)&d.altitude.eq(0)
    d['gps_zero_placeholder']=zero.astype('int8')
    d.loc[zero|invalid,['latitude','longitude','altitude']]=np.nan
    d['altitude_review']=(mask&(d.altitude_raw.lt(PARAMS['altitude_review_below'])|d.altitude_raw.gt(PARAMS['altitude_review_above']))).astype('int8')
    d['protocol_value_review']=((d.variable.eq('DISTANCE TRAVELED SINCE CODES CLEARED')&d.value_num.eq(65520))|
          (d.variable.eq('WARM UPS SINCE CODES CLEARED')&d.value_num.isin([255,2.5]))).astype('int8')
    # These round diagnostic thresholds are flags only; never remove or cap these values.
    d['high_value_review']=((d.variable.eq('ENGINE RPM')&d.value_num.gt(6000))|
                           (d.variable.eq('Vehicle speed')&d.value_num.gt(150))).astype('int8')
    d['counter_decrease_review']=0
    cc=d[d.variable.isin(COUNTERS)]
    decrease=cc.groupby(['vehicle','segment_id','variable']).value_num.diff().lt(0)
    d.loc[cc.index,'counter_decrease_review']=decrease.astype('int8')
    d['matched_position_row_id']=pd.Series(pd.NA,index=d.index,dtype='Int64')
    pos_lookup={(r.deviceId,r.timestamp,r.value,r.alarmClass):r.source_row_id
                for r in d[mask].itertuples()}
    for idx,r in d[d.unknown_variable.eq(1)].iterrows():
        match=pos_lookup.get((r.deviceId,r.timestamp,r.value,r.alarmClass))
        if match is not None: d.loc[idx,'matched_position_row_id']=match
    # Unknown numeric values remain available in value_num, but are excluded from wide features.
    return d

def gps_view(d):
    cols=['source_row_id','vehicle','deviceId','segment_id','time_local','time_utc_inferred',
          'latitude_raw','longitude_raw','altitude_raw','latitude','longitude','altitude',
          'gps_zero_placeholder','position_format_invalid','altitude_review','alarm_class']
    p=d[d.variable.eq('POSITION')][cols].copy()
    group=p.groupby(['vehicle','segment_id'])
    dt=group.time_local.diff().dt.total_seconds()
    lat1=np.radians(group.latitude.shift()); lat2=np.radians(p.latitude)
    dlat=lat2-lat1; dlon=np.radians(p.longitude-group.longitude.shift())
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    dist=6371000*2*np.arcsin(np.sqrt(a.clip(0,1)))
    eligible=dt.gt(0)&dt.le(PARAMS['gps_diagnostic_max_gap_seconds'])&p.latitude.notna()&lat1.notna()
    p['previous_position_row_id']=group.source_row_id.shift().astype('Int64')
    p['gps_pair_dt_seconds']=dt.where(eligible)
    p['gps_pair_distance_m']=dist.where(eligible)
    p['gps_derived_speed_kmh']=(dist/dt*3.6).where(eligible)
    p['gps_jump_review']=p.gps_derived_speed_kmh.gt(PARAMS['gps_speed_review_kmh']).astype('int8')
    return p

def dictionary(d):
    records=[]
    for v,g in d.groupby('variable',sort=True):
        typ=g.value_type.iloc[0]
        ttl=PARAMS['core_ttl_seconds'] if v in CORE else PARAMS['fast_ttl_seconds'] if v in FAST else PARAMS['other_ttl_seconds']
        if typ=='position':ttl=PARAMS['gps_ttl_seconds']
        if typ=='unknown':ttl=0
        records.append(dict(variable=v,variable_key=canonical(v),value_type=typ,
                            unit_status='not_verified',vehicles='|'.join(sorted(g.vehicle.unique())),
                            raw_rows=len(g),unique_raw_values=g.value.nunique(),
                            constant_observed=int(g.value.nunique()==1),ttl_seconds=ttl,
                            wide_policy='exclude_unknown' if typ=='unknown' else 'last_past_reading_with_expiry'))
    return pd.DataFrame(records)

def wide(d, dic, window=10, ttl_scale=1):
    # A core reading at t belongs to (ceil(t/w)*w - w, ceil(t/w)*w].
    anchors=d[d.variable.isin(CORE)][['vehicle','segment_id','time_local']].copy()
    anchors['reference_time']=anchors.time_local.dt.ceil(f'{window}s')
    anchors=anchors[['vehicle','segment_id','reference_time']].drop_duplicates().sort_values(['vehicle','reference_time']).reset_index(drop=True)
    anchors.insert(0,'sample_id',np.arange(1,len(anchors)+1))
    channels=[]
    for r in dic.itertuples():
        if r.value_type=='unknown':continue
        if r.value_type=='position':
            for f in ['latitude','longitude','altitude']:channels.append((r.variable,f,f,r.ttl_seconds))
        else:channels.append((r.variable,r.variable_key,'value_num',r.ttl_seconds))
    rawwide=anchors.copy();filled=anchors.copy(); lineage=[]
    for var,feature,col,ttl in channels:
        source=d[d.variable.eq(var)][['vehicle','segment_id','time_local','source_row_id',col]].rename(columns={col:'candidate_value','time_local':'source_time'})
        source['source_row_id']=source.source_row_id.astype('Int64')
        m=pd.merge_asof(anchors.sort_values('reference_time'),source.sort_values('source_time'),
                        left_on='reference_time',right_on='source_time',by=['vehicle','segment_id'],direction='backward',allow_exact_matches=True)
        m=m.sort_values('sample_id').reset_index(drop=True)
        age=(m.reference_time-m.source_time).dt.total_seconds()
        observed=m.candidate_value.notna()&age.ge(0)&age.lt(window)
        effective_ttl=ttl*ttl_scale
        # Even in the same window a reading must satisfy TTL. Baseline retains raw observations regardless of TTL.
        usable=m.candidate_value.notna()&age.ge(0)&age.le(effective_ttl)
        configured=m.vehicle.isin(d.loc[d.variable.eq(var),'vehicle'].unique())
        status=np.select([~configured,m.source_time.isna(),age.gt(effective_ttl),m.candidate_value.isna(),observed],
                         ['not_observed_on_device','no_history_in_segment','expired','invalid_latest_reading','observed_in_window'],default='carried_from_past')
        rawwide[feature]=m.candidate_value.where(observed).to_numpy()
        filled[feature]=m.candidate_value.where(usable).to_numpy()
        m['feature']=feature;m['age_seconds']=age;m['ttl_seconds']=effective_ttl
        m['raw_window_available']=observed.astype('int8');m['available']=usable.astype('int8')
        m['carried_from_past']=(usable&~observed).astype('int8');m['status']=status
        lineage.append(m[['sample_id','vehicle','segment_id','reference_time','feature','source_row_id','source_time',
                          'candidate_value','age_seconds','ttl_seconds','raw_window_available','available','carried_from_past','status']])
    lin=pd.concat(lineage,ignore_index=True)
    if len(lin):
        matched=lin.source_time.notna()
        assert lin.loc[matched,'age_seconds'].ge(0).all()
        assert lin.loc[lin.available.eq(1),'age_seconds'].le(lin.loc[lin.available.eq(1),'ttl_seconds']).all()
        assert not anchors.duplicated(['vehicle','segment_id','reference_time']).any()
    return rawwide,filled,lin

def alarm_windows(d, window=10):
    a=d.copy();a['window_end']=a.time_local.dt.ceil(f'{window}s')
    keys=['vehicle','segment_id','window_end']
    counts=a.groupby(keys).size().rename('observed_record_count')
    labels=pd.crosstab([a.vehicle,a.segment_id,a.window_end],a.alarm_class).reindex(columns=range(6),fill_value=0)
    labels.columns=[f'class_{c}_records' for c in range(6)]
    x=counts.to_frame().join(labels).reset_index()
    for c in range(1,6): x[f'observed_alarm_{c}']=x[f'class_{c}_records'].gt(0).astype('int8')
    x['window_start_exclusive']=x.window_end-pd.Timedelta(seconds=window)
    x['observed_nonzero_class_count']=x[[f'observed_alarm_{c}' for c in range(1,6)]].sum(axis=1)
    x['observation_interpretation']=np.where(x.observed_nonzero_class_count.gt(0),'alarm_observed','no_alarm_in_observed_records')
    return x

def run(args):
    started=time.perf_counter(); src=Path(args.input).resolve();dest=Path(args.output).resolve();dest.mkdir(parents=True,exist_ok=True)
    source_hash=sha(src)
    raw=pd.read_csv(src,dtype=str,keep_default_na=False,nrows=args.limit)
    d=prepare(raw); dic=dictionary(d); gps=gps_view(d)
    jump_ids=set(gps.loc[gps.gps_jump_review.eq(1),'source_row_id'])
    d['gps_jump_review']=d.source_row_id.isin(jump_ids).astype('int8')
    wr,wf,lin=wide(d,dic)
    alarms=d[d.alarm_class.gt(0)][['source_row_id','vehicle','deviceId','segment_id','time_local','variable','variable_key','value','value_num','alarm_class','unknown_variable']].copy()
    aw=alarm_windows(d)
    segments=d.groupby(['vehicle','segment_id']).agg(start=('time_local','min'),end=('time_local','max'),raw_rows=('source_row_id','size'),variables=('variable','nunique')).reset_index()
    segments['span_seconds']=(segments.end-segments.start).dt.total_seconds()
    flags=['time_mili_was_missing','raw_time_backstep','unknown_variable','numeric_parse_failure','position_format_invalid',
           'gps_zero_placeholder','altitude_review','protocol_value_review','high_value_review','counter_decrease_review','gps_jump_review']
    quality=pd.DataFrame([dict(vehicle=v,flag=f,count=int(g[f].sum()),denominator_all_vehicle_records=len(g))
                          for v,g in d.groupby('vehicle') for f in flags])
    label_audit=d.groupby(['vehicle','alarm_class']).agg(retained_rows=('source_row_id','size')).reset_index()
    label_audit['original_rows']=raw.assign(vehicle=raw.deviceId.map(d[['deviceId','vehicle']].drop_duplicates().set_index('deviceId').vehicle),alarm_class=raw.alarmClass.astype(int)).groupby(['vehicle','alarm_class']).size().reindex(pd.MultiIndex.from_frame(label_audit[['vehicle','alarm_class']])).to_numpy()
    assert label_audit.original_rows.eq(label_audit.retained_rows).all()
    cover=lin.groupby(['vehicle','feature']).agg(samples=('sample_id','size'),raw_available=('raw_window_available','sum'),filled_available=('available','sum'),carried=('carried_from_past','sum')).reset_index()
    cover['raw_missing_rate']=1-cover.raw_available/cover.samples;cover['filled_missing_rate']=1-cover.filled_available/cover.samples
    # Source IDs and all six raw fields survive, including unknown variables and zero positions.
    recovered=d.sort_values('source_row_id')[list(raw.columns)].reset_index(drop=True)
    pd.testing.assert_frame_equal(recovered,raw,check_dtype=False)
    assert d.groupby('vehicle').time_local.apply(lambda t:t.is_monotonic_increasing).all()
    assert d.loc[d.gps_zero_placeholder.eq(1),['latitude','longitude','altitude']].isna().all().all()
    assert int(aw[[f'class_{c}_records' for c in range(6)]].sum().sum())==len(d)
    assert len(alarms)==int(d.alarm_class.gt(0).sum())
    checks={'raw_fields_and_row_ids_conserved':True,'labels_conserved_by_vehicle':True,'time_sorted':True,
            'no_future_matching':True,'age_limits_obeyed':True,'zero_positions_invalidated':True,
            'alarm_window_counts_conserved':True,'input_unchanged':source_hash==sha(src)}
    assert all(checks.values())
    tables={'清洗长表.csv':d,'定位轨迹.csv':gps,'状态宽表_10秒_未填补.csv':wr,'状态宽表_10秒_有限历史对齐.csv':wf,
            '状态宽表_逐值来源.csv':lin,'告警观测表.csv':alarms,'告警窗口_10秒.csv':aw,'变量字典.csv':dic,
            '观测片段.csv':segments,'质量标记统计.csv':quality,'标签保留核对.csv':label_audit,'宽表缺失与填补统计.csv':cover}
    for name,frame in tables.items():write_csv(frame,dest/name)
    sensitivity=[]
    if not args.skip_sensitivity:
        for window,scale in [(5,1),(10,1),(30,1),(10,2)]:
            if window==10 and scale==1:r,f,l=wr,wf,lin
            else:r,f,l=wide(d,dic,window,scale)
            corekeys=[canonical(v) for v in CORE if v in d.variable.unique()]
            for vehicle in d.vehicle.unique():
                a=r[r.vehicle.eq(vehicle)];b=f[f.vehicle.eq(vehicle)]
                sensitivity.append(dict(window_seconds=window,ttl_multiplier=scale,vehicle=vehicle,samples=len(a),
                                        all_core_raw_available=int(a[corekeys].notna().all(axis=1).sum()),
                                        all_core_filled_available=int(b[corekeys].notna().all(axis=1).sum())))
        write_csv(pd.DataFrame(sensitivity),dest/'时间对齐敏感性.csv')
    seg_sensitivity=[]
    for gap in [120,300,600]:
        for v,g in d.groupby('vehicle'):
            count=int(g.time_local.drop_duplicates().sort_values().diff().dt.total_seconds().gt(gap).sum()+1)
            seg_sensitivity.append(dict(gap_seconds=gap,vehicle=v,segments=count))
    write_csv(pd.DataFrame(seg_sensitivity),dest/'分段阈值敏感性.csv')
    info={'source_path':str(src),'source_sha256':source_hash,'source_rows_used':len(d),'sample_limit':args.limit,
          'parameters':PARAMS,'raw_rows_deleted':0,'time_mili_reconstructed':int(d.time_mili_was_missing.sum()),
          'gps_zero_invalidated':int(d.gps_zero_placeholder.sum()),'unknown_pid_position_matches':int(d.matched_position_row_id.notna().sum()),
          'wide_samples':len(wf),'wide_features':len(wf.columns)-4,'alarm_records':len(alarms),'alarm_windows':len(aw),
          'segments':len(segments),'checks':checks,'quality_flags':{f:int(d[f].sum()) for f in flags},
          'runtime_seconds':round(time.perf_counter()-started,3),'python':sys.version.split()[0],'pandas':pd.__version__,'numpy':np.__version__}
    (dest/'清洗验收.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'输出哈希.json').write_text(json.dumps({p.name:sha(p) for p in sorted(dest.glob('*.csv'))},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2))
    return info

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',default=str(Path(__file__).resolve().parent/'results'))
    parser.add_argument('--limit',type=int,default=None,help='仅用于最小真实输入检查，正式结果不要使用')
    parser.add_argument('--skip-sensitivity',action='store_true')
    run(parser.parse_args())
