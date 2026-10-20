"""必要EDA：复用冻结分区、可追溯配对、开发集标签探索；不训练模型。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'utils'))
TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def jsread(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jswrite(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def check(ok, message):
    if not ok:
        raise ValueError(message)


def csv(path, frame):
    frame.to_csv(path, index=False, encoding='utf-8-sig', lineterminator='\n',
                 date_format=TIME_FORMAT, float_format='%.10g')


def stats(values):
    s = pd.Series(values).dropna().astype(float)
    n = len(s)
    q = s.quantile([.05, .25, .5, .75, .95])
    return dict(n=n, mean=s.mean(), sd=s.std(ddof=1), minimum=s.min(), p05=q.loc[.05],
                q1=q.loc[.25], median=q.loc[.5], q3=q.loc[.75], p95=q.loc[.95], maximum=s.max(),
                zero_count=int(s.eq(0).sum()), zero_fraction=s.eq(0).mean() if n else np.nan)


def spearman(x, y):
    """Spearman是平均秩的Pearson相关；不计算逐行独立假设下的p值。"""
    pair = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(pair) < 3:
        return np.nan, 'fewer_than_3_pairs'
    if pair.x.nunique() < 2 or pair.y.nunique() < 2:
        return np.nan, 'constant_variable'
    return pair.x.rank(method='average').corr(pair.y.rank(method='average')), 'ok_descriptive_only'


def movement(g):
    s = g.vehicle_speed
    return pd.Series(np.select([s.isna(), s.eq(0), s.gt(0)],
                               ['missing', 'zero', 'positive'], default='negative'), index=g.index)


def missing_reason(representation, status):
    return 'outside_current_window' if representation == 'unfilled' and status == 'carried_from_past' else status


def continuous_ids(g, max_gap, time_col='reference_time'):
    """先按时间排序再调用；更换车辆、片段或超过间隔均断线。"""
    cut = g.vehicle.ne(g.vehicle.shift()) | g.segment_id.ne(g.segment_id.shift())
    cut |= g[time_col].diff().dt.total_seconds().gt(max_gap)
    return cut.cumsum()


def scopes(frame, config):
    yield 'all_descriptive', frame
    yield 'development', frame[frame.split.isin(config['development_splits'])]


def load_inputs():
    clean = ROOT/'outputs/数据清洗'
    prep = ROOT/'outputs/评价准备'
    checks = []
    for base in [clean, prep]:
        for name, expected in jsread(base/'交付文件哈希.json').items():
            p = base/Path(name.replace('\\', '/'))
            check(sha(p) == expected, f'归档变化：{p}')
            checks.append({'path': str(p.relative_to(ROOT)), 'sha256': expected})
    for row in jsread(ROOT/'outputs/项目管理/输入来源.json')['inputs']:
        check(sha(row['path']) == row['sha256'], '原始输入变化')
        checks.append({'path': row['path'], 'sha256': row['sha256']})
    mapping = pd.read_csv(prep/'results/观测片段划分.csv')
    # 全长表首次读取显式排除两个目标字段；全样本描述不接触测试标签。
    cols = ['source_row_id', 'vehicle', 'segment_id', 'time_local', 'variable_key', 'value_num',
            'latitude', 'high_value_review', 'protocol_value_review', 'numeric_parse_failure']
    raw = pd.read_csv(clean/'results/清洗长表.csv', usecols=cols)
    raw['time_local'] = pd.to_datetime(raw.time_local, format=TIME_FORMAT)
    raw = raw.merge(mapping[['vehicle', 'segment_id', 'split']], on=['vehicle', 'segment_id'],
                    how='left', validate='many_to_one')
    check(not raw.split.isna().any(), '长表分区缺失')
    source = pd.read_csv(clean/'results/状态宽表_逐值来源.csv')
    source['source_time'] = pd.to_datetime(source.source_time, format=TIME_FORMAT)
    wide = {}
    for name, filename in [('unfilled', '状态宽表_10秒_未填补.csv'), ('aligned', '状态宽表_10秒_有限历史对齐.csv')]:
        w = pd.read_csv(clean/'results'/filename)
        w['reference_time'] = pd.to_datetime(w.reference_time, format=TIME_FORMAT)
        w = w.merge(mapping[['vehicle', 'segment_id', 'split']], on=['vehicle', 'segment_id'], validate='many_to_one')
        wide[name] = w
    check(len(raw) == 173853 and len(wide['unfilled']) == 3132, '输入规模不符')
    return raw, source, wide, checks


def analyze(raw, source, wide, config, results):
    core = config['core_features']
    features = list(dict.fromkeys(core + [x for pair in config['pairs'] for x in pair] + ['idling']))
    raw = raw.copy()
    raw['review'] = raw[config['review_flags']].max(axis=1).astype(bool)
    source = source[source.feature.isin(features)].copy()
    src = raw[['source_row_id', 'review', 'vehicle', 'segment_id', 'time_local', 'value_num', 'variable_key']]
    provenance = source.merge(src, on='source_row_id', how='left', suffixes=('', '_raw'), validate='many_to_one')
    has = provenance.source_row_id.notna()
    check((provenance.loc[has, 'vehicle'] == provenance.loc[has, 'vehicle_raw']).all(), '来源车辆错配')
    check((provenance.loc[has, 'segment_id'] == provenance.loc[has, 'segment_id_raw']).all(), '来源片段错配')
    check((provenance.loc[has, 'feature'] == provenance.loc[has, 'variable_key']).all(), '来源变量错配')
    check((provenance.loc[has, 'source_time'] == provenance.loc[has, 'time_local']).all(), '来源时间错配')
    check((provenance.loc[has, 'candidate_value'] == provenance.loc[has, 'value_num']).all(), '来源数值错配')
    check((provenance.loc[has, 'age_seconds'] >= 0).all(), '未来来源')
    coverage, descriptions, states, idling_rows, missing_rows = [], [], [], [], []
    window_detail = []
    for representation, w in wide.items():
        available = 'raw_window_available' if representation == 'unfilled' else 'available'
        for feature in features:
            p = provenance[provenance.feature == feature].set_index('sample_id').reindex(w.sample_id)
            check(np.array_equal(p[available].eq(1).to_numpy(), w[feature].notna().to_numpy()), '宽表有效性与来源不一致')
            for scope, g in scopes(w, config):
                for vehicle, v in g.groupby('vehicle'):
                    z = p.loc[v.sample_id]
                    good = v[feature].notna().to_numpy()
                    usable = z[good]
                    if feature in core:
                        coverage.append(dict(scope=scope, representation=representation, vehicle=vehicle,
                            feature=feature, windows=len(v), available=int(good.sum()),
                            available_fraction=float(good.mean()), missing=int((~good).sum()),
                            carried=int(usable.carried_from_past.sum()) if representation == 'aligned' else 0,
                            age_median=usable.age_seconds.median(), age_p95=usable.age_seconds.quantile(.95),
                            age_max=usable.age_seconds.max(), reviewed_available=int(usable.review.fillna(False).sum())))
                        for quality in ['all_values', 'without_review']:
                            values = v[feature].copy()
                            if quality == 'without_review':
                                values.loc[z.review.fillna(False).to_numpy()] = np.nan
                            descriptions.append(dict(scope=scope, representation=representation, vehicle=vehicle,
                                                     feature=feature, quality=quality, **stats(values)))
                        for status, n in z.loc[~good, 'status'].value_counts().items():
                            # 来源status描述有限历史口径；未填补有值过窗但仍有效时另命名。
                            label = missing_reason(representation, status)
                            missing_rows.append(dict(scope=scope, representation=representation, vehicle=vehicle,
                                feature=feature, source_status=label, missing_count=int(n)))
            if feature in core:
                d = w[['sample_id', 'vehicle', 'segment_id', 'reference_time', 'split', feature]].rename(columns={feature:'value'})
                d = d.assign(representation=representation, feature=feature,
                             source_row_id=p.source_row_id.to_numpy(), source_time=p.source_time.to_numpy(),
                             age_seconds=p.age_seconds.to_numpy(), review=p.review.fillna(False).astype(bool).to_numpy())
                window_detail.append(d)
        for scope, g in scopes(w, config):
            for vehicle, v in g.groupby('vehicle'):
                counts = movement(v).value_counts()
                for state in ['zero', 'positive', 'negative', 'missing']:
                    states.append(dict(scope=scope, representation=representation, vehicle=vehicle,
                        state=state, windows=len(v), count=int(counts.get(state, 0)),
                        fraction=int(counts.get(state, 0))/len(v), all5_complete=int(v[core].notna().all(axis=1).sum())))
                known = v.vehicle_speed.notna() & v.engine_rpm.notna()
                idle = v.vehicle_speed.eq(0) & v.engine_rpm.gt(0)
                comparable = known & v.idling.isin([0,1])
                idling_rows.append(dict(scope=scope, representation=representation, vehicle=vehicle,
                    windows=len(v), speed_rpm_known=int(known.sum()), candidate_idle=int((idle & known).sum()),
                    candidate_fraction_of_known=float(idle[known].mean()) if known.any() else np.nan,
                    paired_idling_flag=int(comparable.sum()),
                    both_true=int((comparable & idle & v.idling.eq(1)).sum()),
                    candidate_true_flag_false=int((comparable & idle & v.idling.eq(0)).sum()),
                    candidate_false_flag_true=int((comparable & ~idle & v.idling.eq(1)).sum()),
                    both_false=int((comparable & ~idle & v.idling.eq(0)).sum())))
    for scope, g in scopes(raw[raw.variable_key.isin(core)], config):
        for (vehicle, feature), v in g.groupby(['vehicle', 'variable_key']):
            for quality in ['all_values', 'without_review']:
                values = v.value_num if quality == 'all_values' else v.loc[~v.review, 'value_num']
                descriptions.append(dict(scope=scope, representation='raw_records', vehicle=vehicle,
                                         feature=feature, quality=quality, **stats(values)))
    pairs, relations = [], []
    for representation, w in wide.items():
        for x, y in config['pairs']:
            g = w[['sample_id','vehicle','segment_id','reference_time','split',x,y]].rename(columns={x:'x',y:'y'}).copy()
            for name, feature in [('x',x),('y',y)]:
                p = provenance[provenance.feature == feature].set_index('sample_id').reindex(g.sample_id)
                for column in ['source_row_id','source_time','age_seconds','review']:
                    g[f'{name}_{column}'] = p[column].to_numpy()
            g = g.dropna(subset=['x','y']).copy()
            g['source_time_difference_seconds'] = (g.x_source_time-g.y_source_time).dt.total_seconds().abs()
            g['review'] = g.x_review.astype(bool) | g.y_review.astype(bool)
            g['representation'], g['x_feature'], g['y_feature'] = representation,x,y
            pairs.append(g)
            for scope, h in scopes(g, config):
                for vehicle in sorted(raw.vehicle.unique()):
                    v = h[h.vehicle == vehicle]
                    for quality in ['all_values','without_review']:
                        q = v if quality == 'all_values' else v[~v.review]
                        for alignment in ['within_ttl','source_delta_le_2s']:
                            a = q if alignment == 'within_ttl' else q[q.source_time_difference_seconds <= config['strict_pair_time_difference_seconds']]
                            rho,status = spearman(a.x,a.y)
                            relations.append(dict(scope=scope,representation=representation,vehicle=vehicle,
                                x_feature=x,y_feature=y,quality=quality,alignment=alignment,n_pairs=len(a),rho=rho,status=status,
                                unique_source_pairs=len(a[['x_source_row_id','y_source_row_id']].drop_duplicates()),
                                source_delta_median=a.source_time_difference_seconds.median(),
                                source_delta_p95=a.source_time_difference_seconds.quantile(.95),
                                source_delta_max=a.source_time_difference_seconds.max()))
    gap_rows, changes = [], []
    for scope,g in scopes(raw[raw.variable_key.isin(core)],config):
        for (vehicle,feature),v in g.groupby(['vehicle','variable_key']):
            v = v.sort_values(['segment_id','time_local','source_row_id']).copy()
            v['previous_source_row_id'] = v.groupby('segment_id').source_row_id.shift()
            v['previous_time'] = v.groupby('segment_id').time_local.shift()
            v['previous_value'] = v.groupby('segment_id').value_num.shift()
            v['dt_seconds'] = (v.time_local-v.previous_time).dt.total_seconds()
            v['delta_raw'] = v.value_num-v.previous_value
            dt = v.dt_seconds.dropna()
            gap_rows.append(dict(scope=scope,vehicle=vehicle,feature=feature,records=len(v),
                adjacent_pairs=len(dt),median_seconds=dt.median(),p95_seconds=dt.quantile(.95),
                max_seconds=dt.max(),gaps_gt_30s=int(dt.gt(30).sum()),gaps_gt_300s=int(dt.gt(300).sum())))
            if scope == 'development':
                valid = v[v.dt_seconds.gt(0) & v.dt_seconds.le(config['raw_line_max_gap_seconds'])].copy()
                valid['abs_delta'] = valid.delta_raw.abs()
                selected = valid.sort_values(['abs_delta','time_local','source_row_id'],ascending=[False,True,True]).head(config['top_changes_per_vehicle_feature'])
                changes.append(selected[['source_row_id','vehicle','segment_id','variable_key','split','time_local',
                    'value_num','review','previous_source_row_id','previous_time','previous_value','dt_seconds','delta_raw']])
    typical = []
    w = wide['unfilled']
    for vehicle,v in w[w.split.isin(config['development_splits'])].groupby('vehicle'):
        good = v[v[core].notna().all(axis=1)].sort_values('reference_time').copy()
        if good.empty:
            typical.append(dict(vehicle=vehicle,status='no_all5_complete_interval',n=0))
            continue
        good['run'] = continuous_ids(good,config['window_continuity_seconds'])
        choices = good.groupby('run').agg(n=('sample_id','size'),start=('reference_time','min'),end=('reference_time','max'))
        best = choices.sort_values(['n','start'],ascending=[False,True]).iloc[0]
        rows = good[good.run == best.name]
        typical.append(dict(vehicle=vehicle,status='selected',n=int(best.n),start=best.start,end=best.end,
            observed_span_seconds=(best.end-best.start).total_seconds(),segment_id=rows.segment_id.iloc[0],
            split=rows.split.iloc[0],first_sample_id=int(rows.sample_id.iloc[0]),last_sample_id=int(rows.sample_id.iloc[-1])))
    # 标签只在读取时过滤开发区source_row_id；不保留测试标签到目标分析链。
    dev_ids = set(raw.loc[raw.split.isin(config['development_splits']), 'source_row_id'])
    chunks = []
    for chunk in pd.read_csv(ROOT/'outputs/数据清洗/results/清洗长表.csv',usecols=['source_row_id','alarm_class'],chunksize=20000):
        chunks.append(chunk[chunk.source_row_id.isin(dev_ids)])
    labels = pd.concat(chunks,ignore_index=True)
    # 用已冻结候选索引确定纳入范围，不依赖未知PID的字符串拼写。
    index = pd.read_csv(ROOT/'outputs/评价准备/results/候选样本索引.csv',usecols=['source_row_id','split'])
    allowed = set(index.loc[index.split.isin(config['development_splits']), 'source_row_id'])
    dev = raw[raw.source_row_id.isin(allowed)].merge(labels,on='source_row_id',validate='one_to_one')
    check(len(dev)==63801 and not dev.split.eq('test').any(), '目标探索范围泄漏')
    counts = dev.groupby(['vehicle','variable_key','alarm_class']).size().rename('records').reset_index()
    counts['denominator_vehicle_variable'] = counts.groupby(['vehicle','variable_key']).records.transform('sum')
    counts['within_variable_fraction'] = counts.records/counts.denominator_vehicle_variable
    target_values = []
    for (vehicle,feature,label),g in dev[dev.variable_key.isin(features)].groupby(['vehicle','variable_key','alarm_class']):
        target_values.append(dict(vehicle=vehicle,feature=feature,alarm_class=int(label),**stats(g.value_num)))
    source_counts = []
    for scope,g in scopes(raw,config):
        for vehicle,v in g.groupby('vehicle'):
            source_counts.append(dict(scope=scope,vehicle=vehicle,raw_records=len(v),
                core_records=int(v.variable_key.isin(core).sum()),gps_records=int(v.variable_key.eq('position').sum()),
                valid_gps=int((v.variable_key.eq('position') & v.latitude.notna()).sum()),
                first_observation=v.time_local.min(),last_observation=v.time_local.max(),segments=v.segment_id.nunique()))
    output = {'核心覆盖.csv':pd.DataFrame(coverage),'核心分布.csv':pd.DataFrame(descriptions),
        '速度状态.csv':pd.DataFrame(states),'候选怠速核对.csv':pd.DataFrame(idling_rows),
        '缺失来源.csv':pd.DataFrame(missing_rows),'配对相关性.csv':pd.DataFrame(relations),
        '配对明细.csv':pd.concat(pairs,ignore_index=True),'核心窗口来源.csv':pd.concat(window_detail,ignore_index=True),
        '采样间隔.csv':pd.DataFrame(gap_rows),'开发集较大变化索引.csv':pd.concat(changes,ignore_index=True),
        '典型连续区间.csv':pd.DataFrame(typical),'开发集变量类别计数.csv':counts,
        '开发集标签内分布.csv':pd.DataFrame(target_values),'观测覆盖.csv':pd.DataFrame(source_counts)}
    for name,frame in output.items():
        csv(results/name,frame)
    return output,raw,provenance


def self_test():
    s=stats([0,1,2,np.nan])
    check(s['n']==3 and s['median']==1 and s['zero_count']==1 and s['sd']==1,'分母/标准差失败')
    check(spearman(pd.Series([1,2,2,3]),pd.Series([4,3,3,1]))[0] == -1, '并列秩相关失败')
    check(spearman(pd.Series([1,1,1]),pd.Series([2,3,4]))[1]=='constant_variable','常量处理失败')
    check(np.isnan(stats([])['median']), '空样本不应填0')
    state=movement(pd.DataFrame({'vehicle_speed':[0,2,np.nan,-1]})).tolist()
    check(state==['zero','positive','missing','negative'],'状态分母失败')
    g=pd.DataFrame({'vehicle':['V1']*4,'segment_id':['a','a','a','b'],
                    'reference_time':pd.to_datetime(['2020-01-01 00:00:00','2020-01-01 00:00:10','2020-01-01 00:00:30','2020-01-01 00:00:40'])})
    check(continuous_ids(g,10).tolist()==[1,1,2,3],'断流/跨片段连线')
    check(missing_reason('unfilled', 'carried_from_past') == 'outside_current_window', '未填补窗外来源标注错误')
    check(missing_reason('aligned', 'carried_from_past') == 'carried_from_past', '已对齐来源标注错误')
    check(missing_reason('unfilled', 'expired') == 'expired', '过期来源标注错误')
    print('PASS: nine statistical, continuity and missing-reason boundary tests')


def run(output, figures=True):
    config=jsread(HERE/'参数.json')
    output=Path(output).resolve()
    check(output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'outputs/数据清洗')
          and not output.is_relative_to(ROOT/'outputs/评价准备'),'输出目录不安全')
    results=output/'results'; results.mkdir(parents=True,exist_ok=True)
    raw,source,wide,inputs=load_inputs()
    tables,raw,provenance=analyze(raw,source,wide,config,results)
    csv(results/'输入核验.csv',pd.DataFrame(inputs))
    if figures:
        from plot_diagnostics import plot_all
        plot_all(raw,wide,tables,config,output)
    jswrite(results/'EDA验收.json',{'status':'PASS','raw_rows':len(raw),'state_windows':len(wide['unfilled']),
        'development_target_rows':63801,'test_target_rows_in_target_analysis':0,'input_files_verified':len(inputs),
        'checks':['archive_hashes','same_vehicle_segment_feature_source','no_future_source',
                  'source_values_equal','availability_equal','target_scope_development_only'],
        'correlation_method':'Pearson correlation of average ranks (Spearman); no iid p-values',
        'unit_status':'unverified; all measurements in raw units',
        'not_executed':['alarm_episode_construction','model_training','formal_report']})
    from repro_manifest import build_manifest
    manifest=build_manifest([HERE/'run_eda.py',HERE/'plot_diagnostics.py',HERE/'参数.json',
        ROOT/'outputs/数据清洗/交付文件哈希.json',ROOT/'outputs/评价准备/交付文件哈希.json']+
        sorted((HERE/'utils').glob('*.py')),42,config,
        'python outputs/探索性分析/run_eda.py',['pandas','numpy','matplotlib','Pillow'])
    jswrite(results/'复现清单.json',manifest)
    paths=[p for p in results.iterdir() if p.is_file() and p.name not in ['输出哈希.json','复现清单.json']]
    paths += list((output/'diagnostics').glob('*.png'))
    jswrite(results/'输出哈希.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(paths)})
    print(json.dumps({'status':'PASS','tables':len(tables),'plots':len(list((output/'diagnostics').glob('*.png')))},ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--no-plots',action='store_true')
    args=p.parse_args()
    self_test() if args.self_test else run(args.output,not args.no_plots)
