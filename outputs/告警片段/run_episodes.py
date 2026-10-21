"""开发范围确定观测间隔；冻结规则后生成全量片段，开发范围关联验证。"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'outputs/探索性分析/utils'))
TIME = '%Y-%m-%d %H:%M:%S.%f'
KEYS = ['vehicle', 'segment_id', 'variable_key']
DEV = ['train', 'validation']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def csv(path, frame):
    frame.to_csv(path, index=False, encoding='utf-8-sig', lineterminator='\n',
                 date_format=TIME, float_format='%.10g')


def check(ok, message):
    if not ok:
        raise ValueError(message)


def verify_inputs():
    rows = []
    for r in read_json(ROOT/'outputs/项目管理/输入来源.json')['inputs']:
        p = Path(r['path'])
        rows.append(dict(path=str(p), expected=r['sha256'], actual=sha(p)))
    for folder in ['数据清洗', '评价准备', '探索性分析']:
        base = ROOT/'outputs'/folder
        for name, digest in read_json(base/'交付文件哈希.json').items():
            rows.append(dict(path=str(base/name), expected=digest, actual=sha(base/name)))
    for r in rows:
        check(r['expected'] == r['actual'], f"输入哈希改变: {r['path']}")
    return pd.DataFrame(rows)


def load_observations():
    # 首先完全不读取标签列。参数确定以后才载入开发标签；全量标签只用于固定规则应用。
    columns = ['source_row_id', 'vehicle', 'segment_id', 'time_local', 'variable',
               'variable_key', 'value_num', 'value_type', 'unknown_variable',
               'high_value_review', 'protocol_value_review', 'numeric_parse_failure',
               'gps_jump_review', 'counter_decrease_review', 'gps_zero_placeholder']
    raw = pd.read_csv(ROOT/'outputs/数据清洗/results/清洗长表.csv', usecols=columns)
    raw['time_local'] = pd.to_datetime(raw.time_local, format=TIME)
    splits = pd.read_csv(ROOT/'outputs/评价准备/results/观测片段划分.csv')
    raw = raw.merge(splits[['vehicle', 'segment_id', 'split']], on=['vehicle', 'segment_id'],
                    how='left', validate='many_to_one')
    check(raw.split.notna().all() and raw.source_row_id.is_unique, '分区或源行不完整')
    return raw


def calibrate(raw, config):
    check(config['selected_candidate'] == 1, '本阶段预登记固定中间候选；改变选择需另立版本')
    dev = raw[raw.split.isin(DEV)].sort_values(KEYS+['time_local', 'source_row_id']).copy()
    dev['delta'] = dev.groupby(KEYS).time_local.diff().dt.total_seconds()
    positive = dev[dev.delta.gt(0)]
    pooled = positive.groupby('variable_key').delta
    rows = []
    # 变量/车辆字典允许来自无标签输入；缺少开发观测时明确使用开发合并或固定回退。
    for (vehicle, variable), _ in raw.groupby(['vehicle', 'variable_key'], sort=True):
        d = positive.loc[positive.vehicle.eq(vehicle) & positive.variable_key.eq(variable), 'delta']
        source = 'vehicle_variable_development'
        used = d
        if len(used) < config['min_positive_intervals']:
            used = pooled.get_group(variable) if variable in pooled.groups else pd.Series(dtype=float)
            source = 'pooled_variable_development'
        if len(used) < config['min_positive_intervals']:
            median = config['fallback_median_seconds']
            source = 'fixed_fallback'
        else:
            median = float(used.median())
        if median <= config['fast_median_seconds']:
            gaps = config['fast_candidates_seconds']
        else:
            gaps = [int(np.clip(np.ceil(median*m/config['round_seconds'])*config['round_seconds'],
                                config['min_gap_seconds'], config['max_gap_seconds']))
                    for m in config['slow_multipliers']]
        rows.append(dict(vehicle=vehicle, variable_key=variable, interval_n=len(d),
                         median_seconds=d.median(), p95_seconds=d.quantile(.95),
                         cadence_source=source, calibration_interval_n=len(used),
                         used_median_seconds=median, short_gap=gaps[0],
                         selected_gap=gaps[config['selected_candidate']], long_gap=gaps[2]))
    return pd.DataFrame(rows)


def add_labels(raw, development_only):
    frame = raw[raw.split.isin(DEV)].copy() if development_only else raw.copy()
    ids = set(frame.source_row_id)
    parts = []
    for chunk in pd.read_csv(ROOT/'outputs/数据清洗/results/清洗长表.csv',
                             usecols=['source_row_id', 'alarm_class'], chunksize=20000):
        parts.append(chunk[chunk.source_row_id.isin(ids)])
    return frame.merge(pd.concat(parts), on='source_row_id', validate='one_to_one')


def construct(raw, gaps, gap_column='selected_gap'):
    """包含class0的完整序列；相邻标签变化/断档/观测段边界均断开。"""
    x = raw.merge(gaps[['vehicle', 'variable_key', gap_column]],
                  on=['vehicle', 'variable_key'], validate='many_to_one')
    x = x.rename(columns={gap_column: 'gap_threshold_seconds'})
    check(x.gap_threshold_seconds.notna().all(), '存在未冻结的间隔')
    x = x.sort_values(KEYS+['time_local', 'source_row_id']).reset_index(drop=True)
    groups = x.groupby(KEYS, sort=False)
    x['previous_label'] = groups.alarm_class.shift()
    x['previous_time'] = groups.time_local.shift()
    x['next_label'] = groups.alarm_class.shift(-1)
    x['next_time'] = groups.time_local.shift(-1)
    before = (x.time_local-x.previous_time).dt.total_seconds()
    after = (x.next_time-x.time_local).dt.total_seconds()
    new = x.previous_label.isna() | x.alarm_class.ne(x.previous_label) | before.gt(x.gap_threshold_seconds)
    x['run_number'] = new.cumsum()
    x['start_reason'] = np.select([x.previous_label.isna(), before.gt(x.gap_threshold_seconds)],
                                  ['segment_or_variable_start', 'sampling_gap'], default='label_change')
    x['end_reason'] = np.select([x.next_label.isna(), after.gt(x.gap_threshold_seconds)],
                                ['segment_or_variable_end', 'sampling_gap'], default='label_change')
    x['previous_gap_seconds'] = before
    x['next_gap_seconds'] = after
    # first/last不可跳过边界处NaN，因此使用head/tail而非groupby.first/last。
    first = x.groupby('run_number', sort=False).head(1).set_index('run_number')
    last = x.groupby('run_number', sort=False).tail(1).set_index('run_number')
    runs = first[['vehicle', 'segment_id', 'variable', 'variable_key', 'split', 'alarm_class',
                  'gap_threshold_seconds', 'previous_label', 'previous_gap_seconds', 'start_reason']].copy()
    runs['first_observed'] = first.time_local
    runs['last_observed'] = last.time_local
    runs['first_source_row_id'] = first.source_row_id
    runs['last_source_row_id'] = last.source_row_id
    runs['record_count'] = x.groupby('run_number', sort=False).size()
    runs['next_label'] = last.next_label
    runs['next_gap_seconds'] = last.next_gap_seconds
    runs['end_reason'] = last.end_reason
    runs['start_censored'] = runs.start_reason.ne('label_change')
    runs['end_censored'] = runs.end_reason.ne('label_change')
    runs['observed_span_seconds'] = (runs.last_observed-runs.first_observed).dt.total_seconds()
    # 稳定ID来自真实首条源行；不同敏感性设置可产生不同分组，不作为模型输入。
    runs['episode_id'] = ['E'+str(i).zfill(6) if c else 'N'+str(i).zfill(6)
                          for i, c in zip(runs.first_source_row_id, runs.alarm_class)]
    runs['identity_status'] = np.where(first.unknown_variable.astype(bool), 'unknown_pid', 'known_variable')
    x['episode_id'] = x.run_number.map(runs.episode_id)
    mapping = x.loc[x.alarm_class.ne(0), ['source_row_id', 'episode_id', 'vehicle', 'segment_id',
                                         'variable_key', 'time_local', 'alarm_class', 'split']]
    check(mapping.source_row_id.is_unique, '告警映射源行不唯一')
    check(mapping.groupby('episode_id').split.nunique().le(1).all(), '片段跨冻结分区')
    check(len(mapping) == raw.alarm_class.ne(0).sum(), '非零记录未守恒')
    return runs.reset_index(drop=True), mapping


def summarize_episodes(episodes, keys):
    return episodes.groupby(keys, dropna=False).agg(
        episodes=('episode_id', 'size'), alarm_records=('record_count', 'sum'),
        singletons=('record_count', lambda s: int(s.eq(1).sum())),
        span_median_seconds=('observed_span_seconds', 'median'),
        span_p95_seconds=('observed_span_seconds', lambda s: s.quantile(.95)),
        span_max_seconds=('observed_span_seconds', 'max'),
        start_censored=('start_censored', 'sum'), end_censored=('end_censored', 'sum')
    ).reset_index()


def match_states(anchors, raw, dictionary, features):
    """真实起点的过去匹配；最新无效值阻断旧值，时龄超TTL留空。"""
    parts = []
    for (vehicle, segment), a in anchors.groupby(['vehicle', 'segment_id'], sort=True):
        a = a.sort_values('first_observed')
        pool = raw[raw.vehicle.eq(vehicle) & raw.segment_id.eq(segment)]
        for feature in features:
            source = pool[pool.variable_key.eq(feature)].sort_values(['time_local', 'source_row_id'])
            ttl = float(dictionary.loc[dictionary.variable_key.eq(feature), 'ttl_seconds'].iloc[0])
            out = a[['episode_id', 'vehicle', 'segment_id', 'variable_key', 'alarm_class',
                     'split', 'first_observed']].copy()
            out['feature'] = feature
            out['same_variable'] = out.variable_key.eq(feature)
            out['ttl_seconds'] = ttl
            out['source_row_id'] = np.nan
            out['source_time'] = pd.NaT
            out['candidate_value'] = np.nan
            out['quality_review'] = False
            if len(source):
                positions = source.time_local.to_numpy().searchsorted(a.first_observed.to_numpy(), side='right')-1
                found = positions >= 0
                selected = source.iloc[np.maximum(positions, 0)]
                out.loc[found, 'source_row_id'] = selected.source_row_id.to_numpy()[found]
                out.loc[found, 'source_time'] = selected.time_local.to_numpy()[found]
                out.loc[found, 'candidate_value'] = selected.value_num.to_numpy()[found]
                flags = selected[['high_value_review', 'protocol_value_review', 'numeric_parse_failure']].any(axis=1)
                out.loc[found, 'quality_review'] = flags.to_numpy()[found]
            out['age_seconds'] = (out.first_observed-out.source_time).dt.total_seconds()
            out['status'] = np.select([out.source_time.isna(), out.age_seconds.gt(ttl),
                                       out.candidate_value.isna()],
                                      ['no_history', 'expired', 'invalid_latest'], default='available')
            out['value'] = out.candidate_value.where(out.status.eq('available'))
            check(out.age_seconds.dropna().ge(0).all(), '匹配使用未来状态')
            parts.append(out)
    return pd.concat(parts, ignore_index=True)


def state_summary(states):
    rows = []
    for keys, g in states.groupby(['vehicle', 'variable_key', 'alarm_class', 'feature', 'same_variable']):
        for policy in ['ttl_available', 'age_le_2_no_review']:
            s = g[g.status.eq('available')]
            if policy == 'age_le_2_no_review':
                s = s[s.age_seconds.le(2) & ~s.quality_review]
            rows.append(dict(zip(['vehicle', 'variable_key', 'alarm_class', 'feature', 'same_variable'], keys))
                        | dict(policy=policy, anchor_count=len(g), available_n=len(s),
                               coverage=len(s)/len(g), median=s.value.median(), q1=s.value.quantile(.25),
                               q3=s.value.quantile(.75), age_median=s.age_seconds.median(),
                               review_n=int(s.quality_review.sum())))
    return pd.DataFrame(rows)


def value_analysis(dev):
    scalar = dev[~dev.variable_key.isin(['position', 'pidnamenotavailable']) & ~dev.unknown_variable.astype(bool)]
    distribution = scalar.groupby(['vehicle', 'variable_key', 'alarm_class']).value_num.agg(
        n='count', minimum='min', q1=lambda s:s.quantile(.25), median='median',
        q3=lambda s:s.quantile(.75), maximum='max').reset_index()
    count = scalar.groupby(['vehicle', 'variable_key', 'alarm_class']).size().rename('records').reset_index()
    count['within_variable_fraction'] = count.records/count.groupby(['vehicle', 'variable_key']).records.transform('sum')
    repeated = scalar.groupby(['vehicle', 'variable_key', 'value_num'], dropna=False).agg(
        records=('source_row_id', 'size'), n_labels=('alarm_class', 'nunique'),
        labels=('alarm_class', lambda s:'|'.join(map(str, sorted(s.unique()))))).reset_index()
    # 相邻实际取值的标签集合变化，不外推成确定阈值。
    selected = repeated[repeated.variable_key.isin(['vehicle_speed', 'engine_rpm', 'idling'])].copy()
    selected = selected.sort_values(['vehicle', 'variable_key', 'value_num'])
    by = selected.groupby(['vehicle', 'variable_key'])
    selected['previous_value'] = by.value_num.shift()
    selected['previous_labels'] = by.labels.shift()
    boundaries = selected[selected.previous_value.notna() & selected.labels.ne(selected.previous_labels)]
    cross = repeated[repeated.variable_key.isin(['vehicle_speed', 'engine_rpm', 'idling'])]
    cross = cross.merge(cross, on=['variable_key', 'value_num'], suffixes=('_a', '_b'))
    cross = cross[cross.vehicle_a.lt(cross.vehicle_b)].copy()
    cross['same_label_set'] = cross.labels_a.eq(cross.labels_b)
    return distribution, count, repeated, boundaries, cross


def cooccurrence(dev):
    rows = []
    nonzero = dev[dev.alarm_class.ne(0) & ~dev.unknown_variable.astype(bool)].copy()
    for mode in ['exact_timestamp', 'right_closed_10s']:
        nonzero['anchor'] = nonzero.time_local if mode == 'exact_timestamp' else nonzero.time_local.dt.ceil('10s')
        for vehicle, g in nonzero.groupby('vehicle'):
            # 每个类在一个实际观测桶中至多计一次；分段ID作为桶身份的一部分。
            sets = {c:set(map(tuple, g.loc[g.alarm_class.eq(c), ['segment_id', 'anchor']].to_numpy()))
                    for c in range(1, 6)}
            for a, b in itertools.combinations(range(1, 6), 2):
                joint, union = len(sets[a]&sets[b]), len(sets[a]|sets[b])
                rows.append(dict(mode=mode, vehicle=vehicle, class_a=a, class_b=b,
                                 buckets_a=len(sets[a]), buckets_b=len(sets[b]),
                                 intersection=joint, union=union, jaccard=joint/union if union else np.nan))
    return pd.DataFrame(rows)


def select_cases(dev, runs, config):
    choices, evidence, eligibility = [], [], []
    for vehicle, variable, label in config['case_targets']:
        pool = runs[runs.vehicle.eq(vehicle) & runs.variable_key.eq(variable) & runs.alarm_class.eq(label)]
        candidates = []
        for r in pool.itertuples():
            t = r.first_observed
            half_window = pd.Timedelta(seconds=config['case_half_window_seconds'])
            window = dev[dev.vehicle.eq(vehicle) & dev.segment_id.eq(r.segment_id)
                         & dev.time_local.between(t-half_window, t+half_window)]
            counts = []
            for feature in ['vehicle_speed', 'engine_rpm']:
                f = window[window.variable_key.eq(feature) & window.value_num.notna()]
                counts.extend([int(f.time_local.lt(t).sum()), int(f.time_local.ge(t).sum())])
            ok = min(counts) >= config['case_min_each_side_speed_rpm_readings']
            eligibility.append(dict(episode_id=r.episode_id, vehicle=vehicle, variable_key=variable,
                                    alarm_class=label, before_speed=counts[0], after_speed=counts[1],
                                    before_rpm=counts[2], after_rpm=counts[3], eligible=ok))
            if ok:
                candidates.append((min(counts), sum(counts), r.record_count, r.first_source_row_id, r, window))
        check(bool(candidates), f'无合格开发案例：{vehicle}/{variable}/{label}')
        best = sorted(candidates, key=lambda z:(-z[0], -z[1], -z[2], z[3]))[0]
        r, window = best[4], best[5]
        record = r._asdict()
        record.pop('Index', None)
        choices.append(record | dict(min_side_readings=best[0], total_speed_rpm_readings=best[1],
                                     requested_half_window_seconds=config['case_half_window_seconds']))
        window = window.copy()
        window['case_episode_id'] = r.episode_id
        window['relative_seconds'] = (window.time_local-r.first_observed).dt.total_seconds()
        evidence.append(window)
    return pd.DataFrame(choices), pd.concat(evidence, ignore_index=True), pd.DataFrame(eligibility)


def case_boundary_evidence(dev, cases):
    rows = []
    for c in cases.itertuples():
        series = dev[dev.vehicle.eq(c.vehicle) & dev.segment_id.eq(c.segment_id)
                     & dev.variable_key.eq(c.variable_key)].sort_values(['time_local','source_row_id'])
        for role, part in [
            ('previous_observation', series[series.time_local.lt(c.first_observed)].tail(1)),
            ('first_observation', series[series.source_row_id.eq(c.first_source_row_id)]),
            ('last_observation', series[series.source_row_id.eq(c.last_source_row_id)]),
            ('next_observation', series[series.time_local.gt(c.last_observed)].head(1))
        ]:
            for r in part.itertuples():
                rows.append(dict(episode_id=c.episode_id, role=role, source_row_id=r.source_row_id,
                                 vehicle=r.vehicle, segment_id=r.segment_id, variable_key=r.variable_key,
                                 time_local=r.time_local, value_num=r.value_num, alarm_class=r.alarm_class,
                                 split=r.split))
    return pd.DataFrame(rows)


def special_checks(dev, runs, mapping, states):
    # 有疑点不等于已确认错误。V2每个极值源行都可追溯，不强行当连续案例。
    extreme = dev[dev.vehicle.eq('V2') & dev.variable_key.isin(['vehicle_speed', 'engine_rpm'])
                  & dev.high_value_review.astype(bool)].copy()
    extreme = extreme.merge(mapping[['source_row_id', 'episode_id']], on='source_row_id', how='left',
                            validate='one_to_one')
    extreme = extreme.merge(runs[['episode_id', 'record_count', 'observed_span_seconds',
                                   'start_censored', 'end_censored']], on='episode_id', how='left',
                            validate='many_to_one')
    # IDLING正常/非零游程起点的两种定义一致性，仅用TTL内已有读数。
    idle = runs[runs.variable_key.eq('idling')][['episode_id', 'vehicle', 'alarm_class', 'first_source_row_id']]
    pivot = states[states.feature.isin(['vehicle_speed', 'engine_rpm', 'idling'])].pivot(
        index='episode_id', columns='feature', values='value').reset_index()
    idle = idle.merge(pivot, on='episode_id', validate='one_to_one')
    idle['comparable'] = idle.vehicle_speed.notna() & idle.engine_rpm.notna() & idle.idling.isin([0, 1])
    idle['candidate_idle'] = (idle.vehicle_speed.eq(0) & idle.engine_rpm.gt(0)).where(idle.comparable)
    idle['idling_flag'] = idle.idling.eq(1).where(idle.comparable)
    return extreme, idle


def self_test():
    def frame(times, labels, segments=None):
        n = len(times)
        return pd.DataFrame(dict(source_row_id=np.arange(1, n+1), vehicle=['V1']*n,
            segment_id=segments or ['S1']*n, variable=['x']*n, variable_key=['x']*n,
            time_local=pd.Timestamp('2020-01-01')+pd.to_timedelta(times, unit='s'),
            alarm_class=labels, split=['train']*n, unknown_variable=[False]*n))
    gaps = pd.DataFrame([dict(vehicle='V1', variable_key='x', selected_gap=20)])
    r, m = construct(frame([0,5,10,15,20,45,50], [1,1,0,1,2,2,0]), gaps)
    a = r[r.alarm_class.ne(0)]
    check(a.record_count.tolist()==[2,1,1,1], '正常/换类/断档边界失败')
    check(a.observed_span_seconds.tolist()==[5,0,0,0], '末次观测跨度失败')
    check(a.start_censored.tolist()==[True,False,False,True], '左删失失败')
    check(a.end_censored.tolist()==[False,False,True,False], '右删失失败')
    check(set(m.source_row_id)=={1,2,4,5,6}, '映射守恒失败')
    r, _ = construct(frame([0,20,40,60], [1]*4, ['S1','S1','S2','S2']), gaps)
    check(len(r)==2 and r.record_count.tolist()==[2,2], '等于阈值/跨段失败')
    # 最新无效、恰好TTL、未来观测不可用。
    raw = frame([0,5,20], [0]*3)
    raw['value_num'] = [7, np.nan, 9]
    for flag in ['high_value_review','protocol_value_review','numeric_parse_failure']:
        raw[flag] = False
    anchor = r.iloc[[0]].copy()
    dictionary = pd.DataFrame([dict(variable_key='x',ttl_seconds=10)])
    anchor['first_observed'] = pd.Timestamp('2020-01-01 00:00:10')
    check(match_states(anchor,raw,dictionary,['x']).status.iloc[0]=='invalid_latest','无效值阻断失败')
    anchor['first_observed'] = pd.Timestamp('2020-01-01 00:00:16')
    check(match_states(anchor,raw,dictionary,['x']).status.iloc[0]=='expired','过期匹配失败')
    anchor['first_observed'] = pd.Timestamp('2020-01-01 00:00:30')
    check(match_states(anchor,raw,dictionary,['x']).value.iloc[0]==9,'TTL等值失败')
    anchor['first_observed'] = pd.Timestamp('2019-12-31 23:59:59')
    check(match_states(anchor,raw,dictionary,['x']).status.iloc[0]=='no_history','未来值匹配失败')
    config = read_json(HERE/'规则.json')
    f = pd.concat([frame(list(range(0,60,5)),[0]*12),
                   frame(list(range(100,300,10)),[0]*20).assign(split='test')])
    c = calibrate(f,config)
    check(c.selected_gap.iloc[0]==20 and c.interval_n.iloc[0]==11,'测试采样影响规则')
    print('PASS: 11 boundary / provenance / calibration checks')


def run(output):
    output = output.resolve()
    work = (ROOT/'work').resolve()
    check(output == HERE.resolve() or (output.is_relative_to(work) and output != work),
          '输出只允许当前告警片段目录或work子目录；不覆盖此前归档')
    output.mkdir(parents=True, exist_ok=True)
    results = output/'results'
    results.mkdir(exist_ok=True)
    config = read_json(HERE/'规则.json')
    verified = verify_inputs()
    raw = load_observations()
    gaps = calibrate(raw, config)
    # 先写入由时间信息确定的规则快照，再读取任何本次标签。
    csv(results/'冻结间隔.csv', gaps)
    rule_digest = sha(results/'冻结间隔.csv')
    write_json(results/'规则冻结记录.json', dict(rule_sha256=sha(HERE/'规则.json'),
        gap_table_sha256=rule_digest, calibration_scope=DEV, labels_used=False,
        selection='predeclared middle candidate; no outcome optimization'))
    dev = add_labels(raw, True)
    devruns, devmap = construct(dev, gaps)
    sensitivity = []
    for option in ['short_gap', 'selected_gap', 'long_gap']:
        r, _ = construct(dev, gaps, option)
        summary = summarize_episodes(r[r.alarm_class.ne(0)], ['vehicle', 'variable_key', 'alarm_class'])
        summary['gap_option'] = option
        sensitivity.append(summary)
    # 规则已冻结；全量只用于机械分段、记录守恒、分区检查。关联与案例继续仅使用dev。
    full = add_labels(raw, False)
    runs, mapping = construct(full, gaps)
    alarms = runs[runs.alarm_class.ne(0)].copy()
    check(sha(results/'冻结间隔.csv')==rule_digest, '冻结规则遭修改')
    check(set(devmap.source_row_id)==set(mapping.loc[mapping.split.isin(DEV),'source_row_id']), '开发映射范围变化')
    compare = mapping[mapping.split.isin(DEV)][['source_row_id','episode_id']].sort_values('source_row_id')
    check(compare.reset_index(drop=True).equals(devmap[['source_row_id','episode_id']].sort_values('source_row_id').reset_index(drop=True)),
          '全量应用改变开发片段')
    # 将后续模型候选与片段连接；正常记录空episode_id，不能当作模型特征。
    candidate = pd.read_csv(ROOT/'outputs/评价准备/results/候选样本索引.csv')
    candidate = candidate.merge(mapping[['source_row_id','episode_id']], on='source_row_id',
                                how='left', validate='one_to_one')
    boundary = mapping.groupby('episode_id').agg(split_count=('split','nunique'),
        source_rows=('source_row_id','size'), split=('split','first')).reset_index()
    boundary['exclude_from_model'] = boundary.split_count.gt(1)
    check(not boundary.exclude_from_model.any(), '跨分区片段需显式排除，不能重划边界')
    dictionary = pd.read_csv(ROOT/'outputs/数据清洗/results/变量字典.csv')
    devknown = devruns[devruns.identity_status.eq('known_variable') & devruns.variable_key.ne('position')]
    states = match_states(devknown, dev, dictionary, config['state_features'])
    # 再独立逐项核对匹配源车、段、变量、时间、值；不止检查时龄。
    found = states[states.source_row_id.notna()].merge(
        dev[['source_row_id','vehicle','segment_id','variable_key','time_local','value_num']],
        on='source_row_id', suffixes=('','_actual'), validate='many_to_one')
    check(found.vehicle.eq(found.vehicle_actual).all() and found.segment_id.eq(found.segment_id_actual).all()
          and found.feature.eq(found.variable_key_actual).all()
          and found.source_time.eq(found.time_local).all(), '状态来源身份不匹配')
    check(np.allclose(found.candidate_value,found.value_num,equal_nan=True), '状态来源值不匹配')
    distributions, counts, repeated, transitions, cross = value_analysis(dev)
    co = cooccurrence(dev)
    cases, evidence, eligibility = select_cases(dev, devruns, config)
    extremes, idle = special_checks(dev,devruns,devmap,states)
    tables = {
        '输入核验':verified, '间隔敏感性_开发':pd.concat(sensitivity,ignore_index=True),
        '告警观测片段_全量':alarms, '源行片段映射_全量':mapping,
        '模型候选片段索引':candidate, '分区边界核查':boundary,
        '片段摘要_按分区':summarize_episodes(alarms,['split','vehicle','alarm_class']),
        '开发片段摘要':summarize_episodes(devruns[devruns.alarm_class.ne(0)],['vehicle','variable_key','alarm_class']),
        '开发游程起点':devknown, '开发起点状态来源':states,
        '开发状态关联':state_summary(states), '开发类别值分布':distributions,
        '开发类别计数':counts, '开发重复值标签':repeated,
        '开发取值边界邻居':transitions, '开发跨车同值对照':cross, '开发类别共现':co,
        '案例索引':cases,'案例原始证据':evidence,'案例覆盖筛选':eligibility,
        '案例边界证据':case_boundary_evidence(dev,cases),
        'V2疑点记录追踪':extremes,'IDLING起点核对':idle,
        '未知PID保留':full[full.unknown_variable.astype(bool)]
    }
    for name, table in tables.items():
        csv(results/(name+'.csv'),table)
    from plot_diagnostics import plot_all
    plot_all(tables, config, output)
    audit = dict(status='PASS', raw_rows=len(raw), candidate_rows=len(candidate),
        development_candidate_rows=int(candidate.split.isin(DEV).sum()),
        nonzero_records=len(mapping), episodes=len(alarms),
        development_episodes=int(alarms.split.isin(DEV).sum()),
        singletons=int(alarms.record_count.eq(1).sum()),
        cross_split_episodes=int(boundary.exclude_from_model.sum()),
        excluded_candidate_rows=0, state_anchors=len(devknown), state_pairs=len(states),
        state_future_matches=int(states.age_seconds.lt(0).sum()),
        test_rows_in_association=0, input_files_verified=len(verified), cases=len(cases),
        unknown_pid_rows=int(full.unknown_variable.astype(bool).sum()),
        unknown_pid_nonzero_records=int((full.unknown_variable.astype(bool)&full.alarm_class.ne(0)).sum()),
        note='观测跨度不是故障持续时间；正常游程和告警片段起点作为状态对照，非独立随机样本。')
    write_json(results/'片段验收.json',audit)
    from repro_manifest import build_manifest
    files = [HERE/'run_episodes.py', HERE/'plot_diagnostics.py', HERE/'规则.json',
             ROOT/'outputs/探索性分析/utils/repro_manifest.py',
             ROOT/'outputs/探索性分析/utils/plot_style.py',
             ROOT/'outputs/探索性分析/utils/profile_data.py',
             ROOT/'outputs/探索性分析/utils/visual_qa.py',
             ROOT/'outputs/评价准备/results/冻结边界.json',
             ROOT/'outputs/数据清洗/results/清洗长表.csv',
             ROOT/'outputs/评价准备/results/候选样本索引.csv']
    write_json(results/'复现清单.json',build_manifest(files,config['seed'],config,
        'python outputs/告警片段/run_episodes.py', ['numpy','pandas','matplotlib','Pillow']))
    generated = [p for p in output.rglob('*') if p.is_file() and p.suffix in ['.csv','.json','.png']
                 and p.name not in ['复现清单.json','输出哈希.json','交付文件哈希.json','规则.json']]
    write_json(results/'输出哈希.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(generated)})
    print(json.dumps(audit,ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    self_test() if args.self_test else run(args.output)
