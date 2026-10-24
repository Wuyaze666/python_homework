"""冻结方法后的主时间测试、固定配置LOCO与片段检出评价。"""
from pathlib import Path
import argparse
import hashlib
import sys
import numpy as np
import pandas as pd
import joblib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'outputs/特征扩展'))
sys.path.insert(0,str(ROOT/'outputs/建模实验'))
from run_baseline import load_dev,evaluate_frame,features,predictions,train_value_evidence
from model_core import FEATURES,build_models,fit_baselines,metrics
from run_episodes import sha,read_json,write_json,csv,check


def verify_all():
    dev,verified=load_dev();extra=[]
    for folder in ['建模实验','特征扩展']:
        base=ROOT/'outputs'/folder
        for name,h in read_json(base/'交付文件哈希.json').items():
            actual=sha(base/name);check(actual==h,'归档变化:'+folder+'/'+name)
            extra.append(dict(path=str(base/name),expected=h,actual=actual))
    return dev,pd.concat([verified,pd.DataFrame(extra)],ignore_index=True)


def freeze():
    cfg=read_json(HERE/'冻结方案.json')
    recommend=read_json(ROOT/'outputs/特征扩展/results/开发候选推荐.json')
    check(cfg['primary_model']==recommend['model'] and cfg['feature_columns']==FEATURES,'冻结选择不符')
    paths=['outputs/特征扩展/models/推荐候选.joblib','outputs/特征扩展/results/开发候选推荐.json',
        'outputs/建模实验/参数.json','outputs/建模实验/models/decision_tree_A.joblib',
        'outputs/建模实验/models/random_forest_A.joblib','outputs/建模实验/results/多数类参照参数.json',
        'outputs/评价准备/results/冻结边界.json','outputs/评价准备/results/留一车辆预登记.csv',
        'outputs/评价准备/results/候选样本索引.csv']
    record=dict(date='2026-10-23',development_commit='5d697af',config_sha256=sha(HERE/'冻结方案.json'),
        frozen_files={p:sha(ROOT/p) for p in paths},primary_parameters=recommend['parameters'],
        primary_feature_group='A',test_evaluated_before_freeze=False,train_validation_refit=False)
    path=HERE/'冻结登记.json'
    if path.exists():check(read_json(path)==record,'冻结登记变化；禁止覆盖后选择')
    else:write_json(path,record)
    print('PASS: frozen config/model/boundaries; no test evaluation')


def load_all():
    # Called only after the freeze record is checked and after P1.
    dev,verified=verify_all()
    index=pd.read_csv(ROOT/'outputs/评价准备/results/候选样本索引.csv')
    cols=['source_row_id','vehicle','segment_id','time_local','variable_key','value_type','value_num',
          'alarm_class','high_value_review','protocol_value_review','numeric_parse_failure']
    ids=set(index.source_row_id);parts=[]
    for chunk in pd.read_csv(ROOT/'outputs/数据清洗/results/清洗长表.csv',usecols=cols,chunksize=20000):
        parts.append(chunk[chunk.source_row_id.isin(ids)])
    frame=index.merge(pd.concat(parts),on=['source_row_id','vehicle','segment_id','time_local'],validate='one_to_one')
    mapping=pd.read_csv(ROOT/'outputs/告警片段/results/模型候选片段索引.csv',usecols=['source_row_id','episode_id'])
    frame=frame.merge(mapping,on='source_row_id',validate='one_to_one').sort_values('source_row_id').reset_index(drop=True)
    check(frame.groupby('split').size().to_dict()=={'test':19662,'train':52892,'validation':10909},'冻结数量变化')
    check(frame.variable_key.notna().all() and frame.source_row_id.is_unique,'来源不完整')
    check(frame.groupby('segment_id').split.nunique().le(1).all(),'片段跨分区')
    check(frame[frame.episode_id.notna()].groupby('episode_id').split.nunique().le(1).all(),'告警片段跨分区')
    return frame,verified


def episode_metrics(prediction):
    positive=prediction[prediction.alarm_class.ne(0)].sort_values(['time_local','source_row_id'])
    check(positive.episode_id.notna().all(),'非零标签未映射片段')
    rows=[]
    for episode,g in positive.groupby('episode_id',sort=True):
        check(g.alarm_class.nunique()==1 and g.vehicle.nunique()==1 and g.variable_key.nunique()==1,'片段身份冲突')
        true=int(g.alarm_class.iloc[0]);any_alarm=g.predicted_class.ne(0);correct=g.predicted_class.eq(true)
        times=pd.to_datetime(g.time_local);first=times.iloc[0]
        rows.append(dict(episode_id=episode,vehicle=g.vehicle.iloc[0],segment_id=g.segment_id.iloc[0],
            variable_key=g.variable_key.iloc[0],alarm_class=true,record_count=len(g),
            first_source_row_id=int(g.source_row_id.iloc[0]),last_source_row_id=int(g.source_row_id.iloc[-1]),
            first_observed=g.time_local.iloc[0],last_observed=g.time_local.iloc[-1],
            observed_span_seconds=(times.iloc[-1]-first).total_seconds(),
            any_alarm_detected=bool(any_alarm.any()),any_correct_class=bool(correct.any()),
            all_normal_miss=bool(not any_alarm.any()),no_correct_class=bool(not correct.any()),
            onset_alarm=bool(any_alarm.iloc[0]),onset_correct=bool(correct.iloc[0]),
            alarm_record_fraction=float(any_alarm.mean()),correct_record_fraction=float(correct.mean()),
            first_alarm_delay_seconds=(times[any_alarm].iloc[0]-first).total_seconds() if any_alarm.any() else np.nan,
            first_correct_delay_seconds=(times[correct].iloc[0]-first).total_seconds() if correct.any() else np.nan,
            length_bin='1' if len(g)==1 else '2-5' if len(g)<=5 else '6+'))
    detail=pd.DataFrame(rows);summary=[]
    for vehicle in ['ALL','V1','V2','V3']:
        g=detail if vehicle=='ALL' else detail[detail.vehicle.eq(vehicle)]
        for cls in ['ALL',1,2,3,4,5]:
            h=g if cls=='ALL' else g[g.alarm_class.eq(cls)]
            for length in ['ALL','1','2-5','6+']:
                a=h if length=='ALL' else h[h.length_bin.eq(length)]
                row=dict(vehicle=vehicle,alarm_class=str(cls),length_bin=length,episodes=len(a),records=int(a.record_count.sum()))
                for field in ['any_alarm_detected','any_correct_class','all_normal_miss','no_correct_class','onset_alarm','onset_correct']:
                    row[field+'_n']=int(a[field].sum());row[field+'_rate']=a[field].mean() if len(a) else np.nan
                summary.append(row)
    return detail,pd.DataFrame(summary)


def self_test():
    # A positive episode with only wrong nonzero predictions is detected, but not correct-class detected.
    frame=pd.DataFrame(dict(source_row_id=[1,2,3,4],vehicle=['V1']*4,segment_id=['s']*4,
        variable_key=['x']*4,time_local=['2020-01-01 00:00:00','2020-01-01 00:00:05','2020-01-01 00:00:10','2020-01-01 00:00:12'],
        alarm_class=[2,2,3,4],predicted_class=[0,2,1,0],episode_id=['a','a','b','c']))
    d,_=episode_metrics(frame);d=d.set_index('episode_id')
    check(d.loc['a','first_alarm_delay_seconds']==5 and not d.loc['a','onset_alarm'],'片段延迟/起点')
    check(d.loc['b','any_alarm_detected'] and d.loc['b','no_correct_class'],'错非零类混淆口径')
    check(d.loc['c','all_normal_miss'] and pd.isna(d.loc['c','first_alarm_delay_seconds']),'漏检不应当0延迟')
    cfg=read_json(ROOT/'outputs/建模实验/参数.json')
    train=pd.DataFrame(dict(variable_key=['x']*12,value_type=['continuous']*12,value_num=list(range(12)),alarm_class=[0,1]*6))
    model=build_models(cfg,True)['random_forest_A'];model.fit(features(train),train.alarm_class)
    pred,p=predictions('test',model,train,None)
    check(p.shape==(12,6) and (p[:,2:]==0).all(),'缺训练类别概率映射')
    print('PASS: episode onset/delay/wrong-class/missed, absent training classes')


def smoke():
    dev,verified=verify_all();cfg=read_json(ROOT/'outputs/建模实验/参数.json')
    for held in ['V1','V2','V3']:
        # Dev only, even held-car diagnostic rows; no reserved test is read into this smoke.
        train=dev[dev.vehicle.ne(held)&dev.split.eq('train')].groupby('alarm_class').head(40)
        held_dev=dev[dev.vehicle.eq(held)].groupby('alarm_class').head(10).copy();held_dev['split']='smoke'
        baseline=fit_baselines(train)
        models={'global_majority':None,'variable_majority':None}|build_models(cfg,True)
        for name,model in models.items():
            if model is not None:model.fit(features(train),train.alarm_class)
            pred,p=predictions(name,model,held_dev,baseline);evaluate_frame(held_dev,pred,name)
        check(not set(train.source_row_id)&set(held_dev.source_row_id),'留车泄漏')
    print('PASS: 3 development-only small LOCO chains; reserved test evaluated=0; verified',len(verified))


def run(output):
    check((HERE/'冻结登记.json').exists(),'必须先冻结并归档方法')
    freeze()  # Check byte identity, never overwrite an existing freeze.
    cfg=read_json(HERE/'冻结方案.json');frame,verified=load_all()
    output=output.resolve();check(output==HERE or (output.is_relative_to(ROOT/'work') and output!=ROOT/'work'),'输出路径非法')
    for sub in ['results','diagnostics','models']:(output/sub).mkdir(parents=True,exist_ok=True)
    results=output/'results';tables={};summaries=[];details=[];matrices=[];binaries=[];predrows=[];members=[];fits=[]

    def evaluate(scope,name,model,target,baseline,training):
        pred,prob=predictions(name,model,target,baseline)
        sm,cl,cm,bi=evaluate_frame(target,pred,name)
        for sink,table in [(summaries,sm),(details,cl),(matrices,cm),(binaries,bi)]:sink.append(table.assign(scope=scope))
        _,evidence=train_value_evidence(training,target)
        out=target.copy();out['scope']=scope;out['model']=name;out['predicted_class']=pred;out['correct']=pred==out.alarm_class.to_numpy()
        for c in range(6):out['p_'+str(c)]=prob[:,c]
        out=out.merge(evidence.drop(columns=FEATURES),on='source_row_id',validate='one_to_one')
        predrows.append(out);return out

    train=frame[frame.split.eq('train')];test=frame[frame.split.eq('test')].copy()
    saved=ROOT/'outputs/建模实验'
    primary=joblib.load(ROOT/cfg['primary_artifact'])
    check(primary.named_steps['features'].columns==FEATURES,'主模型特征不是冻结A组')
    check(primary.named_steps['classifier'].get_params()==read_json(HERE/'冻结登记.json')['primary_parameters'],'主参数改变')
    main_models={'global_majority':None,'variable_majority':None,
        'decision_tree_A':joblib.load(saved/'models/decision_tree_A.joblib'),
        'RF_A_depth16':joblib.load(saved/'models/random_forest_A.joblib'),'RF_A_depth24':primary}
    baselines=read_json(saved/'results/多数类参照参数.json')
    for name,model in main_models.items():
        out=evaluate('main_temporal',name,model,test,baselines,train)
        if name==cfg['primary_model']:primary_predictions=out
    csv(results/'主测试预测.csv',pd.concat(predrows,ignore_index=True))
    # Main results saved before any LOCO fitting; nothing feeds back into selection.
    ep,ep_summary=episode_metrics(primary_predictions)
    full_episodes=pd.read_csv(ROOT/'outputs/告警片段/results/告警观测片段_全量.csv')
    expected=full_episodes[full_episodes.split.eq('test')&full_episodes.identity_status.eq('known_variable')]
    check(set(ep.episode_id)==set(expected.episode_id),'测试片段覆盖不全')
    match=ep.merge(expected[['episode_id','record_count','first_source_row_id','last_source_row_id']],on='episode_id',suffixes=('','_expected'),validate='one_to_one')
    for col in ['record_count','first_source_row_id','last_source_row_id']:check(match[col].equals(match[col+'_expected']),'测试片段成员不完整')
    ep=ep.merge(expected[['episode_id','start_censored','end_censored','start_reason','end_reason']],on='episode_id',validate='one_to_one')
    tables['主测试片段检出']=ep;tables['主测试片段汇总']=ep_summary
    persegment=[]
    for segment,g in primary_predictions.groupby('segment_id'):
        sm,_=metrics(g.alarm_class,g.predicted_class)
        persegment.append(dict(vehicle=g.vehicle.iloc[0],segment_id=segment,first_observed=g.time_local.min(),last_observed=g.time_local.max(),**sm))
    tables['主测试分段指标']=pd.DataFrame(persegment)
    fixed_cfg=read_json(ROOT/'outputs/建模实验/参数.json')
    registered=pd.read_csv(ROOT/'outputs/评价准备/results/留一车辆预登记.csv')
    for held in cfg['loco']['held_vehicles']:
        scope='loco_'+held
        roles=np.where(frame.vehicle.eq(held),'test',np.where(frame.split.eq('train'),'train',np.where(frame.split.eq('validation'),'validation','unused')))
        membership=frame[['source_row_id','vehicle','split']].rename(columns={'split':'main_split'}).assign(held_vehicle=held,fold_role=roles)
        counts=membership.groupby(['held_vehicle','vehicle','main_split','fold_role']).size().rename('actual').reset_index()
        compare=registered[registered.held_vehicle.eq(held)].merge(counts,on=['held_vehicle','vehicle','main_split','fold_role'],how='left')
        check(compare.actual.fillna(0).eq(compare.records).all(),'LOCO预登记数量不一致')
        members.append(membership)
        fitting=frame[roles=='train'].copy();validation=frame[roles=='validation'].copy();testing=frame[roles=='test'].copy()
        check(fitting.vehicle.ne(held).all() and validation.vehicle.ne(held).all(),'留车进入开发')
        check(not set(fitting.source_row_id)&set(testing.source_row_id),'留车样本重叠')
        validation['split']='validation';testing['split']='test'
        state=fit_baselines(fitting);models=build_models(fixed_cfg)
        models={'global_majority':None,'variable_majority':None,'decision_tree_A':models['decision_tree_A'],'RF_A_depth16':models['random_forest_A']}
        for name,model in models.items():
            if model is not None:
                model.fit(features(fitting),fitting.alarm_class)
                path=output/'models'/(scope+'_'+name+'.joblib');joblib.dump(model,path,compress=3)
                check(np.allclose(joblib.load(path).predict_proba(features(validation)),model.predict_proba(features(validation))),'LOCO模型重载不一致')
            for target in [validation,testing]:evaluate(scope,name,model,target,state,fitting)
        fits.append(dict(scope=scope,held_vehicle=held,train_n=len(fitting),validation_n=len(validation),test_n=len(testing),
            unused_n=int((roles=='unused').sum()),train_classes='|'.join(map(str,sorted(fitting.alarm_class.unique()))),
            train_source_ids_sha256=hashlib.sha256(fitting.source_row_id.to_numpy(dtype='<i8').tobytes()).hexdigest(),selection='none_fixed_stage4'))
        if models['RF_A_depth16'] is not None:
            encoder=models['RF_A_depth16'].named_steps['features']
            write_json(results/(scope+'_训练预处理.json'),dict(medians=encoder.medians_,categories=[a.tolist() for a in encoder.encoder_.categories_],baselines=state))
        print(scope,'fit',len(fitting),'validation',len(validation),'held test',len(testing),flush=True)
    allpred=pd.concat(predrows,ignore_index=True)
    tables.update({'指标汇总':pd.concat(summaries,ignore_index=True),'逐类指标':pd.concat(details,ignore_index=True),
        '混淆矩阵':pd.concat(matrices,ignore_index=True),'正常告警二分类指标':pd.concat(binaries,ignore_index=True),
        '跨车预测':allpred[allpred.scope.ne('main_temporal')], '跨车样本归属':pd.concat(members,ignore_index=True),
        '跨车训练登记':pd.DataFrame(fits),'输入核验':verified})
    tables['错误来源汇总']=allpred[~allpred.correct].groupby(['scope','model','split','vehicle','variable_key','alarm_class','predicted_class']).size().rename('errors').reset_index()
    tables['已见值错误分组']=allpred.groupby(['scope','model','split','vehicle','variable_seen_in_train','value_seen_in_train']).agg(n=('source_row_id','size'),errors=('correct',lambda s:int((~s).sum()))).reset_index()
    tables['质量标记错误分组']=allpred.assign(quality_review=allpred[['high_value_review','protocol_value_review','numeric_parse_failure']].any(axis=1)).groupby(['scope','model','split','vehicle','quality_review']).agg(n=('source_row_id','size'),errors=('correct',lambda s:int((~s).sum()))).reset_index()
    # Deterministic post-test explanation samples: first source row in each top error group per car.
    examples=[]
    for vehicle,g in primary_predictions[~primary_predictions.correct].groupby('vehicle'):
        counts=g.groupby(['variable_key','alarm_class','predicted_class']).size().rename('group_errors').reset_index().sort_values(
            ['group_errors','variable_key','alarm_class','predicted_class'],ascending=[False,True,True,True]).head(3)
        for row in counts.itertuples():
            chosen=g[(g.variable_key==row.variable_key)&(g.alarm_class==row.alarm_class)&(g.predicted_class==row.predicted_class)].sort_values('source_row_id').iloc[0].to_dict()
            examples.append(chosen|dict(group_errors=row.group_errors))
    tables['主测试错误案例']=pd.DataFrame(examples)
    for name,table in tables.items():csv(results/(name+'.csv'),table)
    finish(output)


def finish(output):
    """只重绘及封存已有完整计算表；不再次拟合或生成测试预测。"""
    output=output.resolve()
    check(output==HERE or (output.is_relative_to(ROOT/'work') and output!=ROOT/'work'),'输出路径非法')
    freeze();cfg=read_json(HERE/'冻结方案.json');results=output/'results'
    tables={name:pd.read_csv(results/(name+'.csv')) for name in ['指标汇总','混淆矩阵','主测试片段汇总']}
    check(len(pd.read_csv(results/'跨车训练登记.csv'))==3,'跨车计算未完成')
    primary=tables['指标汇总'].query("scope=='main_temporal' and model=='RF_A_depth24' and vehicle=='ALL'")
    check(len(primary)==1 and primary.n.iloc[0]==cfg['main_test_records'],'主测试结果不完整')
    from plot_final import plot_all
    plot_all(tables,output)
    write_json(results/'评价验收.json',dict(status='PASS',primary_model=cfg['primary_model'],main_test_n=int(primary.n.iloc[0]),
        primary_fit_n=52892,main_models=len(cfg['main_models']),loco_folds=3,loco_models_per_fold=4,
        main_test_episodes=len(pd.read_csv(results/'主测试片段检出.csv')),unknown_PID_excluded=True,freeze_matches=True,refit=False,
        post_test_tuning=False,loco_held_car_used_for_fit_or_selection=False,input_hashes=len(pd.read_csv(results/'输入核验.csv'))))
    from repro_manifest import build_manifest
    paths=[HERE/'run_final.py',HERE/'plot_final.py',HERE/'冻结方案.json',HERE/'冻结登记.json',
        ROOT/'outputs/数据清洗/results/清洗长表.csv',ROOT/'outputs/评价准备/results/候选样本索引.csv',ROOT/cfg['primary_artifact']]
    paths += [ROOT/'outputs'/folder/'交付文件哈希.json' for folder in ['数据清洗','评价准备','探索性分析','告警片段','建模实验','特征扩展']]
    write_json(results/'复现清单.json',build_manifest(paths,cfg['seed'],cfg,'python outputs/最终评价/run_final.py',
        ['numpy','pandas','scipy','scikit-learn','joblib','matplotlib']))
    files=[p for sub in ['results','models','diagnostics'] for p in (output/sub).rglob('*') if p.is_file() and p.name not in ['复现清单.json','输出哈希.json']]
    write_json(results/'输出哈希.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(files)})
    print(tables['指标汇总'].query("split=='test' and vehicle=='ALL'").to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--self-test',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--finalize-only',action='store_true')
    p.add_argument('--output',type=Path,default=HERE);a=p.parse_args()
    if a.freeze:freeze()
    elif a.self_test:self_test()
    elif a.smoke:smoke()
    elif a.finalize_only:finish(a.output)
    else:run(a.output)
