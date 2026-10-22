"""任务5唯一入口；有条件逐步扩展、有限调参，永不评价测试集。"""
from pathlib import Path
import argparse
import json
import sys
import hashlib
import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'outputs/建模实验'))
from run_baseline import load_dev, evaluate_frame
from model_core import FEATURES, metrics
from feature_core import RAW, build_states, build_history, columns_for, model_for
from run_episodes import sha,read_json,write_json,csv,check


def audit_sources(dev,provenance):
    ids=dev.set_index('source_row_id')
    p=provenance[provenance.source_row_id.notna()]
    target=ids.loc[p.target_row_id];source=ids.loc[p.source_row_id.astype(int)]
    delta=(pd.to_datetime(target.time_local).to_numpy()-pd.to_datetime(source.time_local).to_numpy())/np.timedelta64(1,'s')
    check(np.array_equal(target.vehicle,source.vehicle),'跨车来源')
    check(np.array_equal(target.segment_id,source.segment_id),'跨段来源')
    check(np.array_equal(target.split,source.split),'跨冻结分区来源')
    check(np.allclose(delta,p.age_seconds) and (delta>=0).all(),'未来来源或年龄错误')
    check(np.array_equal(source.variable_key,p.feature),'状态变量来源错误')
    check(p[p.status.eq('available')].eval('age_seconds<=ttl_seconds').all(),'过期有效状态')


def audit_history(dev,evidence):
    ids=dev.set_index('source_row_id');p=evidence[evidence.raw_count.gt(0)]
    t=ids.loc[p.target_row_id];first=ids.loc[p.first_source_row_id];last=ids.loc[p.last_source_row_id]
    for source in [first,last]:
        age=(pd.to_datetime(t.time_local).to_numpy()-pd.to_datetime(source.time_local).to_numpy())/np.timedelta64(1,'s')
        check((age>0).all() and (age<=p.window_seconds.to_numpy()).all(),'历史窗口越界')
        for col in ['vehicle','segment_id','split']:
            check(np.array_equal(t[col],source[col]),'历史跨'+col)
        check(np.array_equal(source.variable_key,p.feature),'历史变量错误')


def self_test():
    cfg=read_json(HERE/'参数.json')
    cfg['state_features']=['vehicle_speed'];cfg['history_continuous']=['vehicle_speed'];cfg['binary_features']=[]
    raw=pd.DataFrame([
        [1,'V1','s1','2020-01-01 00:00:00','vehicle_speed','continuous',1,0,0,0],
        [2,'V1','s1','2020-01-01 00:00:10','vehicle_speed','continuous',3,0,0,0],
        [3,'V1','s1','2020-01-01 00:00:20','x','continuous',7,0,0,0],
        [4,'V1','s1','2020-01-01 00:00:21','vehicle_speed','continuous',np.nan,0,0,1],
        [5,'V1','s1','2020-01-01 00:00:22','x','continuous',8,0,0,0],
        [6,'V1','s2','2020-01-01 00:00:23','x','continuous',9,0,0,0],
        [7,'V2','s3','2020-01-01 00:00:20','x','continuous',10,0,0,0],
        [8,'V1','s1','2020-01-01 00:00:40','vehicle_speed','continuous',999,0,0,0],
        [9,'V1','s1','2020-01-01 00:00:40','x','continuous',8,0,0,0]],columns=RAW)
    dic=pd.DataFrame({'variable_key':['vehicle_speed'],'ttl_seconds':[10]})
    x,p=build_states(raw,cfg,dic)
    check(x.loc[2,'b__vehicle_speed']==3,'TTL恰好等于10应保留')
    check(x.loc[[4,5,6],'b__vehicle_speed'].isna().all(),'最新无效/跨段/跨车阻断失败')
    strict,_=build_states(raw,cfg,dic,True)
    check(x.loc[8,'b__vehicle_speed']==999 and pd.isna(strict.loc[8,'b__vehicle_speed']),'同刻其他变量敏感性错误')
    h,ev=build_history(raw,cfg,30)
    check(h.loc[7,'c30__vehicle_speed__mean']==3,'历史边界应含t-30排除t')
    check(h.loc[7,'c30__vehicle_speed__n_invalid']==1,'无效历史遗漏')
    check(h.loc[2,'c30__vehicle_speed__rate']==0.2,'真实时间变化率错误')
    try:build_states(raw.assign(alarm_class=5),cfg,dic)
    except ValueError:pass
    else:raise AssertionError('历史接受标签')
    m=model_for('B_age',cfg,smoke=True)
    xx=pd.concat([x]*6,ignore_index=True);yy=np.repeat(np.arange(6),len(x))
    m.fit(xx[columns_for('B_age',cfg)],yy)
    enc=m.named_steps['features'];med=enc.medians_.copy()
    changed=x.copy();changed['b__vehicle_speed']=1e8
    m.predict(changed[columns_for('B_age',cfg)])
    check(enc.medians_.equals(med),'验证值修改训练中位数')
    print('PASS: TTL, invalid latest, same car/segment, simultaneous batch, strict history boundary, label whitelist, train-only encoder')


def run(output,smoke=False):
    cfg=read_json(HERE/'参数.json');dev,verified=load_dev()
    extra=[]
    for name,digest in read_json(ROOT/'outputs/建模实验/交付文件哈希.json').items():
        path=ROOT/'outputs/建模实验'/name
        check(sha(path)==digest,'任务4归档变化:'+name)
        extra.append(dict(path=str(path),expected=digest,actual=digest))
    verified=pd.concat([verified,pd.DataFrame(extra)],ignore_index=True)
    output=output.resolve()
    check(output==HERE or (output.is_relative_to(ROOT/'work') and output!=ROOT/'work'),'输出路径不得覆盖既有阶段')
    if smoke:
        dev=pd.concat([g.groupby('alarm_class').head(n) for s,n in [('train',60),('validation',15)]
                       for g in [dev[dev.split.eq(s)]]]).sort_values('source_row_id').reset_index(drop=True)
    else:
        output.mkdir(exist_ok=True,parents=True)
        for sub in ['results','models','diagnostics']:(output/sub).mkdir(exist_ok=True)
        write_json(output/'results/实验预登记.json',dict(config_sha256=sha(HERE/'参数.json'),config=cfg,
            train_rows=52892,validation_rows=10909,test_rows_evaluated=0))
    dictionary=pd.read_csv(ROOT/'outputs/数据清洗/results/变量字典.csv')
    raw=dev[RAW].copy()
    x,provenance=build_states(raw,cfg,dictionary);audit_sources(dev,provenance)
    train=dev.split.eq('train');val=dev.split.eq('validation')
    summaries=[];classes=[];confusions=[];binaries=[];predictions=[];ledger=[];decisions=[]
    fitted={};groups={};stats={};history_evidence=[]

    def experiment(name,group,kind='rf',overrides=None,seed=None,selectable=True):
        model=model_for(group,cfg,kind,overrides,seed,smoke)
        columns=columns_for(group,cfg)
        model.fit(x.loc[train,columns],dev.loc[train,'alarm_class'])
        for mask in [train,val]:
            prob=model.predict_proba(x.loc[mask,columns]);pred=model.classes_[prob.argmax(axis=1)]
            sm,cl,cm,bi=evaluate_frame(dev.loc[mask],pred,name)
            summaries.append(sm);classes.append(cl);confusions.append(cm);binaries.append(bi)
            if mask is val:
                out=dev.loc[val,['source_row_id','vehicle','segment_id','time_local','variable_key','value_num',
                    'alarm_class','episode_id']+RAW[-3:]].copy()
                out['model']=name;out['predicted_class']=pred;out['correct']=pred==out.alarm_class.to_numpy()
                for c in range(6):out['p_'+str(c)]=prob[:,list(model.classes_).index(c)] if c in model.classes_ else 0
                predictions.append(out)
                by={r.vehicle:r for r in sm.itertuples()}
                normal=cl[(cl.vehicle=='ALL')&(cl.alarm_class==0)].recall.iloc[0]
                stats[name]=dict(macro=by['ALL'].macro_f1_6,accuracy=by['ALL'].accuracy,
                    v1=by['V1'].macro_f1_supported,v2=by['V2'].macro_f1_supported,normal=normal,
                    binary_errors=int((~out.correct & out.variable_key.isin(cfg['binary_features'])).sum()))
        est=model.named_steps['classifier'];trees=est.estimators_ if kind=='rf' else [est]
        nodes=sum(t.tree_.node_count for t in trees)
        ledger.append(dict(model=name,group=group,kind=kind,seed=cfg['seed'] if seed is None else seed,
            selectable=selectable,encoded_features=len(model.named_steps['features'].get_feature_names_out()),
            total_nodes=nodes,parameters=json.dumps(est.get_params(),sort_keys=True),**stats[name]))
        fitted[name]=model;groups[name]=group
        if not smoke:print(name,json.dumps(stats[name]),flush=True)
        return name

    def admissible(child,parent):
        c,p=stats[child],stats[parent]
        return c['normal']>=p['normal']-cfg['normal_recall_drop_max'] and all(
            c[k]>=p[k]-cfg['vehicle_macro_drop_max'] for k in ['v1','v2'])

    def accept(child,parent):
        gain=stats[child]['macro']-stats[parent]['macro']
        passed=gain>=cfg['feature_gain_min'] and admissible(child,parent)
        decisions.append(dict(child=child,parent=parent,macro_gain=gain,guardrails_pass=admissible(child,parent),accepted=passed))
        return passed

    base=experiment('RF_A','A')
    if not smoke:
        old=pd.read_csv(ROOT/'outputs/建模实验/results/验证预测.csv')
        old=old[old.model.eq('random_forest_A')].sort_values('source_row_id')
        now=predictions[-1].sort_values('source_row_id')
        check(np.array_equal(old.source_row_id,now.source_row_id),'A复算样本不等')
        check(np.array_equal(old.predicted_class,now.predicted_class),'A复算预测变化')
        check(np.allclose(old[['p_'+str(c) for c in range(6)]],now[['p_'+str(c) for c in range(6)]]),'A复算概率变化')
    experiment('RF_A_unweighted','A',overrides={'class_weight':None})
    b=experiment('RF_B_values','B_values')
    experiment('DT_B_values','B_values',kind='tree')
    ba=experiment('RF_B_age','B_age')
    chosen=base;accepted_groups={'A'}
    if accept(b,base):
        chosen=b;accepted_groups.add('B_values')
    if accept(ba,chosen):
        chosen=ba;accepted_groups.add('B_age')
    c_gate=chosen!=base and stats[chosen]['binary_errors']>=cfg['c_gate_min_binary_errors']
    if c_gate:
        parent_group=groups[chosen]
        for window in cfg['history_windows']:
            h,ev=build_history(raw,cfg,window);audit_history(dev,ev)
            x=pd.concat([x,h],axis=1);history_evidence.append(ev)
            group=f'C{window}'+('_values' if parent_group=='B_values' else '')
            name=experiment('RF_'+group,group)
            if accept(name,chosen):
                chosen=name;accepted_groups.add(group)
            else:break
    tuning_group=groups[chosen]
    for suffix,change in [('unweighted',{'class_weight':None}),('leaf10',{'min_samples_leaf':10}),('depth24',{'max_depth':24})]:
        name='RF_'+tuning_group+'_'+suffix
        if name not in fitted:experiment(name,tuning_group,overrides=change)
    # Trees are interpretation comparators. Development recommendation chosen from RF only.
    eligible=[r for r in ledger if r['kind']=='rf' and r['group'] in accepted_groups and admissible(r['model'],base)]
    best=max(r['macro'] for r in eligible)
    near=[r for r in eligible if r['macro']>=best-cfg['tie_macro_tolerance']]
    candidate=min(near,key=lambda r:(r['total_nodes'],r['encoded_features'],r['model']))['model']
    if smoke:
        print(json.dumps(dict(status='PASS',train=int(train.sum()),validation=int(val.sum()),test=0,
            experiments=len(ledger),candidate=candidate,c_gate=c_gate),ensure_ascii=False));return
    candidate_params=fitted[candidate].named_steps['classifier'].get_params()
    for seed in cfg['stability_seeds']:
        experiment(f'RF_A_seed{seed}','A',seed=seed,selectable=False)
        overrides={k:candidate_params[k] for k in cfg['random_forest']}
        experiment(f'candidate_seed{seed}',groups[candidate],overrides=overrides,seed=seed,selectable=False)
    # Validate inference robustness on the same records; these are not additional fits or selection scores.
    sensitivity=[];model=fitted[candidate];cols=columns_for(groups[candidate],cfg)
    valid_x=x.loc[val,cols].copy();val_ids=dev.loc[val,'source_row_id'].to_numpy()
    for policy in ['original','age_le_2','mask_review','strict_other_timestamp']:
        altered=valid_x.copy()
        if policy in ['age_le_2','mask_review']:
            for f in cfg['state_features']:
                if 'b__'+f not in cols:continue
                p=provenance[provenance.feature.eq(f)].set_index('target_row_id').loc[val_ids]
                bad=p.age_seconds.gt(2).to_numpy() if policy=='age_le_2' else p.quality_review.to_numpy()
                for col in ['b__'+f,'age__'+f]:
                    if col in cols:altered.loc[altered.index[bad],col]=np.nan
            if policy=='mask_review':
                for ev in history_evidence:
                    w=int(ev.window_seconds.iloc[0])
                    for f in cfg['history_continuous']+cfg['binary_features']:
                        bad=ev[ev.feature.eq(f)].set_index('target_row_id').loc[val_ids].quality_review.to_numpy()
                        hc=[c for c in cols if c.startswith(f'c{w}__{f}__')]
                        altered.loc[altered.index[bad],hc]=np.nan
        elif policy=='strict_other_timestamp':
            strict,p=build_states(raw,cfg,dictionary,True);audit_sources(dev,p)
            bc=[c for c in cols if c.startswith(('b__','age__'))]
            altered.loc[:,bc]=strict.loc[val,bc]
        pred=model.predict(altered)
        sm,_,_,_=evaluate_frame(dev.loc[val],pred,policy)
        sensitivity.append(sm)
    tables={'实验台账':pd.DataFrame(ledger),'特征增量决策':pd.DataFrame(decisions),
        '指标汇总':pd.concat(summaries,ignore_index=True),'逐类指标':pd.concat(classes,ignore_index=True),
        '混淆矩阵':pd.concat(confusions,ignore_index=True),'正常告警二分类指标':pd.concat(binaries,ignore_index=True),
        '验证预测':pd.concat(predictions,ignore_index=True),'敏感性指标':pd.concat(sensitivity,ignore_index=True),
        '状态来源':provenance,'输入核验':verified}
    pp=tables['验证预测']
    tables['验证混淆来源']=pp[~pp.correct].groupby(['model','vehicle','variable_key','alarm_class','predicted_class']).size().rename('errors').reset_index()
    tables['验证疑点分组']=pp.assign(quality_review=pp[RAW[-3:]].any(axis=1)).groupby(['model','vehicle','quality_review']).agg(
        n=('source_row_id','size'),errors=('correct',lambda s:int((~s).sum()))).reset_index()
    tables['状态覆盖']=provenance.merge(dev[['source_row_id','split']],left_on='target_row_id',right_on='source_row_id',suffixes=('','_target')).groupby(
        ['split','vehicle','feature','status']).size().rename('n').reset_index()
    tables['开发特征']=pd.concat([dev[['source_row_id','split']],x],axis=1)
    if history_evidence:tables['历史来源']=pd.concat(history_evidence,ignore_index=True)
    # Audit tree and all experiment metadata saved, only recommended binary model retained.
    path=output/'models/推荐候选.joblib';joblib.dump(model,path,compress=3)
    check(np.allclose(joblib.load(path).predict_proba(valid_x),model.predict_proba(valid_x)),'候选重载失败')
    enc=model.named_steps['features']
    write_json(output/'results/候选预处理.json',dict(a_medians=enc.base_.medians_,context_medians=enc.medians_.to_dict(),
        a_categories=[a.tolist() for a in enc.base_.encoder_.categories_],
        state_categories=[] if enc.encoder_ is None else [a.tolist() for a in enc.encoder_.categories_],
        train_ids_sha256=hashlib.sha256(dev.loc[train,'source_row_id'].to_numpy(dtype='<i8').tobytes()).hexdigest()))
    tables['候选特征字典']=pd.DataFrame({'encoded_feature':enc.get_feature_names_out()})
    recommendation=dict(model=candidate,group=groups[candidate],parameters=candidate_params,input_columns=cols,
        eligible_models=[r['model'] for r in eligible],accepted_groups=sorted(accepted_groups),best_validation_macro=best,tie_tolerance=cfg['tie_macro_tolerance'],
        selection_reason='eligible RF; within0.002 of highest validation macro prefer fewer tree nodes then encoded features then model id',
        final_method_frozen=False,test_evaluated=False,refit_train_validation=False,
        c_gate=c_gate,history_windows_built=[int(e.window_seconds.iloc[0]) for e in history_evidence])
    write_json(output/'results/开发候选推荐.json',recommendation)
    for name,frame in tables.items():csv(output/'results'/(name+'.csv'),frame)
    from plot_extension import plot_all
    plot_all(tables,output,candidate)
    write_json(output/'results/阶段验收.json',dict(status='PASS',train_rows=int(train.sum()),validation_rows=int(val.sum()),
        test_rows_evaluated=0,reserved_test_rows=19662,input_hashes=len(verified),experiments=len(ledger),
        state_matches=len(provenance),history_windows=recommendation['history_windows_built'],
        source_identity_time_partition_audit=True,A_replay_matches=True,reload_matches=True,
        candidate=candidate,final_model_selected=False))
    from repro_manifest import build_manifest
    paths=[HERE/'run_extension.py',HERE/'feature_core.py',HERE/'plot_extension.py',HERE/'参数.json',
        ROOT/'requirements.txt',ROOT/'outputs/建模实验/交付文件哈希.json',ROOT/'outputs/数据清洗/results/清洗长表.csv',
        ROOT/'outputs/评价准备/results/候选样本索引.csv',ROOT/'outputs/建模实验/model_core.py',ROOT/'outputs/建模实验/run_baseline.py']
    write_json(output/'results/复现清单.json',build_manifest(paths,cfg['seed'],cfg,
        'python outputs/特征扩展/run_extension.py',['numpy','pandas','scipy','scikit-learn','joblib','matplotlib']))
    generated=[p for sub in ['results','diagnostics','models'] for p in (output/sub).rglob('*')
        if p.is_file() and p.name not in ['复现清单.json','输出哈希.json']]
    write_json(output/'results/输出哈希.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(generated)})
    print(json.dumps(recommendation,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',type=Path,default=HERE);a=p.parse_args()
    if a.self_test:self_test()
    else:run(a.output,a.smoke)
