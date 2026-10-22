"""阶段4：固定A组基线，只训练/验证，不运行最终测试。"""
from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.tree import export_text

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'outputs/告警片段'))
sys.path.insert(0,str(ROOT/'outputs/探索性分析/utils'))
from run_episodes import sha, read_json, write_json, csv, check, verify_inputs
from model_core import FEATURES, LABELS, AEncoder, build_models, fit_baselines, baseline_proba, metrics


def load_dev():
    verified=verify_inputs()
    extra=[]
    base=ROOT/'outputs/告警片段'
    for name,digest in read_json(base/'交付文件哈希.json').items():
        actual=sha(base/name)
        check(actual==digest,f'告警片段归档改变: {name}')
        extra.append(dict(path=str(base/name),expected=digest,actual=actual))
    verified=pd.concat([verified,pd.DataFrame(extra)],ignore_index=True)
    frozen=pd.read_csv(ROOT/'outputs/评价准备/results/候选样本索引.csv')
    check(frozen.groupby('split').size().to_dict()=={'test':19662,'train':52892,'validation':10909},'冻结数量变化')
    devindex=frozen[frozen.split.isin(['train','validation'])].copy()
    ids=set(devindex.source_row_id)
    columns=['source_row_id','vehicle','segment_id','time_local','variable_key','value_type',
             'value_num','alarm_class','high_value_review','protocol_value_review','numeric_parse_failure']
    parts=[]
    for chunk in pd.read_csv(ROOT/'outputs/数据清洗/results/清洗长表.csv',usecols=columns,chunksize=20000):
        parts.append(chunk[chunk.source_row_id.isin(ids)])
    raw=pd.concat(parts,ignore_index=True)
    dev=devindex.merge(raw,on=['source_row_id','vehicle','segment_id','time_local'],validate='one_to_one',how='left')
    check(len(dev)==63801 and dev.variable_key.notna().all(),'开发记录来源连接失败')
    check(dev.source_row_id.is_unique and dev.alarm_class.isin(LABELS).all(),'ID或标签非法')
    check(not set(dev.source_row_id)&set(frozen.loc[frozen.split.eq('test'),'source_row_id']),'测试记录进入开发')
    # 使用开发范围片段ID只供错误追溯，不作为特征。
    episode_parts=[]
    for chunk in pd.read_csv(ROOT/'outputs/告警片段/results/模型候选片段索引.csv',
                             usecols=['source_row_id','episode_id'],chunksize=20000):
        episode_parts.append(chunk[chunk.source_row_id.isin(ids)])
    dev=dev.merge(pd.concat(episode_parts),on='source_row_id',validate='one_to_one')
    check(dev.groupby('segment_id').split.nunique().le(1).all(),'完整观测片段跨分区')
    check(dev[dev.episode_id.notna()].groupby('episode_id').split.nunique().le(1).all(),'告警片段跨分区')
    return dev.sort_values('source_row_id').reset_index(drop=True),verified


def features(frame):
    return frame.loc[:,FEATURES].copy()


def predictions(model_name,model,frame,baselines):
    if model is None:
        probability=baseline_proba(model_name,frame,baselines)
    else:
        raw=model.predict_proba(features(frame))
        probability=np.zeros((len(frame),6),dtype=float)
        probability[:,model.classes_.astype(int)]=raw
    check(np.isfinite(probability).all() and np.allclose(probability.sum(axis=1),1),'概率非法')
    pred=probability.argmax(axis=1)
    return pred,probability


def evaluate_frame(frame,pred,model_name):
    summaries,details,confusions,binaries=[],[],[],[]
    for vehicle in ['ALL','V1','V2','V3']:
        mask=np.ones(len(frame),dtype=bool) if vehicle=='ALL' else frame.vehicle.eq(vehicle).to_numpy()
        g=frame.loc[mask]; p=pred[mask]
        summary,detail=metrics(g.alarm_class,p)
        meta=dict(model=model_name,split=frame.split.iloc[0],vehicle=vehicle)
        summaries.append(meta|summary)
        details.append(detail.assign(**meta))
        matrix=confusion_matrix(g.alarm_class,p,labels=LABELS)
        for i,j in np.ndindex(6,6):
            confusions.append(meta|dict(true_class=i,predicted_class=j,records=int(matrix[i,j])))
        _,binary=metrics(g.alarm_class.ne(0).astype(int), (p!=0).astype(int),labels=[0,1])
        binaries.append(binary.assign(**meta))
    return pd.DataFrame(summaries),pd.concat(details),pd.DataFrame(confusions),pd.concat(binaries)


def train_value_evidence(train,validation):
    columns=['variable_key','value_type','value_num']
    lookup=train.groupby(columns,dropna=False).agg(
        train_same_value_n=('alarm_class','size'),
        train_same_value_classes=('alarm_class',lambda s:'|'.join(map(str,sorted(s.unique())))),
        train_same_value_n_classes=('alarm_class','nunique')).reset_index()
    lookup['train_value_majority']=train.groupby(columns,dropna=False).alarm_class.agg(
        lambda s:int(np.bincount(s,minlength=6).argmax())).to_numpy()
    lookup['train_value_majority_count']=train.groupby(columns,dropna=False).alarm_class.agg(
        lambda s:int(np.bincount(s,minlength=6).max())).to_numpy()
    info=validation[['source_row_id']+columns].merge(lookup,on=columns,how='left',validate='many_to_one')
    info['variable_seen_in_train']=info.variable_key.isin(train.variable_key.unique())
    info['value_seen_in_train']=info.train_same_value_n.notna()
    return lookup,info


def fit_run(dev,config,smoke=False):
    train=dev[dev.split.eq('train')].copy()
    validation=dev[dev.split.eq('validation')].copy()
    if smoke:
        train=train.groupby('alarm_class',sort=True).head(60).copy()
        validation=validation.groupby('alarm_class',sort=True).head(15).copy()
    baselines=fit_baselines(train)
    models=build_models(config,smoke)
    for model in models.values():
        model.fit(features(train),train.alarm_class)
    model_map={'global_majority':None,'variable_majority':None}|models
    all_metrics,all_classes,all_confusion,all_binary,validation_predictions=[],[],[],[],[]
    for name,model in model_map.items():
        for frame in [train,validation]:
            pred,prob=predictions(name,model,frame,baselines)
            sm,cl,cm,bi=evaluate_frame(frame,pred,name)
            all_metrics.append(sm);all_classes.append(cl);all_confusion.append(cm);all_binary.append(bi)
            if frame.split.iloc[0]=='validation':
                out=frame[['source_row_id','vehicle','segment_id','time_local','variable_key','value_type',
                           'value_num','alarm_class','episode_id','high_value_review','protocol_value_review',
                           'numeric_parse_failure']].copy()
                out['model']=name;out['predicted_class']=pred;out['correct']=pred==frame.alarm_class.to_numpy()
                for c in LABELS:
                    out['p_'+str(c)]=prob[:,c]
                validation_predictions.append(out)
    tables={'指标汇总':pd.concat(all_metrics,ignore_index=True),
            '逐类指标':pd.concat(all_classes,ignore_index=True),
            '混淆矩阵':pd.concat(all_confusion,ignore_index=True),
            '正常告警二分类指标':pd.concat(all_binary,ignore_index=True),
            '验证预测':pd.concat(validation_predictions,ignore_index=True)}
    return train,validation,models,baselines,tables


def self_test():
    x=pd.DataFrame({'variable_key':['a','a','b','b','c','c'],
        'value_type':['continuous']*4+['code']*2,'value_num':[1,np.nan,100,300,0,255]})
    enc=AEncoder().fit(x)
    check(enc.medians_=={'a':1,'b':200},'中位数混合变量')
    v=pd.DataFrame({'variable_key':['b','unseen','c'],'value_type':['continuous','continuous','code'],
                    'value_num':[np.nan,999999,999]})
    matrix=enc.transform(v).toarray()
    check(matrix[0,-2]==200 and matrix[0,-1]==1,'缺失处理失败')
    check(matrix[2,-2]==0 and np.isfinite(matrix).all(),'状态代码被当连续值')
    check(enc.medians_=={'a':1,'b':200} and 'unseen' not in enc.encoder_.categories_[0], '验证拟合泄漏')
    caught=False
    try:enc.transform(v.assign(alarm_class=5))
    except ValueError:caught=True
    check(caught,'允许标签进入特征')
    state=fit_baselines(pd.DataFrame({'variable_key':['a','a','b','b'],'alarm_class':[1,0,2,2]}))
    pred=baseline_proba('variable_majority',pd.DataFrame({'variable_key':['a','z']}),state).argmax(axis=1)
    check(pred.tolist()==[0,2],'并列或未知变量回退失败')
    y=np.array([0,0,1,1,2,3,4,5]);p=np.array([0,1,1,2,2,3,4,0])
    sm,detail=metrics(y,p)
    check(np.isclose(sm['macro_f1_6'],f1_score(y,p,average='macro',labels=LABELS)),'F1计算错误')
    sm,detail=metrics([0,1],[0,0])
    check(pd.isna(sm['macro_f1_6']) and pd.isna(detail.loc[detail.alarm_class.eq(5),'recall']).all(),
          '无支持类别不应写成0%召回')
    check(metrics([],[])[0]['n']==0,'空验证车辆处理失败')
    config=read_json(HERE/'参数.json')
    synthetic=pd.concat([x]*10,ignore_index=True)
    target=np.tile([0,1,2,3,4,5],10)
    for name,model in build_models(config,smoke=True).items():
        model.fit(synthetic,target)
        check(model.predict_proba(v).shape==(3,6),'模型族最小链失败:'+name)
    print('PASS: preprocessing, unseen categories, leakage guard, baselines, missing-class metrics, both model families')


def run(output,smoke=False):
    config=read_json(HERE/'参数.json')
    check(config['feature_columns']==FEATURES and config['labels']==LABELS.tolist(),'特征/标签合同不一致')
    check(config['numeric_types']==['continuous','counter'] and config['categorical_types']==['binary','code'],
          '类型编码规则已预登记；更改需修改实现并另立版本')
    if not smoke:
        output=output.resolve()
        work=(ROOT/'work').resolve()
        check(output==HERE or (output.is_relative_to(work) and output!=work),'禁止覆盖既有归档')
        output.mkdir(parents=True,exist_ok=True)
        results=output/'results';results.mkdir(exist_ok=True)
        write_json(results/'实验预登记.json',dict(config_sha256=sha(HERE/'参数.json'),
            fit_scope='train',evaluation_scopes=['train_apparent_fit','validation'],test_evaluated=False,
            feature_columns=FEATURES,hyperparameter_search=False))
    dev,verified=load_dev()
    train,validation,models,baselines,tables=fit_run(dev,config,smoke)
    if smoke:
        print(json.dumps(dict(status='PASS',train_rows=len(train),validation_rows=len(validation),
              test_rows=0,model_names=list(models),input_hashes=len(verified))))
        return
    lookup,info=train_value_evidence(train,validation)
    vp=tables['验证预测']
    vp=vp.merge(info.drop(columns=['variable_key','value_type','value_num']),on='source_row_id',validate='many_to_one')
    tables['验证预测']=vp
    tables['验证错误索引']=vp[~vp.correct].copy()
    tables['训练同值标签证据']=lookup
    tables['验证变量误差']=vp.groupby(['model','vehicle','variable_key']).agg(
        n=('source_row_id','size'),correct=('correct','sum'),error_rate=('correct',lambda s:1-s.mean()),
        unseen_value_n=('value_seen_in_train',lambda s:int((~s).sum()))).reset_index()
    tables['验证混淆来源']=vp[~vp.correct].groupby(
        ['model','vehicle','variable_key','alarm_class','predicted_class']).size().rename('errors').reset_index()
    tables['验证疑点分组']=vp.assign(quality_review=vp[['high_value_review','protocol_value_review','numeric_parse_failure']].any(axis=1)).groupby(
        ['model','vehicle','quality_review']).agg(n=('source_row_id','size'),error_rate=('correct',lambda s:1-s.mean())).reset_index()
    tables['开发样本索引']=dev[['source_row_id','vehicle','segment_id','time_local','variable_key','value_type','split','alarm_class','episode_id']]
    support=[]
    for split in ['train','validation']:
        for vehicle in ['ALL','V1','V2','V3']:
            g=dev[dev.split.eq(split)]
            if vehicle!='ALL':g=g[g.vehicle.eq(vehicle)]
            support.extend(dict(split=split,vehicle=vehicle,alarm_class=int(c),records=int(g.alarm_class.eq(c).sum())) for c in LABELS)
    tables['开发类别支持']=pd.DataFrame(support)
    tables['输入核验']=verified
    encoder=models['decision_tree_A'].named_steps['features']
    names=encoder.get_feature_names_out()
    tables['编码特征字典']=pd.DataFrame({'feature_index':np.arange(len(names)),'encoded_feature':names})
    preprocess=dict(train_rows=len(train),features=FEATURES,medians=encoder.medians_,
        one_hot_categories=[a.tolist() for a in encoder.encoder_.categories_],
        fit_source_ids_sha256=__import__('hashlib').sha256(train.source_row_id.to_numpy(dtype='<i8').tobytes()).hexdigest(),
        units='raw; no cross-variable unit conversion; tree splits do not require standardization',
        quality_flags='audit only; not model inputs')
    write_json(results/'预处理训练参数.json',preprocess)
    write_json(results/'多数类参照参数.json',baselines)
    modeldir=output/'models';modeldir.mkdir(exist_ok=True)
    complexity=[]
    for name,model in models.items():
        check(model.named_steps['features'].get_feature_names_out().tolist()==names.tolist(),'两模型编码口径不同')
        path=modeldir/(name+'.joblib');joblib.dump(model,path,compress=3)
        loaded=joblib.load(path)
        check(np.allclose(loaded.predict_proba(features(validation)),model.predict_proba(features(validation))), '模型重载预测不同')
        estimator=model.named_steps['classifier']
        trees=[estimator] if name=='decision_tree_A' else estimator.estimators_
        complexity.append(dict(model=name,trees=len(trees),encoded_features=len(names),
            total_nodes=sum(t.tree_.node_count for t in trees),max_observed_depth=max(t.get_depth() for t in trees)))
    tables['模型规模']=pd.DataFrame(complexity)
    (results/'浅树规则.txt').write_text(export_text(models['decision_tree_A'].named_steps['classifier'],
         feature_names=names.tolist(),max_depth=8,decimals=5),encoding='utf-8')
    for name,frame in tables.items():csv(results/(name+'.csv'),frame)
    from plot_baseline import plot_all
    plot_all(tables,output)
    audit=dict(status='PASS',stage=4,train_rows=len(train),validation_rows=len(validation),test_rows_evaluated=0,
        candidate_rows=83463,test_candidates_reserved=19662,encoded_features=len(names),
        input_files_verified=len(verified),models_fitted=2,majority_baselines=2,hyperparameter_search=False,
        future_features=0,history_features=0,model_reload_predictions_match=True,
        forbidden_fields_in_features=[],validation_vehicles=['V1','V2'],V3_validation_rows=0,
        all_rows_retained=True,final_model_selected=False)
    write_json(results/'建模验收.json',audit)
    from repro_manifest import build_manifest
    paths=[HERE/'run_baseline.py',HERE/'model_core.py',HERE/'plot_baseline.py',HERE/'参数.json',
        ROOT/'requirements.txt',ROOT/'outputs/数据清洗/results/清洗长表.csv',
        ROOT/'outputs/评价准备/results/候选样本索引.csv',ROOT/'outputs/评价准备/results/冻结边界.json',
        ROOT/'outputs/告警片段/run_episodes.py',ROOT/'outputs/告警片段/交付文件哈希.json']
    paths.extend((ROOT/'outputs/探索性分析/utils'/p) for p in ['repro_manifest.py','plot_style.py','visual_qa.py','profile_data.py'])
    write_json(results/'复现清单.json',build_manifest(paths,config['seed'],config,
        'python outputs/建模实验/run_baseline.py',['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl','matplotlib','Pillow']))
    generated=[p for sub in ['results','diagnostics','models'] for p in (output/sub).rglob('*')
        if p.is_file() and p.name not in ['复现清单.json','输出哈希.json']]
    write_json(results/'输出哈希.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(generated)})
    print(json.dumps(audit,ensure_ascii=False))
    print(tables['指标汇总'].query("vehicle=='ALL'")[['model','split','n','accuracy','macro_f1_6']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',type=Path,default=HERE)
    a=p.parse_args()
    if a.self_test:self_test()
    else:run(a.output,a.smoke)
