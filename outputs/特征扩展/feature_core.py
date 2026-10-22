"""记录真实时刻的无标签上下文；训练专用预处理。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'outputs/建模实验'))
from model_core import AEncoder, FEATURES

RAW = ['source_row_id','vehicle','segment_id','time_local','variable_key','value_type','value_num',
       'high_value_review','protocol_value_review','numeric_parse_failure']


def check_raw(frame):
    if list(frame.columns) != RAW:
        raise ValueError('历史构建只接受无标签白名单原始列')
    if not frame.source_row_id.is_unique:
        raise ValueError('重复源行')


def build_states(raw, config, dictionary, strict_other=False):
    check_raw(raw)
    raw=raw.copy(); raw['time_local']=pd.to_datetime(raw.time_local).astype('datetime64[ns]')
    values=raw[FEATURES].copy(); provenance=[]
    for feature in config['state_features']:
        values['b__'+feature]=np.nan
        values['age__'+feature]=np.nan
    ttl_map=dictionary.set_index('variable_key').ttl_seconds.to_dict()
    for (vehicle, segment), a in raw.groupby(['vehicle','segment_id'],sort=True):
        a=a.sort_values(['time_local','source_row_id'])
        for feature in config['state_features']:
            source=a[a.variable_key.eq(feature)]
            n=len(a); source_ids=np.full(n,np.nan)
            val=np.full(n,np.nan); age=np.full(n,np.nan); flags=np.zeros(n,dtype=bool)
            times=a.time_local.astype('int64').to_numpy()
            if len(source):
                st=source.time_local.astype('int64').to_numpy()
                pos=np.searchsorted(st,times,side='right')-1
                if strict_other:
                    other=~a.variable_key.eq(feature).to_numpy()
                    pos[other]=np.searchsorted(st,times[other],side='left')-1
                found=pos>=0; take=source.iloc[np.maximum(pos,0)]
                val[found]=take.value_num.to_numpy()[found]
                source_ids[found]=take.source_row_id.to_numpy()[found]
                age[found]=(times[found]-take.time_local.astype('int64').to_numpy()[found])/1e9
                flags[found]=take[RAW[-3:]].any(axis=1).to_numpy()[found]
            ttl=ttl_map[feature]
            status=np.select([np.isnan(source_ids),age>ttl,np.isnan(val)],
                             ['no_history','expired','invalid_latest'],default='available')
            available=status=='available'
            values.loc[a.index,'b__'+feature]=np.where(available,val,np.nan)
            # Expired age is not supplied: bounded by the original TTL, not a segment clock.
            values.loc[a.index,'age__'+feature]=np.where(available,age,np.nan)
            provenance.append(pd.DataFrame(dict(target_row_id=a.source_row_id.to_numpy(),
                vehicle=vehicle,segment_id=segment,feature=feature,source_row_id=source_ids,
                age_seconds=age,ttl_seconds=ttl,status=status,quality_review=flags)))
    return values, pd.concat(provenance,ignore_index=True)


def build_history(raw, config, window):
    """[t-window,t)中真实原始观测。rate仅用至少两个有效时间点，单位原值/秒。"""
    check_raw(raw)
    raw=raw.copy();raw['time_local']=pd.to_datetime(raw.time_local).astype('datetime64[ns]')
    result=pd.DataFrame(index=raw.index); evidence=[]
    channels=config['history_continuous']+config['binary_features']
    for feature in channels:
        stats=['n_valid','n_invalid','last_age']
        stats+=['mean','std','min','max','rate'] if feature in config['history_continuous'] else ['changes']
        for stat in stats: result[f'c{window}__{feature}__{stat}']=np.nan
    for (vehicle,segment), a in raw.groupby(['vehicle','segment_id'],sort=True):
        a=a.sort_values(['time_local','source_row_id'])
        times=a.time_local.astype('int64').to_numpy()
        for feature in channels:
            source=a[a.variable_key.eq(feature)]
            st=source.time_local.astype('int64').to_numpy()
            sv=source.value_num.to_numpy(dtype=float)
            ids=source.source_row_id.to_numpy()
            flags=source[RAW[-3:]].any(axis=1).to_numpy()
            left=st.searchsorted(times-window*10**9,side='left')
            right=st.searchsorted(times,side='left')
            columns=[c for c in result if c.startswith(f'c{window}__{feature}__')]
            matrix=np.full((len(a),len(columns)),np.nan)
            for j,(lo,hi) in enumerate(zip(left,right)):
                v=sv[lo:hi];ts=st[lo:hi];valid=np.isfinite(v);vv=v[valid];tt=ts[valid]
                row=[len(vv),int((~valid).sum()),(times[j]-tt[-1])/1e9 if len(vv) else np.nan]
                if feature in config['history_continuous']:
                    row+=([vv.mean(),vv.std(ddof=0),vv.min(),vv.max(),
                           (vv[-1]-vv[0])/((tt[-1]-tt[0])/1e9) if len(vv)>=2 and tt[-1]>tt[0] else np.nan]
                          if len(vv) else [np.nan]*5)
                else:
                    # Do not bridge an invalid reading when counting changes.
                    row+=[int(np.sum(valid[1:] & valid[:-1] & (v[1:]!=v[:-1])))]
                matrix[j]=row
                evidence.append((int(a.source_row_id.iloc[j]),vehicle,segment,feature,window,hi-lo,
                    int(ids[lo]) if hi>lo else -1,int(ids[hi-1]) if hi>lo else -1,
                    (times[j]-st[hi-1])/1e9 if hi>lo else np.nan,bool(flags[lo:hi].any())))
            result.loc[a.index,columns]=matrix
    ev=pd.DataFrame(evidence,columns=['target_row_id','vehicle','segment_id','feature','window_seconds',
        'raw_count','first_source_row_id','last_source_row_id','last_raw_age','quality_review'])
    return result,ev


def columns_for(group, config):
    columns=FEATURES.copy()
    if group!='A':columns+=['b__'+f for f in config['state_features']]
    if group in ['B_age','C30','C60']:columns+=['age__'+f for f in config['state_features']]
    windows=[] if group not in ['C30','C60','C30_values','C60_values'] else ([30,60] if '60' in group else [30])
    for w in windows:
        for f in config['history_continuous']+config['binary_features']:
            stats=['n_valid','n_invalid','last_age']+(['mean','std','min','max','rate'] if f in config['history_continuous'] else ['changes'])
            columns += [f'c{w}__{f}__{s}' for s in stats]
    return columns


class ContextEncoder(TransformerMixin,BaseEstimator):
    def __init__(self, columns, binary_columns):
        self.columns=columns;self.binary_columns=binary_columns

    def _check(self,x):
        if list(x.columns)!=list(self.columns):raise ValueError('入模列不符合白名单')

    def _categorical(self,x):
        return x[self.binary_columns].map(lambda v:'missing' if pd.isna(v) else format(float(v),'.17g'))

    def fit(self,x,y=None):
        self._check(x);self.base_=AEncoder().fit(x[FEATURES])
        self.numeric_=[c for c in self.columns if c not in FEATURES+self.binary_columns]
        self.medians_=x[self.numeric_].median().fillna(0)
        self.encoder_=None
        if self.binary_columns:
            self.encoder_=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32).fit(self._categorical(x))
        return self

    def transform(self,x):
        self._check(x)
        pieces=[self.base_.transform(x[FEATURES])]
        if self.encoder_ is not None:pieces.append(self.encoder_.transform(self._categorical(x)))
        if self.numeric_:
            vals=x[self.numeric_]
            pieces.append(sparse.csr_matrix(np.column_stack([vals.fillna(self.medians_),vals.isna().astype(float)]).astype(np.float32)))
        result=sparse.hstack(pieces,format='csr')
        if not np.isfinite(result.data).all():raise ValueError('编码非有限')
        return result

    def get_feature_names_out(self,input_features=None):
        names=list(self.base_.get_feature_names_out())
        if self.encoder_ is not None:names+=list(self.encoder_.get_feature_names_out())
        return np.array(names+self.numeric_+[c+'__missing' for c in self.numeric_],dtype=object)


def model_for(group,config,kind='rf',overrides=None,seed=None,smoke=False):
    columns=columns_for(group,config)
    binary=[c for c in columns if c.startswith('b__') and c[3:] in config['binary_features']]
    params=config['random_forest' if kind=='rf' else 'decision_tree'].copy()
    params.update(overrides or {})
    if kind=='rf' and smoke:params['n_estimators']=5
    cls=RandomForestClassifier if kind=='rf' else DecisionTreeClassifier
    return Pipeline([('features',ContextEncoder(columns,binary)),
        ('classifier',cls(random_state=config['seed'] if seed is None else seed,**params))])
