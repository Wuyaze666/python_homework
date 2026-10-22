"""A组特征与固定模型；可被训练入口和独立复核共同导入。"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

FEATURES = ['variable_key', 'value_type', 'value_num']
LABELS = np.arange(6)


class AEncoder(TransformerMixin, BaseEstimator):
    """只拟合训练数据；离散状态不被当作连续大小关系。"""
    def _check(self, x):
        if list(x.columns) != FEATURES:
            raise ValueError('A组只接受变量名、类型、当前值；拒绝标签/身份/时间/片段字段')
        if not x.value_type.isin(['continuous','counter','binary','code']).all():
            raise ValueError('出现未定义的变量类型')
        if np.isinf(x.value_num.to_numpy(dtype=float)).any():
            raise ValueError('当前值含无穷值')

    def _categorical(self, x):
        discrete=x.value_type.isin(['binary','code'])
        tokens=[]
        for var,value,flag in zip(x.variable_key,x.value_num,discrete):
            token='__not_discrete__'
            if flag:
                token=var+'='+('__missing__' if pd.isna(value) else format(float(value),'.17g'))
            tokens.append(token)
        return pd.DataFrame({'variable_key':x.variable_key.to_numpy(),
                             'value_type':x.value_type.to_numpy(),'discrete_value':tokens})

    def fit(self,x,y=None):
        self._check(x)
        numerical=x[x.value_type.isin(['continuous','counter'])]
        self.medians_=numerical.groupby('variable_key').value_num.median().dropna().to_dict()
        self.encoder_=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32)
        self.encoder_.fit(self._categorical(x))
        self.feature_names_in_=np.array(FEATURES,dtype=object)
        self.n_features_in_=3
        return self

    def transform(self,x):
        self._check(x)
        is_numeric=x.value_type.isin(['continuous','counter'])
        values=x.value_num.where(is_numeric,0).copy()
        # 每个变量自己的训练中位数；类别缺失由独立token编码。
        values=values.fillna(x.variable_key.map(self.medians_)).fillna(0)
        missing=x.value_num.isna().astype(float)
        numeric=np.column_stack([values.to_numpy(dtype=float),missing.to_numpy()]).astype(np.float32)
        result=sparse.hstack([self.encoder_.transform(self._categorical(x)),sparse.csr_matrix(numeric)],format='csr')
        if not np.isfinite(result.data).all():
            raise ValueError('编码后存在非有限值')
        return result

    def get_feature_names_out(self,input_features=None):
        return np.concatenate([self.encoder_.get_feature_names_out(),['numeric_current_value','current_value_missing']])


def build_models(config,smoke=False):
    rf=config['random_forest'].copy()
    if smoke:
        rf['n_estimators']=5
    return {
        'decision_tree_A':Pipeline([('features',AEncoder()),('classifier',DecisionTreeClassifier(
            random_state=config['seed'],**config['decision_tree']))]),
        'random_forest_A':Pipeline([('features',AEncoder()),('classifier',RandomForestClassifier(
            random_state=config['seed'],**rf))])
    }


def fit_baselines(train):
    global_counts=np.bincount(train.alarm_class.to_numpy(dtype=int),minlength=6)
    counts={v:np.bincount(g.alarm_class.to_numpy(dtype=int),minlength=6).tolist()
            for v,g in train.groupby('variable_key',sort=True)}
    return {'global_counts':global_counts.tolist(),'variable_counts':counts,
            'tie_rule':'lowest class id','unknown_variable':'global training majority'}


def baseline_proba(model,frame,state):
    fallback=np.asarray(state['global_counts'],dtype=float)
    counts=np.tile(fallback,(len(frame),1))
    if model=='variable_majority':
        counts=np.array([state['variable_counts'].get(v,fallback) for v in frame.variable_key],dtype=float)
    return counts/counts.sum(axis=1,keepdims=True)


def metrics(y,pred,labels=LABELS):
    y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=int)
    rows=[]
    for c in labels:
        support=int((y==c).sum()); predicted=int((pred==c).sum());tp=int(((y==c)&(pred==c)).sum())
        rows.append(dict(alarm_class=int(c),support=support,predicted=predicted,tp=tp,
            fp=predicted-tp,fn=support-tp,
            precision=tp/predicted if predicted else np.nan,
            recall=tp/support if support else np.nan,
            f1=2*tp/(support+predicted) if support else np.nan))
    detail=pd.DataFrame(rows)
    supported=detail[detail.support.gt(0)]
    summary=dict(n=len(y),accuracy=float((y==pred).mean()) if len(y) else np.nan,
        supported_classes=len(supported),macro_f1_supported=supported.f1.mean(),
        macro_f1_6=detail.f1.mean() if len(labels)==6 and len(supported)==6 else np.nan,
        balanced_accuracy=supported.recall.mean())
    return summary,detail
