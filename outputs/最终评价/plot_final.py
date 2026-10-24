"""最终评价必要诊断，正式报告排版后置。"""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'work/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_style import apply_publication_style,COLOR_SEQUENCE
from visual_qa import audit_layout
from profile_data import profile_data


def plot_all(tables,output):
    apply_publication_style(language='en',width='report')
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':9,'ytick.labelsize':9})
    checks=[];contracts=[]
    def save(fig,name,claim,evidence):
        issues=audit_layout(fig)
        if any(level in ['WARN','FAIL'] for level,_ in issues):raise ValueError(f'{name}: {issues}')
        fig.savefig(output/'diagnostics'/(name+'.png'),dpi=160,metadata={'Software':'telemetry-stage6-diagnostic'})
        checks.append(dict(file=name+'.png',status='PASS',issues=issues))
        contracts.append(dict(file=name+'.png',claim=claim,evidence=evidence,backend='matplotlib',
            size_inches=list(fig.get_size_inches()),export='160dpi diagnostic; formal report deferred'))
        plt.close(fig)
    m=tables['指标汇总']
    prof=profile_data(pd.read_csv(output/'results/指标汇总.csv'),group_cols=['scope','split'])
    (output/'results/绘图前数据剖析.json').write_text(json.dumps(prof,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    names=['global_majority','variable_majority','decision_tree_A','RF_A_depth16','RF_A_depth24']
    labels=['Global majority','Variable majority','Shallow tree','RF depth16 (control)','RF depth24 (primary)']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,metric,title in zip(axes,['accuracy','macro_f1_6'],['Accuracy','Macro-F1 (six classes)']):
        g=m.query("scope=='main_temporal' and vehicle=='ALL' and split=='test'").set_index('model').loc[names]
        ax.scatter(g[metric],range(5),s=40,color=COLOR_SEQUENCE[0])
        for y,v in enumerate(g[metric]):ax.text(v+.015,y,f'{v:.4f}',va='center',fontsize=8)
        ax.set_xlim(0,1.1);ax.set_yticks(range(5),labels if ax==axes[0] else names);ax.invert_yaxis();ax.set_xlabel(title)
    fig.suptitle('Frozen temporal test | n=19662 | no post-test selection')
    save(fig,'temporal_comparison','冻结主模型与预登记参照在同一测试范围的成绩','指标汇总.csv；不根据测试排名更换主模型')

    cm=tables['混淆矩阵'];fig,axes=plt.subplots(1,3,figsize=(15,5.8),layout='constrained')
    for ax,car in zip(axes,['V1','V2','V3']):
        g=cm.query("scope=='main_temporal' and model=='RF_A_depth24' and vehicle==@car and split=='test'")
        matrix=g.pivot(index='true_class',columns='predicted_class',values='records').to_numpy();denom=matrix.sum(axis=1,keepdims=True)
        norm=np.divide(matrix,denom,out=np.full(matrix.shape,np.nan),where=denom>0)
        im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i,j in np.ndindex(6,6):
            ax.text(j,i,'N/A' if denom[i,0]==0 else str(matrix[i,j]),ha='center',va='center',fontsize=7,
                color='white' if norm[i,j]>.6 else 'black')
        ax.set_xticks(range(6));ax.set_yticks(range(6));ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title(f'{car} | n={int(matrix.sum())}')
    fig.colorbar(im,ax=axes.tolist(),label='Fraction within true class',shrink=.7)
    fig.suptitle('Primary RF depth24 | temporal test counts and class support')
    save(fig,'primary_confusion','逐车错误流向及缺类边界，不能用总体准确率掩盖车辆差异','混淆矩阵.csv；无真实支持的行留空')

    fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained')
    for k,name in enumerate(['global_majority','variable_majority','decision_tree_A','RF_A_depth16']):
        g=m[(m.model==name)&(m.vehicle=='ALL')&(m.split=='test')&m.scope.str.startswith('loco_')].set_index('scope').reindex(['loco_V1','loco_V2','loco_V3'])
        ax.scatter(np.arange(3)+(k-1.5)*.12,g.macro_f1_supported,label=name,color=COLOR_SEQUENCE[k],marker=['o','s','^','D'][k],s=40)
    ax.set_xticks(range(3),['Held V1','Held V2','Held V3']);ax.set_ylim(0,1.06);ax.set_ylabel('Macro-F1 (supported classes)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.1),ncol=2,fontsize=9)
    ax.set_title('Fixed initial A configurations | held-car labels excluded from fitting')
    save(fig,'loco_comparison','未见车辆迁移按三个预定义折分别报告，不与主时间测试混合','指标汇总.csv；V2五类，V1/V3按实际支持；固定深度16而非主模型24')

    ep=tables['主测试片段汇总'];g=ep[(ep.vehicle=='ALL')&(ep.alarm_class.astype(str)=='ALL')&(ep.length_bin!='ALL')].set_index('length_bin').reindex(['1','2-5','6+'])
    fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained')
    for offset,field,label,color,marker in [(-.12,'any_alarm_detected_rate','Any nonzero prediction',COLOR_SEQUENCE[0],'o'),
            (0,'any_correct_class_rate','Any correct-class prediction',COLOR_SEQUENCE[1],'s'),
            (.12,'onset_correct_rate','Correct class at first observation',COLOR_SEQUENCE[2],'^')]:
        ax.scatter(np.arange(3)+offset,g[field],label=label,color=color,marker=marker,s=45)
    ax.set_xticks(range(3),[f'{b} records\nn={int(n)} episodes' for b,n in zip(g.index,g.episodes)])
    ax.set_ylim(0,1.08);ax.set_ylabel('Fraction of observed alarm episodes');ax.legend(loc='lower right',fontsize=9)
    ax.set_title('Primary temporal test | not a real-fault detection rate')
    save(fig,'episode_detection','片段覆盖受长度和定义影响，至少一次非零与至少一次正确分类不同','主测试片段汇总.csv；真实告警观测片段，未知PID排除，无逐行独立置信区间')
    for name,value in [('图像程序检查',checks),('诊断图契约',contracts)]:
        (output/'results'/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
