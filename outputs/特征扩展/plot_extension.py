"""仅开发期必要诊断图；正式报告图统一后置。"""
from pathlib import Path
import os
import json
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'work/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_style import apply_publication_style,COLOR_SEQUENCE
from visual_qa import audit_layout
from profile_data import profile_data


def plot_all(tables,output,candidate):
    apply_publication_style(language='en',width='report')
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':11,'ytick.labelsize':9,'xtick.labelsize':9})
    folder=output/'diagnostics';folder.mkdir(exist_ok=True)
    checks=[];contracts=[]
    # Profile the delivered precision, so a plot-only replay matches a full run.
    delivered=pd.read_csv(output/'results/实验台账.csv')
    profile=profile_data(delivered[['model','group','seed','macro','accuracy','v1','v2','normal','encoded_features','total_nodes']],group_cols=['group'])
    (output/'results/绘图前数据剖析.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

    def save(fig,name,claim,evidence):
        issues=audit_layout(fig)
        if any(level in ['WARN','FAIL'] for level,_ in issues):raise ValueError(f'{name}: {issues}')
        fig.savefig(folder/(name+'.png'),dpi=160,metadata={'Software':'telemetry-stage5-diagnostic'})
        checks.append(dict(file=name+'.png',status='PASS',issues=issues))
        contracts.append(dict(file=name+'.png',claim=claim,evidence=evidence,backend='matplotlib',
            scope='fixed development train/validation; no test; no independent-sample confidence intervals',
            size_inches=list(fig.get_size_inches()),export='160dpi diagnostic; formal report deferred'))
        plt.close(fig)

    ledger=tables['实验台账'];main=ledger[ledger.selectable].copy()
    names=main.model.tolist();positions=np.arange(len(names))
    m=tables['指标汇总'].query("vehicle=='ALL'")
    fig,ax=plt.subplots(figsize=(10,max(4.5,len(names)*.46)),layout='constrained')
    for offset,split,color,marker in [(-.1,'train',COLOR_SEQUENCE[0],'o'),(.1,'validation',COLOR_SEQUENCE[1],'s')]:
        vals=m[m.split.eq(split)].set_index('model').loc[names].macro_f1_6.to_numpy()
        ax.scatter(vals,positions+offset,color=color,marker=marker,label=split+' (apparent fit)' if split=='train' else split,s=30)
        if split=='validation':
            for xx,yy in zip(vals,positions+offset):ax.text(xx+.012,yy,f'{xx:.4f}',va='center',fontsize=8)
    ax.set_yticks(positions,[n+(' *' if n==candidate else '') for n in names]);ax.invert_yaxis()
    ax.set_xlim(0,1.08);ax.set_xlabel('Macro-F1 (six classes)');ax.legend(loc='lower left',fontsize=9)
    ax.set_title('Fixed validation | * development recommendation')
    save(fig,'experiment_comparison','增量特征及有限调参的真实收益与拟合差距','指标汇总.csv；52892训练，10909验证；种子42')

    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    cm=tables['混淆矩阵'].query("split=='validation'")
    for ax,vehicle in zip(axes,['ALL','V2']):
        matrix=cm[(cm.model==candidate)&(cm.vehicle==vehicle)].pivot(index='true_class',columns='predicted_class',values='records').to_numpy()
        denom=matrix.sum(axis=1,keepdims=True)
        ratio=np.divide(matrix,denom,out=np.full(matrix.shape,np.nan),where=denom>0)
        im=ax.imshow(ratio,vmin=0,vmax=1,cmap='Blues')
        for i,j in np.ndindex(6,6):
            ax.text(j,i,'N/A' if denom[i,0]==0 else str(matrix[i,j]),ha='center',va='center',fontsize=8,
                color='white' if ratio[i,j]>.6 else 'black')
        ax.set_xticks(range(6));ax.set_yticks(range(6));ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
        ax.set_title(f'{vehicle} | n={int(matrix.sum())}')
    fig.colorbar(im,ax=axes.tolist(),label='Fraction within true class',shrink=.7)
    fig.suptitle(candidate+' | counts; color = fraction within true class')
    save(fig,'candidate_confusion','推荐候选仍须检查V2错误与类别5无真实支持的限制','混淆矩阵.csv；颜色0至1，行内比例；V3无验证不展示')

    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    indexed=ledger.set_index('model')
    for ax,metric,title in zip(axes,['macro','v2'],['ALL (six classes)','V2 (five supported classes)']):
        for offset,names,label,color,marker in [(-.08,['RF_A','RF_A_seed43','RF_A_seed44'],'A depth16',COLOR_SEQUENCE[0],'o'),
                (.08,[candidate,'candidate_seed43','candidate_seed44'],'Recommended configuration',COLOR_SEQUENCE[1],'s')]:
            vals=indexed.loc[names,metric].to_numpy()
            ax.scatter(np.arange(3)+offset,vals,s=45,color=color,marker=marker,label=label)
        ax.set_xticks(range(3),['42 (selection)','43 (audit)','44 (audit)'])
        ax.set_ylim(0,1.08);ax.set_xlim(-.4,2.4);ax.set_ylabel('Validation Macro-F1');ax.set_title(title)
    axes[0].legend(loc='lower left',fontsize=9)
    fig.suptitle('Seed sensitivity on the same validation data | no confidence interval')
    save(fig,'seed_sensitivity','推荐方案的小幅收益是否随随机种子改变；不能把三个种子当独立数据验证','实验台账.csv；种子42选择，43/44仅复核，不重选候选')
    for name,value in [('图像程序检查',checks),('诊断图契约',contracts)]:
        (output/'results'/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
