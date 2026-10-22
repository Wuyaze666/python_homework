"""仅训练/验证的必要诊断图；正式图表后置。"""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'work/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import apply_publication_style, COLOR_SEQUENCE
from visual_qa import audit_layout
from profile_data import profile_data

NAMES={'global_majority':'Global majority','variable_majority':'Variable majority',
       'decision_tree_A':'Decision tree A','random_forest_A':'Random forest A'}


def plot_all(tables,output):
    folder=output/'diagnostics';folder.mkdir(exist_ok=True)
    apply_publication_style(language='en',width='report')
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':11,
                         'xtick.labelsize':9,'ytick.labelsize':9})
    contracts=[];checks=[]
    profile=profile_data(tables['开发样本索引'][['vehicle','split','variable_key','value_type','alarm_class']],group_cols=['split'])
    (output/'results/绘图前数据剖析.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

    def save(fig,name,claim,evidence):
        issues=audit_layout(fig)
        if any(level in ['WARN','FAIL'] for level,_ in issues):raise ValueError(f'{name}: {issues}')
        fig.savefig(folder/(name+'.png'),dpi=160,metadata={'Software':'telemetry-stage4-diagnostic'})
        checks.append(dict(file=name+'.png',status='PASS',issues=issues))
        contracts.append(dict(file=name+'.png',claim=claim,evidence=evidence,
            scope='train apparent fit / validation only; no test',backend='matplotlib',
            size_inches=list(fig.get_size_inches()),export='160dpi diagnostic PNG; formal report deferred'))
        plt.close(fig)

    metrics=tables['指标汇总'].query("vehicle=='ALL'")
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    positions=np.arange(4)
    for ax,metric in zip(axes,['accuracy','macro_f1_6']):
        for offset,split,color in [(-.18,'train',COLOR_SEQUENCE[0]),(.18,'validation',COLOR_SEQUENCE[1])]:
            values=metrics[metrics.split.eq(split)].set_index('model').reindex(NAMES)[metric]
            bars=ax.bar(positions+offset,values,width=.34,color=color,label='Train (apparent fit)' if split=='train' else 'Validation')
            ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
        ax.set_xticks(positions,['Global\nmajority','Variable\nmajority','Decision\ntree A','Random\nforest A'])
        ax.set_ylim(0,1.1);ax.set_ylabel('Accuracy' if metric=='accuracy' else 'Macro-F1 (six classes)')
    axes[0].legend(fontsize=8,loc='upper left')
    save(fig,'baseline_comparison','固定A组模型是否超过多数类参照；训练分数只表示拟合','指标汇总.csv；同一冻结划分，非交叉验证均值')

    confusion=tables['混淆矩阵'].query("vehicle=='ALL' and split=='validation'")
    fig,axes=plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for ax,(name,title) in zip(axes.ravel(),NAMES.items()):
        matrix=confusion[confusion.model.eq(name)].pivot(index='true_class',columns='predicted_class',values='records').to_numpy()
        normalized=np.divide(matrix,matrix.sum(axis=1,keepdims=True),out=np.zeros_like(matrix,dtype=float),where=matrix.sum(axis=1,keepdims=True)>0)
        im=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
        for i,j in np.ndindex(6,6):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,
                    color='white' if normalized[i,j]>.6 else 'black')
        ax.set_xticks(range(6));ax.set_yticks(range(6))
        ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title(title)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Fraction within true class',shrink=.65)
    save(fig,'validation_confusion','验证集错误流向；颜色为行内比例，数字为真实记录数','混淆矩阵.csv；每模型验证10909条，含V1和V2')

    perclass=tables['逐类指标'].query("split=='validation'")
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,vehicle in zip(axes,['V1','V2']):
        for offset,name,color in [(-.13,'decision_tree_A',COLOR_SEQUENCE[0]),(.13,'random_forest_A',COLOR_SEQUENCE[1])]:
            g=perclass[perclass.vehicle.eq(vehicle)&perclass.model.eq(name)].sort_values('alarm_class')
            ax.scatter(g.alarm_class+offset,g.recall,label=NAMES[name],color=color,s=30)
        support=g.support.to_numpy()
        ax.set_xticks(range(6),[f'{c}\nn={n}' for c,n in enumerate(support)])
        ax.set_title(vehicle);ax.set_ylim(-.04,1.12);ax.set_xlabel('True class and support');ax.set_ylabel('Recall')
        if vehicle=='V2':ax.text(5,.5,'N/A',ha='center',color='gray')
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Validation only | V3 has no validation observations')
    save(fig,'validation_class_recall','逐类逐车支持数限制：V2无类别5，V3无验证样本','逐类指标.csv；无支持类别留空，不写0%召回')
    for name,value in [('图像程序检查',checks),('诊断图契约',contracts)]:
        (output/'results'/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
