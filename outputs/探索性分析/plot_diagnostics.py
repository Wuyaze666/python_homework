"""仅供EDA检查的图；正式版式和出版导出留待末阶段。"""
from pathlib import Path
import json
import os

# 将可再生字体缓存放在项目工作目录，避免写共享用户配置目录。
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2]/'work/matplotlib-cache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from plot_style import apply_publication_style, COLOR_SEQUENCE
from visual_qa import audit_layout
from profile_data import profile_data

LABELS={'vehicle_speed':'Speed','engine_rpm':'Engine RPM','acceleration_x':'Acceleration X',
        'acceleration_y':'Acceleration Y','acceleration_z':'Acceleration Z',
        'throttle_position':'Throttle position','calculated_engine_load':'Engine load'}


def line_runs(ax, g, time, value, max_gap, color):
    g=g.sort_values(time)
    breaks=g.segment_id.ne(g.segment_id.shift()) | g[time].diff().dt.total_seconds().gt(max_gap)
    for _,r in g.groupby(breaks.cumsum()):
        ax.plot(r[time],r[value],color=color,lw=.7,marker='.',ms=1.5)


def plot_all(raw,wide,tables,config,output):
    directory=output/'diagnostics'; directory.mkdir(parents=True,exist_ok=True)
    results=output/'results'
    apply_publication_style(language='en',width='report')
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':11,
                         'xtick.labelsize':9,'ytick.labelsize':9,'figure.dpi':100})
    core=config['core_features']; vehicles=['V1','V2','V3']
    colors=dict(zip(vehicles,COLOR_SEQUENCE[:3]))
    profile=profile_data(wide['unfilled'][['vehicle']+core],group_cols=['vehicle'])
    (results/'绘图前数据剖析.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    contracts=[]; qa=[]

    def save(fig,name,question,evidence):
        fig.set_layout_engine('constrained')
        issues=audit_layout(fig)
        if any(level in ('FAIL','WARN') for level,_ in issues):
            raise ValueError(f'图像布局检查未通过 {name}: {issues}')
        fig.savefig(directory/(name+'.png'),dpi=160,metadata={'Software':'telemetry-eda-diagnostic'})
        qa.append({'file':name+'.png','issues':issues,'status':'PASS','purpose':'diagnostic_only'})
        contracts.append({'file':name+'.png','question':question,'evidence':evidence,
                          'backend':'matplotlib','size_inches':list(fig.get_size_inches()),
                          'scope':'all_descriptive unless title says development',
                          'units':'raw measurement units; unverified','export':'PNG 160dpi diagnostic; formal figures deferred'})
        plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True,layout='constrained')
    for ax,vehicle in zip(axes,vehicles):
        g=raw[raw.vehicle==vehicle]
        for level,feature in enumerate(['position']+core):
            t=g.loc[g.variable_key==feature,'time_local']
            ax.plot(t,np.full(len(t),level),linestyle='none',marker='|',ms=3,color=colors[vehicle])
        ax.set_yticks(range(6),['GPS']+[LABELS[c] for c in core]); ax.set_title(vehicle,loc='left')
    locator=mdates.AutoDateLocator(minticks=4,maxticks=8)
    axes[-1].xaxis.set_major_locator(locator); axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle('Observed timestamps: GPS and core signals (no duration interpolation)')
    save(fig,'coverage','GPS持续上报是否能代表核心信号持续覆盖？','观测覆盖.csv; 原始清洗长表时间戳')

    for vehicle in vehicles:
        g=raw[(raw.vehicle==vehicle) & raw.variable_key.isin(core)]
        fig,axes=plt.subplots(5,1,figsize=(11,9),sharex=True,layout='constrained')
        for ax,feature in zip(axes,core):
            v=g[g.variable_key==feature]
            line_runs(ax,v,'time_local','value_num',config['raw_line_max_gap_seconds'],colors[vehicle])
            bad=v[v.review]
            ax.scatter(bad.time_local,bad.value_num,s=18,facecolors='none',edgecolors='#D55E00')
            ax.set_ylabel(LABELS[feature]+'\n(raw)')
        locator=mdates.AutoDateLocator(minticks=4,maxticks=8)
        axes[-1].xaxis.set_major_locator(locator); axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.suptitle(f'{vehicle}: observed raw signals; gaps >30s disconnected; circles = review flags')
        save(fig,f'overview_{vehicle}','核查原始时序、断流和疑点是否主导极值','采样间隔.csv; 清洗长表source_row_id')
        choice=tables['典型连续区间.csv'].query('vehicle == @vehicle').iloc[0]
        if choice.status=='selected':
            w=wide['unfilled']; v=w[(w.vehicle==vehicle)&(w.segment_id==choice.segment_id)&w.reference_time.between(choice.start,choice.end)]
            fig,axes=plt.subplots(5,1,figsize=(10,8),sharex=True,layout='constrained')
            for ax,feature in zip(axes,core):
                ax.plot(v.reference_time,v[feature],color=colors[vehicle],lw=1,marker='.',ms=3)
                ax.set_ylabel(LABELS[feature]+'\n(raw)')
            locator=mdates.AutoDateLocator(minticks=3,maxticks=6)
            axes[-1].xaxis.set_major_locator(locator); axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            fig.suptitle(f'{vehicle}: longest complete development interval, n={len(v)} windows')
            save(fig,f'continuous_{vehicle}','在覆盖完整的开发区间核查五核心变量同步变化','典型连续区间.csv; 核心窗口来源.csv')

    fig,axes=plt.subplots(5,1,figsize=(10,11),layout='constrained')
    for ax,feature in zip(axes,core):
        for vehicle,style in zip(vehicles,['-','--',':']):
            v=wide['unfilled'].loc[wide['unfilled'].vehicle==vehicle,feature].dropna().sort_values()
            ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=colors[vehicle],linestyle=style,
                    label=f'{vehicle} n={len(v)}')
        ax.set_xlabel(LABELS[feature]+' (raw)'); ax.set_ylabel('ECDF'); ax.set_ylim(0,1.02)
        ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
    fig.suptitle('Core distributions: unfilled 10s windows; all values retained')
    save(fig,'distributions','三车核心变量分布有何差异，极值是否压缩主体分布？','核心分布.csv; 未填补状态表')

    pairdata=tables['配对明细.csv']; relation=tables['配对相关性.csv']
    for x,y in config['pairs']:
        fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
        for ax,vehicle in zip(axes,vehicles):
            v=pairdata[(pairdata.representation=='unfilled')&(pairdata.x_feature==x)&(pairdata.vehicle==vehicle)]
            ax.scatter(v.x,v.y,s=8,alpha=.3,color=colors[vehicle])
            bad=v[v.review]; ax.scatter(bad.x,bad.y,s=22,facecolors='none',edgecolors='#D55E00')
            r=relation[(relation.scope=='all_descriptive')&(relation.representation=='unfilled')&
                (relation.vehicle==vehicle)&(relation.x_feature==x)&(relation.quality=='all_values')&
                (relation.alignment=='within_ttl')].iloc[0]
            ax.set_title(f'{vehicle}: n={int(r.n_pairs)}, rho={r.rho:.3f}')
            ax.set_xlabel(LABELS[x]+' (raw)'); ax.set_ylabel(LABELS[y]+' (raw)')
        fig.suptitle('Unfilled pairs; asynchronous readings; circles = review flags')
        save(fig,f'relation_{x}','分车核查关系形状与疑点影响，不解释为因果','配对明细.csv; 配对相关性.csv')
    (results/'诊断图契约.json').write_text(json.dumps(contracts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (results/'图像程序检查.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
