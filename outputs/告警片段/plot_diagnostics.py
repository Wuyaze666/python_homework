"""任务3诊断图：来源为开发范围真实读数，正式报告图后置。"""
from pathlib import Path
import os
import json

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2]/'work/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_publication_style, COLOR_SEQUENCE
from visual_qa import audit_layout
from profile_data import profile_data


def plot_all(tables, config, output):
    directory=output/'diagnostics'
    directory.mkdir(exist_ok=True)
    apply_publication_style(language='en', width='report')
    plt.rcParams.update({'font.size':10, 'axes.labelsize':10, 'axes.titlesize':11,
                         'xtick.labelsize':9, 'ytick.labelsize':9})
    contracts, audits = [], []
    profile=profile_data(tables['开发片段摘要'], group_cols=['vehicle'])
    (output/'results/绘图前数据剖析.json').write_text(
        json.dumps(profile,ensure_ascii=False,indent=2,default=str)+'\n', encoding='utf-8')

    def save(fig,name,claim,evidence):
        issues=audit_layout(fig)
        if any(level in ('FAIL','WARN') for level,_ in issues):
            raise ValueError(f'{name}布局检查不通过: {issues}')
        fig.savefig(directory/(name+'.png'),dpi=160,metadata={'Software':'telemetry-episode-diagnostic'})
        contracts.append(dict(file=name+'.png', claim=claim, evidence=evidence,
            scope='development only', backend='matplotlib', size_inches=list(fig.get_size_inches()),
            units='raw measurement units, not verified', export='160dpi PNG diagnostic; formal figures deferred'))
        audits.append(dict(file=name+'.png', status='PASS', issues=issues))
        plt.close(fig)

    # 精确同刻和10秒窗采用相同尺度；空分母留白，不当作零关联。
    co=tables['开发类别共现']
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for row,mode in enumerate(['exact_timestamp','right_closed_10s']):
        for col,vehicle in enumerate(['V1','V2','V3']):
            ax=axes[row,col]
            matrix=np.full((5,5),np.nan)
            subset=co[co.vehicle.eq(vehicle)&co['mode'].eq(mode)]
            for r in subset.itertuples():
                matrix[r.class_a-1,r.class_b-1]=r.jaccard
                matrix[r.class_b-1,r.class_a-1]=r.jaccard
            im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues')
            ax.set_xticks(range(5),range(1,6)); ax.set_yticks(range(5),range(1,6))
            for i in range(5):
                for j in range(5):
                    if np.isfinite(matrix[i,j]):
                        ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',
                                fontsize=9,color='white' if matrix[i,j]>.6 else 'black')
            ax.set_title(f'{vehicle} | {"Exact time" if row==0 else "10 s window"}')
            ax.set_xlabel('Observed class');ax.set_ylabel('Observed class')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Jaccard (observed alarm buckets)',shrink=.75)
    save(fig,'class_cooccurrence','同刻与窗口共现是不同的描述口径，不证明因果','开发类别共现.csv')

    summary=tables['开发状态关联']
    fig,axes=plt.subplots(1,3,figsize=(11,4),sharey=True,layout='constrained')
    for ax,vehicle in zip(axes,['V1','V2','V3']):
        g=summary[summary.vehicle.eq(vehicle)&summary.variable_key.eq('vehicle_speed')
                  &summary.feature.eq('engine_rpm')&summary.policy.eq('ttl_available')]
        for r in g.itertuples():
            if r.available_n:
                ax.plot([r.alarm_class]*2,[r.q1,r.q3],color=COLOR_SEQUENCE[0],lw=2)
                ax.scatter(r.alarm_class,r.median,color=COLOR_SEQUENCE[0],s=25)
                ax.annotate(f'{r.available_n}/{r.anchor_count}',(r.alarm_class,r.q3),
                            xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(6));ax.set_xlim(-.5,5.5)
        ax.set_xlabel('Speed-run class');ax.set_title(vehicle)
        ax.margins(y=.25)
    axes[0].set_ylabel('Engine RPM: median and IQR (raw)')
    axes[0].set_ylim(bottom=0)
    fig.suptitle('Development run starts | available anchors / all anchors')
    save(fig,'speed_run_rpm','同车车速游程起点对应转速随类别变化，必须同时看覆盖数','开发状态关联.csv；竖线Q1—Q3，点中位数；class0为正常游程起点')

    sensitivity=tables['间隔敏感性_开发'].groupby(['vehicle','gap_option']).episodes.sum().unstack()
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    positions=np.arange(3)
    for i,option in enumerate(['short_gap','selected_gap','long_gap']):
        values=sensitivity.reindex(['V1','V2','V3'])[option]
        bars=ax.bar(positions+(i-1)*.25,values,width=.25,label=option,color=COLOR_SEQUENCE[i])
        ax.bar_label(bars,fontsize=8,padding=3)
    ax.set_xticks(positions,['V1','V2','V3']);ax.set_ylabel('Observed episodes')
    ax.set_title('Development only | frozen gap sensitivity')
    ax.margins(y=.2);ax.legend(fontsize=8)
    save(fig,'gap_sensitivity','告警片段数依赖采样间隔口径，不能直接当故障数','间隔敏感性_开发.csv；车辆×变量专属阈值见冻结间隔.csv')

    evidence=tables['案例原始证据']
    for case in tables['案例索引'].itertuples():
        window=evidence[evidence.case_episode_id.eq(case.episode_id)]
        features=list(dict.fromkeys([case.variable_key,'vehicle_speed','engine_rpm']))
        # 每个变量独立y轴；额外一面板显示目标变量真实标签。
        fig,axes=plt.subplots(len(features)+1,1,figsize=(10,2+1.7*len(features)),
                              sharex=True,layout='constrained')
        for ax,feature in zip(axes,features):
            g=window[window.variable_key.eq(feature)].sort_values(['time_local','source_row_id'])
            # 超10秒不画连线，点仍保存；不把片段合并阈值当信号有效期。
            group=g.time_local.diff().dt.total_seconds().gt(10).cumsum()
            for _,part in g.groupby(group):
                style='.' if feature=='idling' else '.-'
                ax.plot(part.relative_seconds,part.value_num,style,color=COLOR_SEQUENCE[0],lw=.8,ms=3)
            suspect=g[['high_value_review','protocol_value_review','numeric_parse_failure']].any(axis=1)
            ax.scatter(g.loc[suspect,'relative_seconds'],g.loc[suspect,'value_num'],
                       facecolors='none',edgecolors=COLOR_SEQUENCE[1],s=40)
            ax.set_ylabel(feature.replace('_',' ')+'\n(raw)')
        target=window[window.variable_key.eq(case.variable_key)].sort_values('time_local')
        axes[-1].scatter(target.relative_seconds,target.alarm_class,s=18,color=COLOR_SEQUENCE[2])
        axes[-1].set_yticks(range(6));axes[-1].set_ylabel('Observed class')
        for ax in axes:
            ax.axvline(0,color='black',ls='--',lw=.7)
            half=config['case_half_window_seconds']
            ax.set_xlim(-half,half)
        axes[-1].set_xlabel('Seconds relative to first observation of selected episode')
        axes[0].set_title(f'{case.vehicle} | {case.variable_key} | class {case.alarm_class} | {case.episode_id}',
                          loc='left')
        save(fig,'case_'+case.episode_id,
             '源行可核对的告警前后实际读数；点间缺测不补成连续状态',
             '案例索引.csv、案例原始证据.csv；虚线为片段首观测，圆圈为质量复核标记')
    for name,value in [('诊断图契约',contracts),('图像程序检查',audits)]:
        (output/'results'/(name+'.json')).write_text(
            json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
