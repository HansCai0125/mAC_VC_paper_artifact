"""Publication plots of evaluated points, without fitted/interpolated curves."""
import csv
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, NullFormatter

STYLE = [
    ('mAC',r'$mAC$','#0072B2','o','-',1.65),
    ('VC',r'$VC$','#D55E00','s','-',1.65),
    ('F1','F1','#009E73','^','--',1.25),
    ('one_minus_LRP',r'$1-\mathrm{LRP}$','#6F6F6F','D','-.',1.2),
    ('AP50_95',r'diagnostic $mAP_{50:95}$','#7A5195','X',':',1.3),
]


def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def setup():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':8,'axes.labelsize':8,'axes.titlesize':8.3,
        'legend.fontsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.linewidth':.75,'savefig.facecolor':'white'})


def axis_style(ax):
    ax.set_ylim(-.035,1.075); ax.set_yticks([0,.25,.5,.75,1])
    ax.grid(axis='y',color='.88',lw=.55)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',length=3,width=.7,labelsize=7.2)


def handles(styles):
    return [Line2D([0],[0],color=c,marker=m,linestyle=ls,lw=lw,markersize=3.4,label=label)
            for _,label,c,m,ls,lw in styles]


def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',pad_inches=.035)
    fig.savefig(out/(name+'.png'),dpi=400,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)


def fig5(out):
    rows=read(out/'fig5_fragmentation_dense.csv')
    fig,axs=plt.subplots(2,2,figsize=(7.16,5.15),sharey=True)
    plotrows=[]
    for i,ax in enumerate(axs.flat):
        ds=('mscoco','visdrone')[i%2]; exp='fragmentation' if i<2 else 'retention'
        subset=[r for r in rows if r['dataset']==ds and r['experiment']==exp]
        levels=sorted({float(r['level']) for r in subset})
        expected=3 if ds=='mscoco' else 15
        axis_style(ax)
        ax.axhline(1,color='.5',ls=(0,(3,2)),lw=.7,zorder=0)
        for field,label,c,m,ls,lw in STYLE:
            key='normalized_'+field
            values=[[float(r[key]) for r in subset if float(r['level'])==x] for x in levels]
            if any(len(v)!=expected for v in values): raise AssertionError('missing figure5 runs')
            means=[statistics.fmean(v) for v in values]
            lo=[min(v) for v in values]; hi=[max(v) for v in values]
            ax.fill_between(levels,lo,hi,color=c,alpha=.075,lw=0)
            ax.plot(levels,means,color=c,ls=ls,lw=lw,marker=m,markersize=3.3 if i<2 else 2.7,
                    markevery=1 if i<2 else 2,markeredgecolor='white',markeredgewidth=.45)
            plotrows.extend(dict(dataset=ds,experiment=exp,level=x,metric=field,n_runs=expected,
                                 mean=mean,minimum=l,maximum=h)
                            for x,mean,l,h in zip(levels,means,lo,hi))
        name='MS COCO' if ds=='mscoco' else 'VisDrone'
        title='fixed union' if i<2 else 'retained support'
        ax.set_title(f'({"abcd"[i]}) {name}: {title}',loc='left',weight='bold',pad=4)
        if i<2:
            ax.set_xticks([1,4,9,16]); ax.set_xlabel(r'Number of fragments $K$')
        else:
            ax.set_xticks([.2,.4,.6,.8,1]); ax.set_xlabel(r'Retained support fraction $\rho$')
        if i%2==0: ax.set_ylabel('Score / controlled baseline')
    axs[0,0].annotate('regional scores preserve\nfixed union support',xy=(16,1),xytext=(6.1,.77),
        fontsize=7,color='.3',va='top',arrowprops=dict(arrowstyle='->',color='.45',lw=.7))
    axs[0,0].annotate('one-to-one scores\nsaturate near zero',xy=(4,.03),xytext=(7,.29),
        fontsize=7,color='.3',va='top',arrowprops=dict(arrowstyle='->',color='.45',lw=.7))
    axs[1,0].annotate('graded regional response',xy=(.6,.73),xytext=(.27,.93),
        fontsize=7,color='.3',va='top',arrowprops=dict(arrowstyle='->',color='.45',lw=.7))
    fig.legend(handles=handles(STYLE),loc='upper center',bbox_to_anchor=(.52,.995),ncol=5,
               frameon=False,columnspacing=1.2,handlelength=2.2,handletextpad=.45,fontsize=7.4)
    fig.subplots_adjust(left=.09,right=.985,bottom=.095,top=.88,wspace=.12,hspace=.35)
    save(fig,out,'fig5_collective_fragmentation_dense')
    with (out/'fig5_plot_data.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(plotrows[0])); w.writeheader(); w.writerows(plotrows)


def fig6(out):
    rows=read(out/'fig6_canonical_dense.csv')
    fig,axs=plt.subplots(1,3,figsize=(7.16,2.28),sharey=True)
    experiments=[('equal_mac_expansion','center_scale','scale'),
                 ('orphan_injection','fixed_perfect_union','added_orphan_count'),
                 ('responsibility_competition','single_shared_owner','gap_ratio')]
    anchors=[(.5,.71,1,1.41,2,3),(0,1,2,4,8,16),(0,.5,1,2,3,4)]
    for i,(ax,(exp,variant,xkey)) in enumerate(zip(axs,experiments)):
        subset=sorted([r for r in rows if r['experiment']==exp and r['variant']==variant],
                      key=lambda r:float(r[xkey]))
        xs=[float(r[xkey]) for r in subset]
        marks=[k for k,x in enumerate(xs) if any(abs(x-a)<1e-12 for a in anchors[i])]
        axis_style(ax); ax.set_ylim(-.04,1.07); ax.set_yticks([0,.25,.5,.75,1])
        ax.tick_params(labelsize=6.6,length=2.5,pad=2)
        for field,label,c,m,ls,lw in STYLE[:4]:
            ys=[float(r[field]) for r in subset]
            if i==2 and field=='F1':
                # F1 is nonzero only at zero gap. Show the isolated endpoint;
                # no false sloping transition over a positive interval.
                ax.plot(xs[1:],ys[1:],color=c,ls=ls,lw=1.2)
                ax.plot([0],[ys[0]],marker=m,color=c,ms=3.5,ls='none',clip_on=False)
                ax.plot([0],[0],marker=m,mec=c,mfc='white',ms=3.4,ls='none',clip_on=False)
            else:
                ax.plot(xs,ys,color=c,ls=ls,lw=1.35 if field in ('mAC','VC') else 1.1,
                        drawstyle='steps-post' if i==0 and field=='F1' else 'default',
                        marker=m,markevery=marks,ms=3.0,mec='white',mew=.35)
        if i==0:
            ax.set_xscale('log'); ax.set_xlim(.48,3.12)
            ax.set_xticks([.5,.71,1,1.41,2,3]); ax.set_xticklabels(['0.5','0.71','1','1.41','2','3'])
            ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
            ax.axvline(1,color='.5',ls=':',lw=.7)
        elif i==1:
            ax.set_xscale('symlog',linthresh=1,base=2)
            ax.set_xticks([0,1,2,4,8,16]); ax.set_xticklabels(['0','1','2','4','8','16'])
            ax.xaxis.set_minor_locator(NullLocator())
        else:
            ax.set_xlim(-.08,4.08); ax.set_xticks([0,1,2,3,4])
    titles=['(a) Centered box scaling','(b) Fixed-union orphan injection','(c) Shared-prediction ownership']
    labels=[r'Box side-length scale $s$','Added orphan predictions per GT','Inter-target gap / target width']
    for ax,title,label in zip(axs,titles,labels):
        ax.set_title(title,fontsize=7.2,weight='bold',pad=4)
        ax.set_xlabel(label,fontsize=6.9,labelpad=3)
    axs[0].set_ylabel('Score',fontsize=7.5)
    fig.legend(handles=handles(STYLE[:4]),loc='upper center',bbox_to_anchor=(.52,1.015),
               ncol=4,frameon=False,columnspacing=2,handlelength=2.5,fontsize=7.5)
    fig.subplots_adjust(left=.06,right=.99,bottom=.23,top=.78,wspace=.22)
    save(fig,out,'fig6_vc_incremental_value_dense')


def plot_all(out):
    out=Path(out); setup(); fig5(out); fig6(out)
