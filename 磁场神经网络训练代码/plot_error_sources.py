import os
os.environ['MPLCONFIGDIR']='D:/mag/训练结果/field_mlp_v1_analysis/mpl_cache'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
out=Path('D:/mag/训练结果/field_mlp_v1_analysis/error_sources')
d=json.loads((out/'diagnostics.json').read_text(encoding='utf-8'))
v=json.loads((out/'verification.json').read_text(encoding='utf-8'))
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
 font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
keys=['trajectory','neighbor','workspace'];labels=['真实轨迹','轨迹邻域','随机工作空间'];xx=np.arange(3)
axs[0,0].bar(xx-.18,[d['by_sampling_kind'][k]['sample_fraction']*100 for k in keys],.36,label='样本占比',color='#4C78A8')
axs[0,0].bar(xx+.18,[d['by_sampling_kind'][k]['total_loss_share']*100 for k in keys],.36,label='总损失占比',color='#E45756')
axs[0,0].set(xticks=xx,xticklabels=labels,ylabel='占比 (%)',title='随机点占约 25% 样本，却贡献约 93% 总损失');axs[0,0].legend()
for j,(k,label,color) in enumerate([('field','磁场','#4C78A8'),('gradient','梯度','#F58518')]):
 axs[0,1].bar(xx+(j-.5)*.36,[d['by_sampling_kind'][a][k]['relative_l2']*100 for a in keys],.36,label=label,color=color)
axs[0,1].set(xticks=xx,xticklabels=labels,ylabel='整体相对 L2 误差 (%)',title='当前网络在训练集上的响应误差');axs[0,1].legend()
with np.load(out/'per_sample_diagnostics.npz') as a:
 edges=np.array([1,3,5,10,20,30,50,100,200,500]);shares=[]
 for lo,hi in zip(edges[:-1],edges[1:]):
  mask=(a['source_envelope_clearance_m']*1000>=lo)&(a['source_envelope_clearance_m']*1000<hi)
  shares.append(a['gradient_loss'][mask].sum()/a['gradient_loss'].sum()*100)
 axs[1,0].bar(np.arange(len(shares)),shares,color='#E45756')
 axs[1,0].set(xticks=np.arange(len(shares)),xticklabels=[f'{lo}–{hi}' for lo,hi in zip(edges[:-1],edges[1:])],xlabel='距线圈保守圆柱包络距离 (mm)',ylabel='梯度损失占比 (%)',title='梯度损失集中在靠近线圈的少量点')
 axs[1,0].tick_params(axis='x',rotation=35)
comparison=[d['by_sampling_kind']['trajectory']['field']['relative_l2'],v['other_point_interpolation']['trajectory']['nearest_field']['relative_l2'],d['by_sampling_kind']['trajectory']['gradient']['relative_l2'],v['other_point_interpolation']['trajectory']['nearest_gradient']['relative_l2']]
bars=axs[1,1].bar(np.arange(4),np.array(comparison)*100,color=['#E45756','#4C78A8','#E45756','#4C78A8'])
axs[1,1].set(xticks=np.arange(4),xticklabels=['网络磁场','邻点磁场','网络梯度','邻点梯度'],ylabel='整体相对 L2 误差 (%)',title='轨迹局部信息可用，网络尚未有效拟合')
for bar,value in zip(bars,comparison):axs[1,1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.7,f'{value*100:.2f}%',ha='center')
fig.suptitle('现有数据误差来源诊断 · 原模型保持不变',fontsize=17)
fig.supxlabel('均为已有训练数据诊断；邻点估计排除自身，但可能共享轨迹或父样本，不能当作独立验证。',fontsize=10)
fig.savefig(out/'error_sources_summary.png',dpi=170)
fig.savefig(out/'error_sources_summary.pdf')
print(out/'error_sources_summary.png')
