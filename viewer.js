(function () {
  const payload = JSON.parse(document.getElementById('payload').textContent);
  const {registry, datasets, source, labels} = payload;
  const Core = window.ViewerCore;
  const fieldDomains=window.FieldScope.create(source,registry);
  const element = id => document.getElementById(id);
  const coreKeys = ['time_s','arc_s_mm','total_cost_raw','risk_norm','actual_position_x_mm','actual_position_y_mm','actual_position_z_mm','actual_velocity_x_mm_s','actual_velocity_y_mm_s','actual_velocity_z_mm_s','actual_acceleration_x_mm_s2','actual_acceleration_y_mm_s2','actual_acceleration_z_mm_s2','ref_speed_mm_s','actual_speed_mm_s','ref_acceleration_norm_mm_s2','actual_acceleration_norm_mm_s2','position_error_mm','axis_error_deg','required_field_norm_mT','applied_field_norm_mT','required_gradient_norm_T_m','applied_gradient_norm_T_m','required_force_norm_mN','applied_force_norm_mN','required_torque_norm_uN_m','applied_torque_norm_uN_m'];
  const metricEntries = [
    ['risk_norm','当前风险','','risk'],['actual_speed_mm_s','实际速度','mm/s',''],
    ['actual_acceleration_norm_mm_s2','实际加速度','mm/s²',''],['position_error_mm','位置误差','mm',''],
    ['applied_field_norm_mT','施加总场 |B|','mT',''],['axis_error_deg','长轴方向误差','°',''],
  ];
  let robot = registry.robots[0], data = datasets[robot.id];
  let currentIndex = 0, wantedIndex = 0, playing = false, rendering = false, renderQueued = false;
  let playWall = 0, playTime = 0, ready = false, switching = false, lastPaintTime = 0, lastProbeIndex = null;
  let linkPromise = null, linkQueued = false;
  let pendingProbe = null;
  let requestedRobotId = robot.id;
  const options = {responsive: true, displaylogo: false, scrollZoom: true};
  const format = (value, digits = 4) => Math.abs(value) > 0 && (Math.abs(value) < 0.0001 || Math.abs(value) >= 1e5) ? value.toExponential(3) : Number(value).toFixed(digits);
  const rowAt = index => Object.fromEntries(Object.keys(data).map(key => [key, data[key][index]]));
  const state = () => rowAt(currentIndex);
  const axisStyle = title => ({title:{text:title,font:{size:10,color:'#6b7b8d'}},tickfont:{size:9,color:'#8996a6'},gridcolor:'#e9eef3',zeroline:false,showbackground:false});
  const liveSceneCamera = () => {
    const scene=element('scene')._fullLayout?.scene;
    const camera=scene?._scene?.getCamera?.()||scene?.camera;
    return camera?JSON.parse(JSON.stringify(camera)):{eye:{x:1.55,y:1.45,z:.95}};
  };
  const sceneLayout = item => {
    const follow=element('viewMode').value==='follow';
    const scene={aspectmode:follow?'cube':'manual',camera:liveSceneCamera(),bgcolor:'#fff',uirevision:'unified-camera'};
    for(const axis of Core.axes){
      const component=Core.axes.indexOf(axis);
      const values=[...source[`${axis}_mm`],...data[`actual_position_${axis}_mm`],payload.geometry.bounds[0][component],payload.geometry.bounds[1][component]];
      const center=item[`actual_position_${axis}_mm`];
      scene[`${axis}axis`]={...axisStyle(`${axis} (mm)`),range:follow?[center-48,center+48]:[Math.min(...values)-42,Math.max(...values)+42],autorange:false};
    }
    if(!follow){
      const spans=Core.axes.map(axis=>scene[`${axis}axis`].range[1]-scene[`${axis}axis`].range[0]);
      const scale=Math.cbrt(spans.reduce((product,span)=>product*span,1));
      scene.aspectratio=Object.fromEntries(Core.axes.map((axis,index)=>[axis,spans[index]/scale]));
    }
    return {margin:{l:0,r:42,t:2,b:0},paper_bgcolor:'#fff',showlegend:false,font:{family:'Segoe UI, Microsoft YaHei'},scene,uirevision:'unified-camera'};
  };
  const plotLayout = () => ({margin:{l:48,r:47,t:12,b:37},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'Segoe UI, Microsoft YaHei',size:10,color:'#6b7b8d'},legend:{orientation:'h',x:0,y:1.17,font:{size:10}},xaxis:{title:{text:'时间 (s)',font:{size:10}},gridcolor:'#f0f3f6',zeroline:false},yaxis:{gridcolor:'#f0f3f6',zeroline:false},hovermode:'closest',uirevision:'line-camera',shapes:[]});

  function pathTrace() {
    return {type:'scatter3d',mode:'lines+markers',x:source.x_mm,y:source.y_mm,z:source.z_mm,visible:element('showPath').checked,customdata:source.arc_s_mm.map((arc,index)=>[source.total_cost_raw[index],source.risk_norm[index],arc,index]),line:{color:source.risk_norm,colorscale:Core.colorscale,cmin:0,cmax:1,width:7,showscale:element('showPath').checked,colorbar:{title:{text:'风险',font:{size:10}},len:.36,y:.76,thickness:10,tickfont:{size:9},x:1.005}},marker:{color:source.risk_norm,colorscale:Core.colorscale,cmin:0,cmax:1,size:2,opacity:.75},name:'风险中心线',hovertemplate:'初始风险 %{customdata[0]:.5f}<br>归一化风险 %{customdata[1]:.5f}<br>弧长 %{customdata[2]:.1f} mm<extra>原始中心线采样点</extra>'};
  }

  const fieldSettings = () => ({scope:element('fieldScope').value,domain:fieldDomains[element('fieldScope').value],mode:element('fieldMode').value,plane:element('slicePlane').value,arrows:element('showField').checked,lines:element('showLines').checked,slice:element('showSlice').checked,force:element('showForce').checked,tangent:element('showTangent').checked,steering:element('showSteering').checked,torque:element('showTorque').checked,robot:element('showRobot').checked,robotScale:Number(element('robotScale').value)});
  const fieldTraces = item => window.FieldView.traces(item,robot,registry,fieldSettings());

  function probeTrace() {
    const index=lastProbeIndex;
    return {type:'scatter3d',mode:'markers',x:index===null?[]:[source.x_mm[index]],y:index===null?[]:[source.y_mm[index]],z:index===null?[]:[source.z_mm[index]],marker:{size:7,color:'#2166ac',line:{color:'#fff',width:2}},visible:index!==null&&element('showPath').checked,hoverinfo:'skip',showlegend:false,name:'中心线取点'};
  }

  function sceneTraces(item) {
    return [pathTrace(),{type:'scatter3d',mode:'lines',x:data.actual_position_x_mm,y:data.actual_position_y_mm,z:data.actual_position_z_mm,line:{color:'#8e9aab',width:2,dash:'dot'},visible:element('showTrajectory').checked,name:'实际轨迹',hoverinfo:'skip'},...fieldTraces(item),probeTrace(),...window.CameraView.traces(robot.id,currentIndex)];
  }

  function moduleRow(label, value, main = false) {
    return `<div class="module-row ${main?'main':''}"><span>${label}</span><b>${value}</b></div>`;
  }

  function updateModules(item) {
    const demand = Core.forceProjection(item,'required'), applied = Core.forceProjection(item,'applied');
    const pair = (first,second,unit,digits=4) => `${format(first,digits)} / ${format(second,digits)} ${unit}`;
    element('propulsionPanel').hidden=!element('showPropulsion').checked;
    element('turningPanel').hidden=!element('showTurningPanel').checked;
    element('propulsionValues').innerHTML='<p class="detail-note">左：参考需求 / 右：实际施加</p>'+moduleRow('梯度 ‖G‖',pair(item.required_gradient_norm_T_m,item.applied_gradient_norm_T_m,'T/m'),true)+moduleRow('总磁力 |F|',pair(item.required_force_norm_mN,item.applied_force_norm_mN,'mN'),true)+moduleRow('沿路径有符号推进',pair(demand.parallel,applied.parallel,'mN'),true)+'<p class="detail-note">正值沿参考切向，负值反向；总磁力还承担抵消重力等作用。</p>';
    element('turningValues').innerHTML='<p class="detail-note">左：参考需求 / 右：实际施加</p>'+moduleRow('垂直场 |B⊥|',pair(item.required_field_perpendicular_mT*1000,item.applied_field_perpendicular_mT*1000,'μT'),true)+moduleRow('可控转矩 |τ⊥|',pair(Math.hypot(...Core.axes.map(axis=>item[`required_torque_${axis}_uN_m`]-item[`required_unavailable_torque_${axis}_uN_m`])),item.applied_torque_norm_uN_m,'μN·m',5),true)+moduleRow('长轴方向误差',format(item.axis_error_deg,5)+' °')+moduleRow('不可控轴向需求',format(item.required_unavailable_torque_norm_uN_m,5)+' μN·m');
    element('gradientMatrix').innerHTML=['required','applied'].map(prefix=>`<p class="detail-note">${prefix==='required'?'需求':'施加'} G（T/m），行：Bx/By/Bz，列：x/y/z偏移</p><div class="gradient-matrix">${Core.axes.map(first=>Core.axes.map(second=>item[`${prefix}_G_${first}${second}_T_m`].toFixed(6).padStart(10)).join(' ')).join('\n')}</div>`).join('')+'<p class="detail-note">G 在当前仿射模型中空间恒定；场强切片变化不等于 G 自身有热点。</p>';
    element('forceDetails').innerHTML=Core.axes.map(axis=>moduleRow(`磁力 F${axis}`,pair(item[`required_force_${axis}_mN`],item[`applied_force_${axis}_mN`],'mN'))).join('')+moduleRow('实际横向磁力',format(Math.hypot(...applied.transverse),5)+' mN');
    element('turningDetails').innerHTML=moduleRow('参考 / 实际角速度',pair(item.ref_angular_speed_rad_s,item.actual_angular_speed_rad_s,'rad/s'))+Core.axes.map(axis=>moduleRow(`实际角加速度 α${axis}`,format(item[`actual_alpha_body_${axis}_rad_s2`],5)+' rad/s²')).join('')+moduleRow('完整姿态误差',format(item.full_attitude_error_deg,5)+' °')+'<p class="detail-note">单偶极子不能直接提供沿磁轴的滚转转矩。</p>';
    const settings=fieldSettings(), unit=settings.mode==='turning'?'μT':'mT';
    const grid=settings.scope==='local'?Core.fieldGrid(item,robot,30,5,settings.mode,robot.config.bias_field_T*1000):window.FieldScope.grid(item,robot,settings.domain,settings.mode,robot.config.bias_field_T*1000);
    element('fieldCount').textContent=settings.arrows?`${grid.x.length} 个场箭头`:'场箭头已隐藏';
    const scopeLabels={centerline:'覆盖中心线 · 沿路径 ±8 mm',local:'局部 · 机器人周围 ±30 mm',whole:'整体 · 中心线包围空间 +30 mm'};
    element('fieldUnit').textContent=`${scopeLabels[settings.scope]} · 固定 ${unit} 色标 · 世界坐标`;
    element('gradientBadge').hidden=!element('showGradient').checked;
    element('gradientBadge').textContent=`梯度 ‖G‖：需求 ${format(item.required_gradient_norm_T_m,4)} / 施加 ${format(item.applied_gradient_norm_T_m,4)} T/m；沿路径磁力 ${format(applied.parallel,4)} mN`;
    const messages={total:'完整场含原模型15 mT平行偏置；颜色和悬停数值表示模型场强。',residual:'仅在显示时扣除平行偏置；仍包含 G·Δx 的空间变化，不是只剩转向分量。',turning:'中心垂直场 B⊥（μT）均匀展示；非零箭头等长6.5 mm仅示方向，真实大小看固定色标/悬停。'};
    const scopeMessages={centerline:'覆盖中心线：沿整条中心线采样当前时刻的场；远处数值是局部仿射模型的外推。',local:'局部：保留机器人周围60 mm立方网格，适合观察当前位置。',whole:'整体：采样覆盖中心线的包围空间；这是局部仿射模型的大范围外推，不是真实线圈或磁铁的全空间解。'};
    element('fieldExplanation').textContent=messages[settings.mode]+' '+scopeMessages[settings.scope]+' 箭头长度仅为展示缩放，外观放大不参与计算。';
  }

  function robotCards() {
    const unit = (value,label,small,sub='') => `<div class="parameter"><div class="parameter-label">${label}</div><div class="parameter-value">${value}<small>${small}</small></div><div class="parameter-sub">${sub}</div></div>`;
    element('robotParameters').innerHTML = unit(`${format(robot.length_mm,0)} × ${format(robot.diameter_mm,0)}`,'长度 × 最大直径','mm','真实几何尺寸') + unit(format(robot.mass_g,4),'质量','g','当前建模假设') + unit(format(robot.magnetic_moment_A_m2,5),'磁偶极矩','A·m²','磁轴沿机器人长轴') + `<div class="parameter inertia"><div class="parameter-label">转动惯量 · kg·m²</div><div class="parameter-value">Jx ${robot.inertia_kg_m2[0].toExponential(3)}</div><div class="parameter-sub">Jy=Jz ${robot.inertia_kg_m2[1].toExponential(3)}</div></div>` + unit('6','刚体状态自由度','DOF','单偶极子不直接控制轴向滚转');
    element('shapeNote').textContent = `${robot.name} · 独立积分 · ${robot.sample_count} 个时刻`;
  }

  function groupName(key) {
    if (key.startsWith('applied')) return '闭环实际施加 · 磁场与驱动力';
    if (key.startsWith('required')) return '参考需求 · 逆解结果';
    if (key.startsWith('actual_')) return '实际运动 · 前向动力学';
    if (key.startsWith('ref_')) return '参考运动 · 路径导数';
    return '时间、风险与跟踪状态';
  }

  function updateState() {
    const item = state();
    element('stateTime').textContent = `${format(item.time_s,2)} s`;
    element('timeReadout').textContent = `${format(item.time_s,2)} s / ${format(data.time_s.at(-1),2)} s`;
    element('arcReadout').textContent = `沿路径 ${format(item.arc_s_mm,1)} mm`;
    element('slider').value = currentIndex;
    element('metrics').innerHTML = metricEntries.map(([key,title,unit,css])=>`<div class="metric ${css}"><div class="metric-title">${title}</div><div class="metric-value">${format(item[key],key==='risk_norm'?3:Math.abs(item[key])<.01?5:3)}<small>${unit}</small></div></div>`).join('');
    updateTable();
    updateModules(item);
  }

  function positionRoundTripMarker() {
    const marker=element('roundTripJoinMarker');
    if(!marker||!payload.roundTrip)return;
    const maximum=Number(element('slider').max)||1;
    marker.style.left=`${100*payload.roundTrip.join_index/maximum}%`;
    marker.title=`${format(payload.roundTrip.join_time_s,2)} s · 到达肠道末端，开始退镜`;
  }

  function updateTable() {
    const item = state(), query = element('filter').value.trim().toLowerCase();
    const keys = element('showAll').checked ? Object.keys(data) : coreKeys;
    element('fieldTotal').textContent = element('showAll').checked ? `${keys.length} 个字段` : '核心字段';
    const groups = new Map();
    for (const key of keys) {
      if (query && !(`${labels[key]} ${key}`).toLowerCase().includes(query)) continue;
      const name = groupName(key);
      if (!groups.has(name)) groups.set(name,[]);
      groups.get(name).push(key);
    }
    element('stateTable').innerHTML = [...groups].map(([name,entries])=>`<table class="state-table"><tbody><tr class="group"><td colspan="2">${name}</td></tr>${entries.map(key=>`<tr data-key="${key}"><td title="${key}">${labels[key]||key}</td><td>${format(item[key])}</td></tr>`).join('')}</tbody></table>`).join('') || '<div class="empty">没有匹配参数</div>';
  }

  function probe(index,origin) {
    if (!Number.isInteger(index) || index<0 || index>=source.arc_s_mm.length) return;
    if (index===lastProbeIndex) return;
    if (linkPromise!==null || rendering || switching) {pendingProbe={index,origin};return;}
    lastProbeIndex = index;
    element('probeRaw').textContent = format(source.total_cost_raw[index],5);
    element('probeNorm').textContent = format(source.risk_norm[index],5);
    element('probeArc').textContent = format(source.arc_s_mm[index],1);
    const replayIndex=Core.nearestSampleIndex(data.arc_s_mm,source.arc_s_mm[index]);
    element('probeTime').textContent = format(data.time_s[replayIndex],2);
    element('probeHint').textContent = `${origin} · 点 ${index+1}/${source.arc_s_mm.length} · 最近回放样本`;
    element('riskProbe').dataset.index = index;
    element('clearProbe').disabled=false;
    updateLinkedCursors().catch(fail);
  }

  function cursorLine(position,probePosition=null) {
    const lines=[{type:'line',xref:'x',yref:'paper',x0:position,x1:position,y0:0,y1:1,line:{color:'#132f49',width:1.3,dash:'dash'}}];
    if(probePosition!==null) lines.push({type:'line',xref:'x',yref:'paper',x0:probePosition,x1:probePosition,y0:0,y1:1,line:{color:'#2166ac',width:2,dash:'dot'}});
    return lines;
  }

  const probeReplayIndex = () => lastProbeIndex===null?null:Core.nearestSampleIndex(data.arc_s_mm,source.arc_s_mm[lastProbeIndex]);
  const probeTime = () => {const index=probeReplayIndex();return index===null?null:data.time_s[index];};

  function updateMagProbe() {
    const index=probeReplayIndex();
    if(index===null){element('magProbe').innerHTML='<span>悬停中心线，即可查看该位置的梯度和磁力；也可直接悬停本曲线。</span>';return;}
    const values=[['取点时刻',format(data.time_s[index],2),'s'],['需求梯度',format(data.required_gradient_norm_T_m[index],5),'T/m'],['施加梯度',format(data.applied_gradient_norm_T_m[index],5),'T/m'],['需求总磁力',format(data.required_force_norm_mN[index],5),'mN']];
    if(element('showAppliedForce').checked) values.push(['施加总磁力',format(data.applied_force_norm_mN[index],5),'mN']);
    element('magProbe').innerHTML=values.map(([label,value,unit])=>`<span>${label} <b>${value}</b> ${unit}</span>`).join('');
    element('magProbe').dataset.replayIndex=index;
    element('probeTime').textContent=format(data.time_s[index],2);
  }

  function updateLinkedCursors() {
    linkQueued=true;
    if(linkPromise) return linkPromise;
    linkPromise=Promise.resolve().then(async()=>{
      while(linkQueued){
        linkQueued=false;
        const item=state(),index=probeReplayIndex(),time=index===null?null:data.time_s[index];
        updateMagProbe();
        const marker=probeTrace();
        const updates=[
          Plotly.relayout(element('riskPlot'),{shapes:cursorLine(item.arc_s_mm,lastProbeIndex===null?null:source.arc_s_mm[lastProbeIndex])}),
          Plotly.relayout(element('statePlot'),{shapes:cursorLine(item.time_s,time)}),
          Plotly.update(element('scene'),{x:[marker.x],y:[marker.y],z:[marker.z],visible:marker.visible},{'scene.camera':liveSceneCamera()},[15]),
        ];
        if(index!==null){
          const keys=['required_gradient_norm_T_m','applied_gradient_norm_T_m','required_force_norm_mN','applied_force_norm_mN'];
          updates.push(Plotly.update(element('magPlot'),{x:keys.map(()=>[time]),y:keys.map(key=>[data[key][index]]),visible:[true,true,true,element('showAppliedForce').checked]},{shapes:cursorLine(item.time_s,time)},[4,5,6,7]));
        } else updates.push(Plotly.update(element('magPlot'),{visible:false},{shapes:cursorLine(item.time_s)},[4,5,6,7]));
        await Promise.all(updates);
      }
    }).finally(()=>{linkPromise=null;flushPendingProbe();});
    return linkPromise;
  }

  function clearProbe() {
    pendingProbe=null;
    lastProbeIndex=null;
    for(const id of ['probeRaw','probeNorm','probeArc','probeTime']) element(id).textContent='—';
    element('probeHint').textContent='悬停中心线，联动下方梯度与磁力曲线';
    delete element('riskProbe').dataset.index;
    delete element('magProbe').dataset.replayIndex;
    element('clearProbe').disabled=true;
    updateLinkedCursors().catch(fail);
  }

  function flushPendingProbe() {
    if (!pendingProbe || rendering || switching || linkPromise!==null) return;
    const next=pendingProbe;
    pendingProbe=null;
    requestAnimationFrame(()=>probe(next.index,next.origin));
  }

  function stateChart() {
    const metric = element('chartMetric').value;
    const definitions = {
      motion:[['actual_speed_mm_s','实际速度','#2166ac','速度 (mm/s)'],['actual_acceleration_norm_mm_s2','实际加速度','#b2182b','加速度 (mm/s²)']],
      magnetics:[['applied_field_norm_mT','施加 |B|','#2166ac','磁场 (mT)'],['applied_gradient_norm_T_m','施加 ‖G‖','#b2182b','梯度 (T/m)']],
      steering:[['applied_field_perpendicular_mT','转向 |B⊥|','#8055bc','垂直场 (μT)'],['applied_torque_norm_uN_m','可控转矩','#b82167','转矩 (μN·m)']],
      tracking:[['position_error_mm','位置误差','#2166ac','误差 (mm)'],['axis_error_deg','长轴方向误差','#b2182b','角度 (°)']],
    };
    const entries = definitions[metric];
    const traces = entries.map(([key,name,color],index)=>({type:'scatter',mode:'lines',x:data.time_s,y:metric==='steering'&&index===0?data[key].map(value=>value*1000):data[key],name,yaxis:index?'y2':'y',line:{color,width:1.8},hovertemplate:`${name}: %{y:.6f}<br>时间 %{x:.2f} s<extra>${robot.name}</extra>`}));
    const layout = plotLayout();
    layout.yaxis.title = {text:entries[0][3],font:{size:10}};
    layout.yaxis.tickformat = '.4~g';
    if (metric === 'magnetics') {
      layout.yaxis.range = [0,Math.max(16,Math.max(...data.applied_field_norm_mT)*1.1)];
      layout.yaxis.tickformat = '.1f';
    }
    layout.yaxis2 = {title:{text:entries[1][3],font:{size:10}},tickfont:{size:9},overlaying:'y',side:'right',showgrid:false,zeroline:false};
    layout.shapes = cursorLine(data.time_s[currentIndex],probeTime());
    return Plotly.react(element('statePlot'),traces,layout,options);
  }

  function propulsionChart() {
    const traces=[
      {key:'required_gradient_norm_T_m',name:'需求 ‖G‖ (T/m)',color:'#e6423c',dash:'solid',axis:'y'},
      {key:'applied_gradient_norm_T_m',name:'闭环 ‖G‖ (T/m)',color:'#ef9847',dash:'dash',axis:'y'},
      {key:'required_force_norm_mN',name:'需求 F (mN)',color:'#198c74',dash:'solid',axis:'y2'},
      {key:'applied_force_norm_mN',name:'闭环 F (mN)',color:'#58a58b',dash:'dash',axis:'y2'},
    ].map((definition,index)=>({type:'scatter',mode:'lines',x:data.time_s,y:data[definition.key],name:definition.name,yaxis:definition.axis,line:{color:definition.color,width:1.8,dash:definition.dash},visible:index===3?element('showAppliedForce').checked:true,hovertemplate:`${definition.name}: %{y:.6f} ${definition.axis==='y'?'T/m':'mN'}<br>时间 %{x:.2f} s<extra>${robot.name}</extra>`}));
    const selected=probeReplayIndex();
    const markers=traces.map((trace,index)=>({type:'scatter',mode:'markers',x:selected===null?[]:[data.time_s[selected]],y:selected===null?[]:[trace.y[selected]],yaxis:trace.yaxis,marker:{color:trace.line.color,size:8,line:{color:'#fff',width:1.5}},showlegend:false,hoverinfo:'skip',visible:selected!==null&&(index!==3||element('showAppliedForce').checked),name:'取点 · '+trace.name}));
    const layout=plotLayout();layout.margin={l:58,r:58,t:33,b:37};
    layout.yaxis.title={text:'梯度 (T/m)',font:{size:11}};
    layout.yaxis2={title:{text:'总磁力 (mN)',font:{size:11}},overlaying:'y',side:'right',showgrid:false,zeroline:false};
    layout.legend.y=1.23;layout.shapes=cursorLine(data.time_s[currentIndex],probeTime());
    updateMagProbe();
    return Plotly.react(element('magPlot'),[...traces,...markers],layout,options);
  }

  async function renderFrame() {
    if (!ready || rendering || switching) {renderQueued = true; return;}
    rendering = true;
    if(linkPromise) await linkPromise;
    do {
      renderQueued = false;
      currentIndex = wantedIndex;
      const item = state();
      updateState();
      await Plotly.react(element('scene'),sceneTraces(item),sceneLayout(item),options);
      await updateLinkedCursors();
      window.CameraView.update(robot.id,currentIndex);
    } while (renderQueued || currentIndex !== wantedIndex);
    rendering = false;
    lastPaintTime = performance.now();
    flushPendingProbe();
  }

  function requestFrame(index) {
    wantedIndex = Math.max(0,Math.min(data.time_s.length-1,Math.round(index)));
    renderFrame().catch(fail);
  }

  function pause() {playing=false; element('play').textContent='▶ 播放';}

  function animate(now) {
    if (!playing) return;
    const target = Math.min(playTime+(now-playWall)/1000*Number(element('playbackSpeed').value),data.time_s.at(-1));
    if (now-lastPaintTime>95 && !rendering) requestFrame(Core.nearestIndex(data.time_s,target));
    if (target >= data.time_s.at(-1)) {requestFrame(data.time_s.length-1); pause(); return;}
    requestAnimationFrame(animate);
  }

  async function switchRobot(id) {
    requestedRobotId=id;
    if (!ready || switching) return;
    pause();
    switching = true;
    while (rendering) await new Promise(resolve=>setTimeout(resolve,25));
    if(linkPromise) await linkPromise;
    const time = data.time_s[currentIndex];
    robot = registry.robots.find(entry=>entry.id===requestedRobotId);
    data = datasets[robot.id];
    currentIndex = Core.nearestIndex(data.time_s,time);
    wantedIndex = currentIndex;
    robotCards();
    element('slider').max=data.time_s.length-1;
    positionRoundTripMarker();
    await stateChart();
    await propulsionChart();
    switching = false;
    await renderFrame();
    if(requestedRobotId!==robot.id) await switchRobot(requestedRobotId);
  }

  function fail(error) {
    console.error(error);
    pause();
    element('status').textContent='页面异常，请刷新';
    rendering=false;
  }

  async function initialize() {
    window.CameraView.initialize(payload,options);
    for (const entry of registry.robots) {
      const option = document.createElement('option'); option.value=entry.id; option.textContent=entry.name;
      element('robotSelect').append(option);
    }
    robotCards();
    element('slider').max=data.time_s.length-1;
    positionRoundTripMarker();
    updateState();
    const item=state();
    await Plotly.newPlot(element('scene'),sceneTraces(item),sceneLayout(item),options);
    const riskLayout=plotLayout();
    riskLayout.margin.r=18; riskLayout.showlegend=false;
    riskLayout.xaxis.title={text:'沿中心线弧长 (mm)',font:{size:10}};
    riskLayout.yaxis={...riskLayout.yaxis,title:{text:'归一化风险',font:{size:10}},range:[-.05,1.08]};
    riskLayout.shapes=cursorLine(0);
    await Plotly.newPlot(element('riskPlot'),[{type:'scatter',mode:'lines+markers',x:source.arc_s_mm,y:source.risk_norm,line:{color:'#b9c5d1',width:1.1},marker:{color:source.risk_norm,colorscale:Core.colorscale,cmin:0,cmax:1,size:4},customdata:source.arc_s_mm.map((arc,index)=>[source.total_cost_raw[index],source.risk_norm[index],arc,index]),hovertemplate:'初始风险 %{customdata[0]:.5f}<br>归一化风险 %{customdata[1]:.5f}<br>弧长 %{customdata[2]:.1f} mm<extra>原始中心线采样点</extra>'}],riskLayout,options);
    await stateChart();
    await propulsionChart();
    element('scene').on('plotly_hover',event=>{const point=event.points[0]; if(point.curveNumber===0) probe(point.customdata[3],'三维中心线');});
    element('riskPlot').on('plotly_hover',event=>probe(event.points[0].customdata[3],'拉直风险曲线'));
    element('slider').addEventListener('input',()=>{pause();requestFrame(Number(element('slider').value));});
    element('robotSelect').addEventListener('change',()=>switchRobot(element('robotSelect').value).catch(fail));
    element('showAll').addEventListener('change',updateTable);
    element('filter').addEventListener('input',updateTable);
    for(const id of ['showPath','showTrajectory','showRobot','viewMode','fieldScope','showField','showLines','showSlice','slicePlane','fieldMode','showForce','showTangent','showSteering','showTorque','showGradient','showPropulsion','showTurningPanel']) element(id).addEventListener('change',()=>requestFrame(currentIndex));
    element('showAppliedForce').addEventListener('change',()=>propulsionChart().catch(fail));
    element('clearProbe').addEventListener('click',clearProbe);
    element('robotScale').addEventListener('change',()=>requestFrame(currentIndex));
    element('chartMetric').addEventListener('change',()=>stateChart().catch(fail));
    element('restart').addEventListener('click',()=>{pause();requestFrame(0);});
    element('play').addEventListener('click',async()=>{
      if(playing){pause();return;}
      if(currentIndex===data.time_s.length-1){wantedIndex=0;await renderFrame();}
      playing=true;playTime=data.time_s[currentIndex];playWall=performance.now();element('play').textContent='Ⅱ 暂停';requestAnimationFrame(animate);
    });
    element('playbackSpeed').addEventListener('change',()=>{if(playing){playTime=data.time_s[currentIndex];playWall=performance.now();}});
    element('downloadCsv').addEventListener('click',()=>{
      const keys=Object.keys(data),rows=[keys.join(',')];
      for(let index=0;index<data.time_s.length;index++) rows.push(keys.map(key=>data[key][index]).join(','));
      const url=URL.createObjectURL(new Blob([rows.join('\r\n')],{type:'text/csv;charset=utf-8'}));
      const link=document.createElement('a');link.href=url;link.download=robot.csv;link.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
    });
    element('openDocs').addEventListener('click',()=>element('docsDialog').showModal());
    element('closeDocs').addEventListener('click',()=>element('docsDialog').close());
    ready=true;
    window.CameraView.update(robot.id,currentIndex);
    element('status').textContent='加载完成';
    window.viewer = {get ready(){return ready;},get robot(){return robot;},get currentIndex(){return currentIndex;},get busy(){return rendering||switching||linkPromise!==null;},get state(){return state();},get lastProbeIndex(){return lastProbeIndex;},get probeReplayIndex(){return probeReplayIndex();},registry,datasets,source};
  }
  initialize().catch(fail);
})();
