(function(root){
  let payload,renderer,scene,camera,light,canvas,options,currentRobot='capsule',currentIndex=0;
  const element=id=>document.getElementById(id);
  const number=(value,digits=2)=>value===null||value===undefined?'—':Number(value).toFixed(digits);
  const plus=(first,second,scale=1)=>first.map((value,index)=>value+second[index]*scale);
  const modePrefix={capsule:'fixed',capsule_slowdown:'slowdown',capsule_wait:'wait',capsule_cooperative:'coordinated',capsule_withdrawal:'withdrawal',capsule_roundtrip:'roundtrip'};
  const trackKey=robot=>robot==='capsule_roundtrip'?'roundtrip_new':robot==='capsule_withdrawal'?'withdrawal_new':(modePrefix[robot]||'fixed')+'_'+element('cameraMethod').value;
  const rotation=quaternion=>{
    const matrix=new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion(...quaternion));
    return [[matrix.elements[0],matrix.elements[1],matrix.elements[2]],[matrix.elements[4],matrix.elements[5],matrix.elements[6]],[matrix.elements[8],matrix.elements[9],matrix.elements[10]]];
  };
  const frameAt=(robot,index)=>{
    const key=trackKey(robot),track=payload.cameras[key];
    const motion=payload.datasets[robot];
    const motionTime=motion.time_s[Math.min(index,motion.time_s.length-1)];
    index=ViewerCore.nearestIndex(track.time_s,motionTime);
    return {key,track,index,origin:track.position[index],axes:rotation(track.quaternion[index]),time:track.time_s[index]};
  };
  function meshTrace(frame){
    const geometry=payload.geometry,vertices=geometry.points;
    let color='#b8a7a6',facecolor;
    if(element('showCoverage').checked&&(frame.key==='withdrawal_new'||frame.key==='roundtrip_new')){
      const masks=payload.withdrawalCoverage.masks;
      const roundTrip=frame.key==='roundtrip_new';
      const added=masks.withdrawal_first_seen_s;
      const insertedAt=payload.evaluation.methods.slowdown_new.first_seen_s;
      const joinTime=roundTrip?payload.roundTrip.join_time_s:0;
      facecolor=geometry.faces.map((face,index)=>{
        const seenOnInsertion=roundTrip?insertedAt[index]>=0&&insertedAt[index]<=Math.min(frame.time,joinTime):masks.insertion_covered[index];
        const withdrawalTime=added[index]>=0?added[index]+joinTime:-1;
        if(seenOnInsertion)return '#45a798';
        if(added[index]>=0&&withdrawalTime<=frame.time)return '#338bd1';
        return masks.fold_hidden[index]?'#4c5361':'#df6260';
      });
      color=undefined;
    }else if(element('showCoverage').checked){
      const method=payload.evaluation.methods[frame.key];
      const faceCount=geometry.faces.length;
      facecolor=method.first_seen_s.slice(0,faceCount).map((atTime,index)=>atTime<0||atTime>frame.time?'#df6260':method.repeat_seen_s[index]>=0&&method.repeat_seen_s[index]<=frame.time?'#dcab42':'#45a798');
      color=undefined;
    }
    return {type:'mesh3d',x:vertices.map(point=>point[0]),y:vertices.map(point=>point[1]),z:vertices.map(point=>point[2]),
      i:geometry.faces.map(face=>face[0]),j:geometry.faces.map(face=>face[1]),k:geometry.faces.map(face=>face[2]),
      color,facecolor,opacity:element('showCoverage').checked?.58:Number(element('bowelOpacity').value),
      lighting:{ambient:.8,diffuse:.4,specular:.15},hoverinfo:'skip',showlegend:false,visible:element('showBowel').checked,name:'肠道内壁'};
  }
  function traces(robot,index){
    const frame=frameAt(robot,index),settings=payload.cameras.settings;
    const [forward,left,up]=frame.axes;
    const length=Number(element('frustumLength').value),vertical=length*Math.tan(settings.fov_vertical_deg*Math.PI/360),horizontal=vertical*settings.aspect;
    const center=plus(frame.origin,forward,length);
    const corners=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([side,height])=>plus(plus(center,left,side*horizontal),up,height*vertical));
    const segments=[];
    corners.forEach(point=>segments.push([frame.origin,point]));
    corners.forEach((point,index)=>segments.push([point,corners[(index+1)%4]]));
    const coordinates={x:[],y:[],z:[]};
    segments.forEach(segment=>['x','y','z'].forEach((axis,component)=>coordinates[axis].push(segment[0][component],segment[1][component],null)));
    const solid=[frame.origin,...corners];
    const target=frame.track.target[frame.index];
    return [meshTrace(frame),
      {type:'scatter3d',mode:'lines',...coordinates,line:{color:'#176fc3',width:3},hoverinfo:'skip',showlegend:false,visible:element('showFrustum').checked,name:'摄像头视锥'},
      {type:'mesh3d',x:solid.map(point=>point[0]),y:solid.map(point=>point[1]),z:solid.map(point=>point[2]),i:[0,0,0,0],j:[1,2,3,4],k:[2,3,4,1],color:'#4c9dde',opacity:.09,hoverinfo:'skip',showlegend:false,visible:element('showFrustum').checked,name:'视锥范围'},
      {type:'scatter3d',mode:'lines+markers',x:[frame.origin[0],target[0]],y:[frame.origin[1],target[1]],z:[frame.origin[2],target[2]],line:{color:'#00a6a0',width:4,dash:'dot'},marker:{size:[4,5],color:'#00a6a0'},hoverinfo:'skip',showlegend:false,visible:element('showCameraTarget').checked,name:'前瞻目标'}];
  }
  function render(frame){
    const width=Math.max(200,canvas.clientWidth),height=Math.round(width/payload.cameras.settings.aspect);
    renderer.setSize(width,height,false);
    camera.position.set(...frame.origin);
    camera.up.set(...frame.axes[2]);
    camera.lookAt(...plus(frame.origin,frame.axes[0],10));
    light.position.copy(camera.position);
    renderer.render(scene,camera);
  }
  function evaluationPair(){
    if(currentRobot==='capsule_withdrawal')return ['withdrawal_new','withdrawal_new'];
    const prefix=modePrefix[currentRobot]||'fixed';
    return element('evaluationMode').value==='system'?['fixed_endoscopy',prefix+'_new']:[prefix+'_endoscopy',prefix+'_new'];
  }
  function metricAt(method,time){
    if(method==='roundtrip_new')method='withdrawal_new';
    const rows=payload.evaluation.methods[method].rows;
    let index=ViewerCore.nearestIndex(rows.map(row=>row.time_s),Math.min(time,rows.at(-1).time_s));
    if(rows[index].time_s>time&&index>0)index--;
    return rows[index];
  }
  function update(robot,index){
    const motionChanged=robot!==currentRobot;
    currentRobot=robot;currentIndex=index;
    const frame=frameAt(robot,index);render(frame);
    element('cameraMethod').disabled=robot==='capsule_withdrawal'||robot==='capsule_roundtrip';
    if(robot==='capsule_withdrawal'||robot==='capsule_roundtrip')element('cameraMethod').value='new';
    const angles=frame.track.joint_angle_deg?.[frame.index]||[0,0,0];
    const velocity=frame.track.joint_velocity_deg_s?.[frame.index];
    const acceleration=frame.track.joint_acceleration_deg_s2?.[frame.index];
    element('cameraReadout').innerHTML=`<span>相机位置 <b>${frame.origin.map(value=>number(value,1)).join(' / ')} mm</b></span><span>偏航 / 俯仰 / 滚转 <b>${angles.map(value=>number(value,1)).join(' / ')} °</b></span><span>前瞻 <b>${number(frame.track.preview_mm[frame.index],1)} mm</b></span><span>目标偏差 <b>${number(frame.track.aim_error_deg?.[frame.index],1)} °</b></span>`;
    if(frame.track.tracking_error_deg)element('cameraReadout').innerHTML+=`<span>可达姿态跟踪滞后 <b>${number(frame.track.tracking_error_deg[frame.index],1)} °</b></span><span>锥外剩余误差 <b>${number(frame.track.cone_residual_deg[frame.index],1)} °</b></span>`;
    const motion=payload.cameraMotion[frame.key];
    const roll=motion.roll_rate_deg_s[frame.index];
    element('cameraJointReadout').textContent=(velocity?`关节角速度 ${velocity.map(value=>number(value,1)).join(' / ')} °/s · 角加速度 ${acceleration.map(value=>number(value,1)).join(' / ')} °/s²`:'Endoscopy 默认相机轨迹 · 不施加新算法关节限幅')+` · 光轴滚转速率 ${number(roll,2)} °/s`;
    const waiting=payload.datasets[robot].camera_wait_planned?.[frame.index]>.5;
    const roundTrip=payload.roundTrip;
    const phase=robot==='capsule_roundtrip'?(frame.time<roundTrip.join_time_s?'进镜 · 只减速':Math.abs(frame.time-roundTrip.join_time_s)<1e-6?'交接点 · 即将退镜':'退镜 · 连续补拍'):robot==='capsule_withdrawal'?'退镜 · 无停留':waiting?'跟踪等待（离线）':element('cameraMethod').value==='new'?'前瞻稳定':'Endoscopy';
    element('cameraStamp').textContent=`${number(frame.time)} s · ${phase}`;
    const metrics=metricAt(frame.key,frame.time);
    if(robot==='capsule_roundtrip'||robot==='capsule_withdrawal'){
      const summary=payload.withdrawalCoverage.summary;
      const entries=[['insertion_covered_area_mm2','当前累计覆盖','mm²'],['withdrawal_added_area_mm2','退镜新增补拍','mm²'],['coverage_pct','全肠壁覆盖率','%']];
      element('coverageCards').innerHTML=entries.map(([key,label,unit])=>`<div class="coverage-card"><span>${label}</span><strong>${number(metrics[key]??summary[key])}<small> ${unit}</small></strong></div>`).join('');
      const delta=metrics.withdrawal_added_area_mm2||0;
      const gainPoints=metrics.coverage_gain_percentage_points??(100*delta/Math.max(summary.total_lumen_area_mm2,1e-9));
      element('coverageStamp').textContent=`${phase} · 退镜新增 ${number(delta)} mm²，相当于完整肠壁覆盖率 +${number(gainPoints,2)} 个百分点 · 完整肠壁 ${number(summary.total_lumen_area_mm2)} mm²`;
    }else{
      const entries=[['coverage_pct','已观察覆盖'],['unseen_pct','未观察区域'],['effective_pct','当前有效视野'],['repeat_pct','重复观察率']];
      element('coverageCards').innerHTML=entries.map(([key,label])=>`<div class="coverage-card"><span>${label}</span><strong>${number(metrics[key])}<small>${metrics[key]===null?'':'%'}</small></strong></div>`).join('');
      element('coverageStamp').textContent=`离线评价采样 ${number(metrics.time_s)} s · 路径完成 ${number(metrics.completion_pct,1)}%`;
    }
    updateComparison();
    if(motionChanged)chart();
    const cursorShape={type:'line',x0:frame.time,x1:frame.time,y0:0,y1:1,yref:'paper',line:{color:'#35495e',width:1,dash:'dot'}};
    const handoffShape={type:'line',x0:roundTrip?.join_time_s,x1:roundTrip?.join_time_s,y0:0,y1:1,yref:'paper',line:{color:'#d23b39',width:3,dash:'dash'}};
    Plotly.relayout(element('evaluationPlot'),{shapes:robot==='capsule_roundtrip'?[handoffShape,cursorShape]:[cursorShape]});
  }
  function updateComparison(){
    if(currentRobot==='capsule_roundtrip'||currentRobot==='capsule_withdrawal'){
      const summary=payload.withdrawalCoverage.summary;
      const rows=[['进镜覆盖面积','insertion_covered_area_mm2','mm²'],['退镜新增补拍','withdrawal_added_area_mm2','mm²'],['进退镜累计覆盖','final_covered_area_mm2','mm²'],['完整肠壁总面积','total_lumen_area_mm2','mm²'],['当前未覆盖面积','uncovered_area_mm2','mm²'],['其中：候选角度下不可见','fold_hidden_area_mm2','mm²'],['退镜补拍提升','coverage_gain_percentage_points','百分点']];
      element('evaluationTable').innerHTML='<table><thead><tr><th>全表面积统计（褶皱后区域保留在分母）</th><th>结果</th><th>单位</th></tr></thead><tbody>'+rows.map(([label,key,unit])=>`<tr><td>${label}</td><td>${number(summary[key])}</td><td>${unit}</td></tr>`).join('')+'</tbody></table>';
      element('motionComparison').innerHTML=`进退镜交界处位置残差 ${number(payload.roundTrip?.endpoint_position_gap_mm,3)} mm。退镜相机选点仍排除当前采样方向下不可见面片；面积统计不排除它们。`;
      element('evaluationNotice').textContent='新增面积按面片去重；褶皱后区域仍作为未覆盖面积计入完整肠壁统计分母。退镜选点仅针对采样中可见的漏拍区域。';
      return;
    }
    const pair=evaluationPair(),scope=element('evaluationSummary').value;
    const results=pair.map(method=>payload.evaluation.methods[method][scope]);
    const labels=[['coverage_pct','覆盖率','%'],['unseen_pct','未观察比例','%'],['effective_time_mean_pct','平均有效视野','%'],['repeat_pct','重复观察率','%'],['time_s','统计时刻','s'],['completion_pct','路径完成比例','%']];
    element('evaluationTable').innerHTML='<table><thead><tr><th>指标</th><th>Endoscopy</th><th>新算法</th><th>新 − 基线</th></tr></thead><tbody>'+labels.map(([key,label,unit])=>`<tr><td>${label}</td><td>${number(results[0][key])} ${unit}</td><td>${number(results[1][key])} ${unit}</td><td>${results.some(row=>row[key]===null)?'—':number(results[1][key]-results[0][key])} ${unit==='%'?'百分点':unit}</td></tr>`).join('')+'</tbody></table>';
    const auxiliary=[['roll_rate_p95_deg_s','光轴滚转速率 P95','°/s'],['angular_acceleration_p95_deg_s2','世界角加速度 P95','°/s²']];
    element('motionComparison').innerHTML=auxiliary.map(([key,label,unit])=>`<p>${label}：Endoscopy <b>${number(payload.cameraMotion[pair[0]][key])}</b> / 新算法 <b>${number(payload.cameraMotion[pair[1]][key])}</b> ${unit}</p>`).join('')+'<p>统计各自完整轨迹。世界角加速度包含有意转弯，不能单独视为抖动；需结合滚转、画面和覆盖判断。</p>';
    element('evaluationNotice').textContent=element('evaluationMode').value==='system'?'完整系统：共同时间预算必须结合完成比例；走完全程的收益不能忽略额外耗时。':'相机算法：固定同一位置、速度与时间，只改变相机朝向；默认基线无人工关键帧。';
  }
  async function chart(){
    if(currentRobot==='capsule_roundtrip'||currentRobot==='capsule_withdrawal'){
      const rows=payload.evaluation.methods.withdrawal_new.rows;
      const data=[
        {type:'scatter',mode:'lines',x:rows.map(row=>row.time_s),y:rows.map(row=>row.insertion_covered_area_mm2),name:'进镜累计覆盖',line:{color:'#45a798',width:2}},
        {type:'scatter',mode:'lines',x:rows.map(row=>row.time_s),y:rows.map(row=>row.withdrawal_added_area_mm2),name:'退镜新增面积',line:{color:'#338bd1',width:2}},
        {type:'scatter',mode:'lines',x:rows.map(row=>row.time_s),y:rows.map(row=>row.final_covered_area_mm2),name:'累计覆盖面积',line:{color:'#183d66',width:2.5}}
      ];
      await Plotly.react(element('evaluationPlot'),data,{margin:{l:56,r:18,t:12,b:38},height:245,paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'Segoe UI, Microsoft YaHei'},xaxis:{title:'进镜 → 退镜总时间 (s)',gridcolor:'#eef1f4'},yaxis:{title:'已覆盖面积 / 完整肠壁面积 (mm²)',rangemode:'tozero',gridcolor:'#eef1f4'},legend:{orientation:'h',y:1.16},shapes:[{type:'line',x0:payload.roundTrip.join_time_s,x1:payload.roundTrip.join_time_s,y0:0,y1:1,yref:'paper',line:{color:'#d23b39',width:3,dash:'dash'}}],annotations:[{x:payload.roundTrip.join_time_s,y:1,yref:'paper',text:'交接点 · 开始退镜',showarrow:false,yshift:13,font:{color:'#b92f31',size:11}}],uirevision:'evaluation'},options);
      updateComparison();
      return;
    }
    const key=element('evaluationMetric').value,pair=evaluationPair();
    const labels={coverage_pct:'覆盖率 (%)',unseen_pct:'未观察比例 (%)',effective_pct:'有效视野 (%)',repeat_pct:'重复观察率 (%)'};
    const data=pair.map((method,index)=>{const rows=payload.evaluation.methods[method].rows;return {type:'scatter',mode:'lines',x:rows.map(row=>row.time_s),y:rows.map(row=>row[key]),name:index?'新算法':'Endoscopy',line:{color:index?'#168c82':'#777e91',width:2.4},hovertemplate:'%{x:.2f} s<br>%{y:.2f}%<extra>%{fullData.name}</extra>'};});
    await Plotly.react(element('evaluationPlot'),data,{margin:{l:48,r:18,t:12,b:38},height:245,paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'Segoe UI, Microsoft YaHei'},xaxis:{title:'仿真时间 (s)',gridcolor:'#eef1f4'},yaxis:{title:labels[key],range:[0,100],gridcolor:'#eef1f4'},legend:{orientation:'h',y:1.16},uirevision:'evaluation'},options);
    updateComparison();
  }
  function initialize(value,plotOptions){
    payload=value;options=plotOptions;
    canvas=element('endoscopyCanvas');
    renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.5));
    renderer.setClearColor(0x11151b,1);
    renderer.outputColorSpace=THREE.SRGBColorSpace;
    scene=new THREE.Scene();
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(payload.geometry.points.flat(),3));
    geometry.setIndex(payload.geometry.faces.flat());geometry.computeVertexNormals();
    const material=new THREE.MeshPhongMaterial({color:0xc77f79,side:THREE.DoubleSide,shininess:18,specular:0x4e2b27});
    scene.add(new THREE.Mesh(geometry,material));
    scene.add(new THREE.AmbientLight(0xffe2d2,.65));
    light=new THREE.PointLight(0xffeee3,35,160,1);scene.add(light);
    const settings=payload.cameras.settings;
    camera=new THREE.PerspectiveCamera(settings.fov_vertical_deg,settings.aspect,settings.near_mm,settings.far_mm);
    for(const id of ['cameraMethod','showBowel','showCoverage','showFrustum','showCameraTarget','bowelOpacity','frustumLength'])element(id).addEventListener('change',()=>{const slider=element('slider');slider.dispatchEvent(new Event('input',{bubbles:true}));});
    for(const id of ['evaluationMode','evaluationMetric','evaluationSummary'])element(id).addEventListener('change',()=>chart());
    element('downloadEvaluation').addEventListener('click',()=>{
      const report={settings:payload.evaluation.settings,budget_s:payload.evaluation.budget_s,target_area_mm2:payload.evaluation.target_area_mm2,notice:payload.evaluation.notice,methods:Object.fromEntries(Object.entries(payload.evaluation.methods).map(([method,result])=>[method,{common_budget:result.common_budget,final:result.final}]))};
      const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='camera_evaluation.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
    });
    new ResizeObserver(()=>{if(payload)render(frameAt(currentRobot,currentIndex));}).observe(canvas);
    chart();
  }
  root.CameraView={initialize,traces,update,get frame(){return frameAt(currentRobot,currentIndex);},get evaluation(){return payload.evaluation;},get renderer(){return renderer;}};
})(window);
