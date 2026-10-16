(function(root){
  const core = root.ViewerCore;
  const colorRange = (registry,mode,scope='local') => registry.field_scope_ranges?.[scope]?.[mode]||registry.field_mode_ranges[mode];

  function vectorGlyph(vector, color, name, scale, visible, title, valueText, minimumLength = 0) {
    const length = Math.hypot(...vector);
    const direction = length>1e-15 ? vector.map(value=>value/length) : [0,0,0];
    const drawnLength = length>1e-15?Math.min(Math.max(length*scale,minimumLength),31):0;
    const tip = direction.map(value=>value*drawnLength);
    return [
      {type:'scatter3d',mode:'lines',x:[0,tip[0]*.82],y:[0,tip[1]*.82],z:[0,tip[2]*.82],line:{color,width:6},hovertemplate:`${title}: ${valueText}<extra>${name}</extra>`,showlegend:false,visible:visible&&length>1e-15,name},
      {type:'cone',x:[tip[0]],y:[tip[1]],z:[tip[2]],u:[direction[0]],v:[direction[1]],w:[direction[2]],sizemode:'absolute',sizeref:3.5,anchor:'tip',colorscale:[[0,color],[1,color]],showscale:false,hovertemplate:`${title}: ${valueText}<extra>视觉缩放，长度${minimumLength}–31 mm</extra>`,visible:visible&&length>1e-15,name},
    ];
  }

  function traces(item,robot,registry,settings) {
    const mode=settings.mode, bias=robot.config.bias_field_T*1000;
    const scope=settings.scope||'local', domain=settings.domain;
    const grid=scope==='local'?core.fieldGrid(item,robot,30,5,mode,bias):root.FieldScope.grid(item,robot,domain,mode,bias);
    const range=colorRange(registry,mode,scope), unit=mode==='turning'?'μT':'mT';
    const originalScale=mode==='turning'?(grid.magnitude[0]>1e-10?6.5/grid.magnitude[0]:0):{total:.4,residual:.58}[mode];
    const scale=1.15*(scope==='local'?originalScale:Math.min(originalScale,10/Math.max(1e-10,...grid.magnitude)));
    const shafts={x:[],y:[],z:[],color:[]},tips={x:[],y:[],z:[]};
    for(let index=0;index<grid.x.length;index++){
      for(const [axis,component] of [['x','u'],['y','v'],['z','w']]){
        shafts[axis].push(grid[axis][index],grid[axis][index]+grid[component][index]*scale*.75,null);
        tips[axis].push(grid[axis][index]+grid[component][index]*scale);
      }
      shafts.color.push(grid.magnitude[index],grid.magnitude[index],0);
    }
    const colorbar={title:{text:`|B| (${unit})`,font:{size:10}},len:.36,y:.25,thickness:10,tickfont:{size:9},x:1.005};
    const lines=settings.lines?(scope==='local'?core.streamlines(item,robot,mode,bias):root.FieldScope.streamlines(item,robot,domain,mode,bias)):{x:[],y:[],z:[],magnitude:[]};
    const slice=scope==='local'?core.fieldSlice(item,robot,mode,bias,settings.plane,25,30):root.FieldScope.slice(item,robot,domain,mode,bias,settings.plane);
    const force=core.forceProjection(item,'applied');
    const steering=core.steeringVector(item).map(value=>value*1000);
    const torque=core.axes.map(axis=>item[`applied_torque_${axis}_uN_m`]);
    const toText=(vector,units)=>`(${vector.map(value=>value.toExponential(3)).join(', ')}) ${units}`;
    const sliceData=slice.magnitude.map((row,rowIndex)=>row.map((magnitude,columnIndex)=>[magnitude,slice.x[rowIndex][columnIndex],slice.y[rowIndex][columnIndex],slice.z[rowIndex][columnIndex]]));
    const localTraces = [
      {type:'mesh3d',...core.transformedRobot(item,robot,settings.robotScale||1,true),color:'#e9a640',opacity:1,lighting:{ambient:.55,diffuse:.75,specular:.4},hoverinfo:'skip',showlegend:false,visible:settings.robot!==false,name:robot.name},
      {type:'scatter3d',mode:'lines',x:shafts.x,y:shafts.y,z:shafts.z,line:{color:shafts.color,colorscale:core.colorscale,cmin:range[0],cmax:range[1],width:scope==='local'?2.7:1.75},hoverinfo:'skip',showlegend:false,visible:settings.arrows,name:'场箭头杆'},
      {type:'cone',...tips,u:grid.u,v:grid.v,w:grid.w,colorscale:core.colorscale,cmin:range[0],cmax:range[1],sizemode:'raw',sizeref:Math.max(scale*.21,1e-12),anchor:'tip',showscale:settings.arrows,colorbar,customdata:grid.magnitude,hovertemplate:`场强 |B| %{customdata:.5g} ${unit}<br>Bx=%{u:.5g} ${unit}<br>By=%{v:.5g} ${unit}<br>Bz=%{w:.5g} ${unit}<extra>${mode==='turning'?'中心转向分量均匀展示':scope==='local'?'局部仿射场':'仿射模型外推 · 非真实线圈全场'}</extra>`,visible:settings.arrows&&scale>0,name:'场箭头尖'},
      {type:'scatter3d',mode:'lines',x:lines.x,y:lines.y,z:lines.z,line:{color:lines.magnitude,colorscale:core.colorscale,cmin:range[0],cmax:range[1],width:2},showscale:false,hoverinfo:'skip',showlegend:false,visible:settings.lines,name:'数值积分流线'},
      {type:'surface',x:slice.x,y:slice.y,z:slice.z,surfacecolor:slice.magnitude,colorscale:core.colorscale,cmin:range[0],cmax:range[1],opacity:.5,connectgaps:false,showscale:settings.slice&&!settings.arrows,colorbar,customdata:sliceData,hovertemplate:`场强 |B| %{customdata[0]:.5g} ${unit}<br>Δx=%{customdata[1]:.1f} mm<br>Δy=%{customdata[2]:.1f} mm<br>Δz=%{customdata[3]:.1f} mm<extra>${settings.plane}切片</extra>`,visible:settings.slice,name:'场强热力切片'},
      ...vectorGlyph(force.vector,'#19825f','总磁力',3,settings.force,'实际总磁力',toText(force.vector,'mN')),
      ...vectorGlyph(force.projected,'#dc7820','沿路径推进',10,settings.tangent,'沿路径有符号分量',force.parallel.toFixed(5)+' mN'),
      ...vectorGlyph(steering,'#8055bc','转向分量',12,settings.steering,'垂直磁场',toText(steering,'μT'),6),
      ...vectorGlyph(torque,'#b82167','可控转矩',450,settings.torque,'实际转矩',toText(torque,'μN·m'),6),
    ];
    const origin=core.axes.map(axis=>item[`actual_position_${axis}_mm`]);
    return localTraces.map(trace=>core.translateTrace(trace,origin));
  }

  root.FieldView={traces,colorRange};
})(window);
