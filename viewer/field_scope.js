(function(root){
  const core=root.ViewerCore;
  const coordinates=point=>Object.fromEntries(core.axes.map((axis,index)=>[axis,point[index]]));

  function create(source,registry){
    const indices=[0];
    let previous=0;
    for(let index=1;index<source.arc_s_mm.length-1;index++){
      if(source.arc_s_mm[index]-source.arc_s_mm[previous]>=24){indices.push(index);previous=index;}
    }
    indices.push(source.arc_s_mm.length-1);
    const centers=indices.map(index=>core.axes.map(axis=>source[`${axis}_mm`][index]));
    const points=indices.flatMap((index,pointIndex)=>{
      const center=centers[pointIndex];
      return [center,...['n','b'].flatMap(prefix=>[-1,1].map(sign=>center.map((value,coordinate)=>value+sign*8*source[prefix+core.axes[coordinate]][index])))];
    });
    const centerBounds=core.axes.map((axis,index)=>[Math.min(...centers.map(point=>point[index]))-18,Math.max(...centers.map(point=>point[index]))+18]);
    const centerline={points,bounds:centerBounds,contains:point=>centers.some(center=>center.reduce((sum,value,index)=>sum+(value-point[index])**2,0)<=18**2)};
    const bounds=registry.field_world_bounds_mm;
    const wholePoints=[];
    const counts=[9,7,13];
    for(let first=0;first<counts[0];first++)for(let second=0;second<counts[1];second++)for(let third=0;third<counts[2];third++){
      wholePoints.push([first,second,third].map((index,coordinate)=>bounds[coordinate][0]+index*(bounds[coordinate][1]-bounds[coordinate][0])/(counts[coordinate]-1)));
    }
    const whole={points:wholePoints,bounds,contains:point=>point.every((value,index)=>value>=bounds[index][0]-1e-8&&value<=bounds[index][1]+1e-8)};
    return {centerline,whole};
  }

  const origin=state=>core.axes.map(axis=>state[`actual_position_${axis}_mm`]);
  const toWorld=(state,point)=>point.map((value,index)=>value+origin(state)[index]);
  const toLocal=(state,point)=>point.map((value,index)=>value-origin(state)[index]);
  const outsideRobot=(state,robot,point)=>{
    const body=['axis','normal','binormal'].map((basis,index)=>core.axes.reduce((sum,axis,coordinate)=>sum+point[coordinate]*(state[`actual_${basis}_${axis}_unit`]??Number(index===coordinate)),0));
    return !core.insideRobot(robot.shape,body,robot.length_mm,robot.diameter_mm);
  };

  function grid(state,robot,domain,mode,bias){
    const result={x:[],y:[],z:[],u:[],v:[],w:[],magnitude:[]};
    for(const worldPoint of domain.points){
      const point=toLocal(state,worldPoint);
      if(!outsideRobot(state,robot,point))continue;
      const field=core.fieldVector(state,point,mode,bias);
      core.axes.forEach((axis,index)=>result[axis].push(point[index]));
      ['u','v','w'].forEach((axis,index)=>result[axis].push(field[index]));
      result.magnitude.push(Math.hypot(...field));
    }
    return result;
  }

  function slice(state,robot,domain,mode,bias,plane,count=25){
    const selected={XY:[0,1],XZ:[0,2],YZ:[1,2]}[plane];
    const center=origin(state);
    const result={x:[],y:[],z:[],magnitude:[]};
    for(let row=0;row<count;row++){
      const points=[],values=[];
      for(let column=0;column<count;column++){
        const point=[0,0,0];
        for(const [coordinate,sample]of [[selected[0],column],[selected[1],row]]){
          point[coordinate]=domain.bounds[coordinate][0]-center[coordinate]+sample*(domain.bounds[coordinate][1]-domain.bounds[coordinate][0])/(count-1);
        }
        const valid=domain.contains(toWorld(state,point))&&outsideRobot(state,robot,point);
        values.push(valid?Math.hypot(...core.fieldVector(state,point,mode,bias)):null);
        points.push(point);
      }
      core.axes.forEach((axis,index)=>result[axis].push(points.map((point,column)=>axis==='z'&&values[column]===null?null:point[index])));
      result.magnitude.push(values);
    }
    return result;
  }

  function streamlines(state,robot,domain,mode,bias){
    const result={x:[],y:[],z:[],magnitude:[]};
    const step=Math.max(1.6,Math.min(...domain.bounds.map(range=>range[1]-range[0]))/40);
    const unit=point=>{
      const field=core.fieldVector(state,point,mode,bias),length=Math.hypot(...field);
      return length>1e-10?field.map(value=>value/length):null;
    };
    const stride=Math.max(1,Math.ceil(domain.points.length/48));
    for(let seed=0;seed<domain.points.length;seed+=stride){
      const start=toLocal(state,domain.points[seed]);
      if(!outsideRobot(state,robot,start))continue;
      for(const sign of [-1,1]){
        let point=start.slice();
        const points=[point];
        for(let iteration=0;iteration<120;iteration++){
          const direction=unit(point);
          if(!direction)break;
          const midpoint=point.map((value,index)=>value+step*.5*sign*direction[index]);
          const nextDirection=unit(midpoint);
          if(!nextDirection)break;
          const next=point.map((value,index)=>value+step*sign*nextDirection[index]);
          if(!domain.contains(toWorld(state,next))||!outsideRobot(state,robot,next))break;
          points.push(next);point=next;
        }
        if(points.length<3)continue;
        for(const sample of points){
          const values=coordinates(sample);
          core.axes.forEach(axis=>result[axis].push(values[axis]));
          result.magnitude.push(Math.hypot(...core.fieldVector(state,sample,mode,bias)));
        }
        core.axes.forEach(axis=>result[axis].push(null));result.magnitude.push(0);
      }
    }
    return result;
  }

  root.FieldScope={create,grid,slice,streamlines};
})(window);
