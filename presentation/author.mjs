export async function buildDeck(presentation, data, fs, root) {
  presentation.slides.keep([]);
  presentation.theme.defaultFont='Arial';
  presentation.theme.colorScheme={name:'Academic navy and teal',themeColors:{accent1:'#236D73',accent2:'#B36A43',accent3:'#9C813E',accent4:'#536D82',accent5:'#74667A',accent6:'#789382',bg1:'#F8F7F3',bg2:'#FFFFFF',tx1:'#142B3C',tx2:'#53646A',dk1:'#142B3C',dk2:'#263E4E',lt1:'#FFFFFF',lt2:'#E6E8E2',hlink:'#236D73',folHlink:'#74667A'}};
  const font='Arial', ink='#142B3C', muted='#53646A', teal='#236D73', bg='#F8F7F3';
  function text(s,name,value,x,y,w,h,size=28,color=ink,bold=false,typeface=font) {
    const sh=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
    sh.text=value;sh.text.style={fontSize:size,typeface,color,bold,insets:0,wrap:'square',autoFit:'none',verticalAlignment:'top'};return sh;
  }
  function bodyLines(s,items,x=72,y=200,w=1120,gap=97,size=30) {
    items.forEach((v,i)=>text(s,'body-'+i,v,x,y+i*gap,w,gap-15,size));
  }
  function table(s,d) {
    const n=d.rows.length, h=n>=7?448:n===6?420:n===5?370:330;
    const y=188;
    const tb=s.tables.add({rows:n,columns:d.rows[0].length,left:72,top:y,width:1120,height:h,columnWidths:d.colWidths,values:d.rows});
    tb.styleOptions={headerRow:false,bandedRows:false};
    tb.cells.block({row:0,column:0,rowCount:n,columnCount:d.rows[0].length}).assign({fill:bg,textStyle:{fontSize:24,typeface:font,color:ink},margins:{left:14,right:12,top:9,bottom:9}});
    tb.borders.assign({style:'solid',fill:'#DFE3DC',width:.8});
    tb.cells.block({row:0,column:0,rowCount:1,columnCount:d.rows[0].length}).assign({fill:ink,textStyle:{fontSize:24,typeface:font,color:'#FFFFFF',bold:true}});
    for(let i=0;i<n;i++)tb.rows[i].height=h/n;
  }
  function chart(s,d) {
    const c=data.charts[d.chart];
    const isRoc=c.type==='scatter';
    const axisText={typeface:font,fontSize:21,fill:ink};
    const options={name:'chart-'+d.chart,position:{left:90,top:180,width:1080,height:425},series:c.series.map(series=>({...series,values:series.values?.map(v=>v==null?v:Number(v.toPrecision(12))),...(series.xValues?{xValues:series.xValues.map(v=>v==null?v:Number(v.toPrecision(12)))}:{})})),hasLegend:c.series.length>1,
      chartFill:bg,plotAreaFill:bg,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},
      legend:{position:'bottom',overlay:false,textStyle:{typeface:font,fontSize:20,fill:ink}},
      xAxis:{visible:true,textStyle:axisText,line:{fill:'#AEB9B6',width:1},majorGridlines:null},
      yAxis:{visible:true,min:c.min,max:c.max,numberFormatCode:c.format,textStyle:axisText,line:{fill:'none',width:0},majorGridlines:{fill:'#DDE2DC',width:1},title:{text:c.axis,textStyle:{typeface:font,fontSize:21,fill:ink}}},
      dataLabels:{showValue:!isRoc,position:'outEnd',numberFormatCode:d.chart==='quality'?'0.0000':(c.format.includes('%')?'0.00%':c.format),textStyle:{typeface:font,fontSize:23,fill:ink,bold:true}}
    };
    if(isRoc){options.scatterOptions={style:'line'};options.xAxis={...options.xAxis,min:0,max:1,numberFormatCode:'0.0',title:{text:'Clean false-positive rate',textStyle:{typeface:font,fontSize:21,fill:ink}}};options.position.height=430;}
    else{options.categories=c.categories;options.barOptions={direction:'column',grouping:'clustered',gapWidth:100};}
    s.charts.add(c.type,options);
  }
  for(const d of data.slides){
    const s=presentation.slides.add({width:1280,height:720});s.background.fill=d.kind==='cover'?ink:bg;
    s.speakerNotes.text=d.notes;
    if(d.kind==='cover'){
      text(s,'cover-title',d.title,72,132,1120,224,58,'#FFFFFF',true);
      text(s,'cover-subtitle',d.body[0],76,403,1100,52,31,'#D2E2E3');
      text(s,'cover-team',d.body[1],76,469,1100,48,29,'#FFFFFF');
      text(s,'cover-context',d.body[2]+'\n'+d.body[3],76,583,1100,72,23,'#BCCDD0');
      continue;
    }
    text(s,'headline',d.title,72,52,1130,114,44,ink,true);
    if(d.kind==='comparison'){d.body.forEach((group,i)=>{const x=i===0?72:660;text(s,'model-group-'+i,group[0],x,197,530,50,29,teal,true);group.slice(1).forEach((v,j)=>text(s,'model-item-'+i+'-'+j,v,x,282+j*90,540,67,j===0?34:28,ink,j===0));});}
    else if(d.kind==='table')table(s,d);
    else if(d.kind==='chart')chart(s,d);
    else if(d.kind==='code'){
      text(s,'code-excerpt',d.code,72,211,740,338,d.number===12?24:24,ink,false,'Menlo');
      d.callouts.forEach((v,i)=>text(s,'code-explanation-'+i,v,850,205+i*123,350,110,27,teal,i===0));
      text(s,'code-path',d.number===6?'unixcoder_adapter.py: forward()':'unixcoder_adapter.py: features()',72,582,1100,34,22,muted,false,'Menlo');
    }
    else if(d.kind==='codepair'){
      text(s,'before-label','Original input (excerpt)',72,177,510,40,27,teal,true);
      text(s,'after-label','Renamed input (excerpt)',660,177,540,40,27,teal,true);
      text(s,'before-code',d.leftCode,72,236,540,348,22,ink,false,'Menlo');
      text(s,'after-code',d.rightCode,660,236,540,200,22,ink,false,'Menlo');
      text(s,'before-output-label','Model summary',72,460,540,35,24,teal,true);
      text(s,'after-output-label','Model summary',660,460,540,35,24,teal,true);
      text(s,'before-output',d.leftPrediction,72,506,540,95,25,ink);
      text(s,'after-output',d.rightPrediction,660,506,540,95,25,ink);
    }
    else if(d.kind==='methods'){
      d.body.forEach((row,i)=>{const y=194+i*214;text(s,'method-name-'+i,row[0],72,y,1100,44,31,teal,true);text(s,'method-equation-'+i,row[1],72,y+55,1100,55,35,ink);text(s,'method-description-'+i,row[2],72,y+126,1100,46,27,muted);});
    }
    else if(d.kind==='formula'){
      d.body.forEach((row,i)=>{const y=194+i*105;text(s,'metric-label-'+i,row[0],72,y,370,65,26,teal,true);text(s,'metric-equation-'+i,row[1],446,y,754,72,28,ink);});
    }
    else if(d.kind==='image'){
      s.images.add({name:'measured-training-history',dataUrl:'data:image/png;base64,'+(await fs.readFile(root+'/'+d.image)).toString('base64'),contentType:'image/png',alt:'Recorded clean UniXcoder training loss and validation BLEU with selected epoch',fit:'contain',position:{left:65,top:210,width:842,height:340}});
      bodyLines(s,d.body,942,207,265,115,25);
    }
    else if(d.kind==='demo'){
      bodyLines(s,d.body,72,193,1110,99,30);
      text(s,'live-demo-command',d.command,72,544,1110,52,26,teal,true,'Menlo');
    }
    else bodyLines(s,d.body,72,205,1115,100,31);
    if(d.bottom)text(s,'evidence-context',d.bottom,72,624,1120,43,20,muted);
    text(s,'presenter-and-page',d.owner+'    '+String(d.number).padStart(2,'0')+' / '+data.slides.length,72,684,1120,24,16,muted);
  }
}
