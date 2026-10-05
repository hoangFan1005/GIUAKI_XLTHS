/** Build the three English decks from current exported test evidence.
 *
 * Workspace paths are relative to this file, so the repository can be moved.
 * Configure SKILL_DIR, RUNTIME_NODE_MODULES and RUNTIME_PYTHON for another runtime.
 * First run: python tools/export_slide_data.py
 * Then run this file with the bundled Node runtime.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const buildRoot = path.join(root, '_build', 'slides');
const dependencies = path.join(os.homedir(), '.cache', 'codex-runtimes',
  'codex-primary-runtime', 'dependencies');
const runtimeModules = process.env.RUNTIME_NODE_MODULES || path.join(dependencies, 'node', 'node_modules');
const runtimePython = process.env.RUNTIME_PYTHON || path.join(dependencies, 'python', 'python.exe');
const skill = process.env.SKILL_DIR || path.join(os.homedir(), '.codex', 'plugins', 'cache',
  'openai-primary-runtime', 'presentations', '26.909.12148', 'skills', 'presentations');
await fs.mkdir(buildRoot, {recursive: true});
const moduleLink = path.join(buildRoot, 'node_modules');
try { await fs.access(moduleLink); }
catch { await fs.symlink(runtimeModules, moduleLink, process.platform === 'win32' ? 'junction' : 'dir'); }
const require = createRequire(path.join(buildRoot, 'runtime.mjs'));
const {Presentation, PresentationFile} = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const {resolvePresentationFont, applyPresentationChartFont, finalizePresentation} = await import(
  pathToFileURL(path.join(skill, 'container_tools', 'artifact_tool_utils.mjs')).href);
const family = resolvePresentationFont({fontFamily: 'Arial'});
const data = JSON.parse(await fs.readFile(path.join(buildRoot, 'deck_data.json'), 'utf8'));
const backupStamp = new Date().toISOString().replace(/[:.]/g, '-');

const titles = ['Binary Search (Hodgkinson)', 'Energy Histogram', 'Gaussian Distributions'];
const commonFirst = 'Read WAV with 25/10 ms framing';
const endpointFlow = 'Hysteresis: merge <200ms gaps, discard <100ms speech';
const flows = [
  [commonFirst, 'Compute and normalize STE', 'Learn threshold by binary search',
    'Create candidate speech using the learned threshold', endpointFlow],
  [commonFirst, 'Compute Energy as mean squared amplitude',
    'Build Energy histogram for candidate speech', endpointFlow,
    'Select global W on TRAIN FINAL regions'],
  [commonFirst, 'Compute and normalize STE', 'Estimate training mean and standard deviation',
    'Create candidate speech using the Gaussian threshold', endpointFlow],
];
const comments = ['Noise extends speech tails.', 'Residual noise affects phone_F2.',
  'Gaussian fit depends on training data.'];

function checkCopy(value) {
  for (const line of String(value).split('\n')) {
    if (line.trim().split(/\s+/).length > 10) throw new Error(`More than ten words: ${line}`);
  }
  if (/[^\x00-\x7F]/.test(value)) throw new Error(`Use English ASCII slide copy: ${value}`);
}
function text(slide, value, x, y, w, h, size=30, color='#152D43', bold=false) {
  checkCopy(value);
  if (size < 24) throw new Error('Slide type must be at least 18 pt / 24 px');
  const shape = slide.shapes.add({geometry: 'textbox',
    position: {left:x, top:y, width:w, height:h}, fill:'none', line:{fill:'none', width:0}});
  shape.text = value;
  shape.text.style = {typeface:family, fontSize:size, color, bold, autoFit:'none'};
  return shape;
}
function addSlide(presentation, title, number) {
  const slide = presentation.slides.add();
  slide.background.fill = '#FFFFFF';
  text(slide, title, 60, 32, 1160, 85, 44, '#152D43', true);
  text(slide, String(number), 1160, 665, 60, 35, 24);
  return slide;
}
function setNotes(slide, value) {
  if (/[^\x00-\x7F]/.test(value)) throw new Error('Speaker notes must be composed in English');
  slide.speakerNotes.textFrame.setText(value);
}
function endpointNotes(d) {
  const n = d.parameters.endpoint_noise;
  return `Saved schema ${d.parameters.schema_version}; protocol ${d.parameters.evaluation_protocol}; ` +
    `historical TEST exposure=${d.parameters.historical_test_exposure}. Models and noise were fitted on TRAIN before TEST. ` +
    'Common endpoint refinement uses raw normalized STE. LOW continues speech, ' +
    'and a HIGH frame with the algorithm seed confirms speech. Noise calibration uses training silence only: ' +
    `mean ${n.noise_mean}, population standard deviation ${n.noise_std}, ` +
    `Q95 ${n.noise_q95}, and mean plus three standard deviations ${n.noise_upper}. ` +
    'The system merges internal gaps shorter than 200 ms and discards speech regions shorter than 100 ms. ' +
    'FINAL endpoints follow the union of active frame supports, without fixed final padding. ' +
    'Candidate regions remain diagnostic. Only ordered FINAL START/END pairs enter the primary MAE and RMSE.';
}
function algorithmNotes(index, d) {
  const q = d.parameters;
  if (index === 0) return 'Algorithm 1 learns one normalized STE threshold from the four training WAV/LAB pairs ' +
    `using the Hodgkinson binary search count rule. The saved threshold is ${d.threshold}, ` +
    `with ${q.iterations} iterations. ` +
    'Sources: algorithms/tt1_hodgkinson.py and outputs/models/tt1.json; ' +
    'Source/references/CS425 Audio and Speech Processing_Hodgkinson_2012.pdf, pages 38-39, equation 2.8 and steps 1-10. ' +
    'Test LABs enter evaluation after detection and do not tune this algorithm. ' + endpointNotes(d);
  if (index === 1) {
    const selection = q.selection;
    const tied = selection.shared_optimal_W;
    return 'Algorithm 2 uses Energy only, as required by the teacher. It does not use spectral centroid or FFT. ' +
      'Energy is the mean squared frame amplitude, E=STE/L, where L is the actual number of samples ' +
      'in that frame. STE is the sum of squared frame samples. ' +
      `The histogram uses ${q.bins} bins ` +
      `and smoothing radius ${q.smooth_radius}. The weighted threshold follows the histogram adaptation ` +
      'of Giannakopoulos, while framing follows the assignment at 25 ms / 10 ms. ' +
      `Candidate frame-F1 proposal W=${q.candidate_frame_selected_W}, F1=${q.candidate_frame_f1}. ` +
      `One global W=${q.W} is selected using TRAIN FINAL region error on ${selection.evaluated_files.join(', ')}. ` +
      'The rule rejects invalid region counts, minimizes worst per-file regret, then mean MAE. ' +
      `Shared optimal W values: ${tied.length ? tied.join(', ') : 'none'}. ` +
      `Declared tie preference W=${selection.tie_preference_W}. ` +
      `Selection set ${selection.selection_set}; protocol ${selection.evaluation_protocol}. ` +
      'TEST was viewed historically; these reused scores are not independent holdout estimates. ' +
      `Candidate padding is ${q.padding_ms} ms. Candidate padding does not set FINAL endpoints. ` +
      'Sources: core/features.py, algorithms/tt2_histogram.py, app/weight_selection.py, outputs/models/tt2.json, ' +
      'outputs/tables/tt2_w_selection/selection.json and summary.csv; ' +
      'Source/references/A method for silence removal and segmentation of speech signals_Giannakopoulos_2014.pdf, pages 1-2. ' +
      endpointNotes(d);
  }
  return 'Algorithm 3 estimates Gaussian distributions from the four training WAV/LAB pairs. ' +
    'The implementation uses population standard deviations and chooses the equal-density crossing between class means. ' +
    `Silence: mean ${q.muSil}, standard deviation ${q.stdSil}, ${q.silence_count} frames. ` +
    `Speech: mean ${q.muSp}, standard deviation ${q.stdSp}, ${q.speech_count} frames. ` +
    `The saved threshold is ${d.threshold}. Test LABs enter evaluation after detection and do not tune this algorithm. ` +
    'Sources: algorithms/tt3_gaussian.py, outputs/models/tt3.json, outputs/tables/training_frames.csv ' +
    'and the current assignment in Source/assignment. ' + endpointNotes(d);
}

for (let index=0; index<3; index++) {
  const number=index+1, key=`tt${number}`, d=data[key];
  if (process.env.ONLY_ALGORITHM && key!==process.env.ONLY_ALGORITHM) continue;
  if (d.files.length!==4) throw new Error(`${key}: expected four test files`);
  const output=path.join(root, 'slides', `THUAT_TOAN_${number}`);
  const build=path.join(buildRoot, key);
  await fs.mkdir(output, {recursive:true});
  await fs.mkdir(build, {recursive:true});
  const presentation=Presentation.create({slideSize:{width:1280,height:720}});
  let slide=addSlide(presentation, `Algorithm ${number}`, 1);
  text(slide, titles[index], 60,160,1160,100,48,'#1766BD',true);
  text(slide, 'Speech and silence segmentation',60,310,1150,60,36);
  text(slide, 'Frame 25 ms. Hop 10 ms.',60,405,1150,55,32);
  text(slide, 'Minimum silence duration: 200 ms',60,475,1150,55,32);
  text(slide, 'Name / Student ID: pending',60,565,1150,55,28);
  setNotes(slide, `Midterm presentation for Algorithm ${number}. Name / Student ID: pending. ` +
    'The planned presentation lasts about three minutes, followed by a one-minute demonstration. ' +
    'Source: the current assignment and slide submission guidance in Source/assignment.');

  slide=addSlide(presentation, `Algorithm ${number} workflow`, 2);
  let previous;
  flows[index].forEach((value, item)=>{
    const label=`${item+1}. ${value}`;
    checkCopy(label);
    const box=slide.shapes.add({geometry:'rect',position:{left:85,top:145+item*82,width:1100,height:60},
      fill:'#F0F5FA',line:{fill:'#1766BD',width:1}});
    box.text=label;
    box.text.style={typeface:family,fontSize:32,color:'#152D43',autoFit:'none'};
    if (previous) slide.shapes.connect(previous,box,{kind:'straight',fromSide:'bottom',toSide:'top',
      line:{fill:'#1766BD',width:2},tail:{type:'triangle',width:'sm',length:'sm'}});
    previous=box;
  });
  text(slide, index===1 ? `Candidate W=${d.parameters.candidate_frame_selected_W}. TRAIN FINAL W=${d.parameters.W}.` :
    `Learned threshold: ${d.threshold.toPrecision(7)}`,85,575,1100,48,28,'#1766BD');
  setNotes(slide, algorithmNotes(index,d));

  for (let item=0; item<d.files.length; item++) {
    const f=d.files[item];
    slide=addSlide(presentation, `${f.name}: test results`, item+3);
    text(slide, `MAE ${f.mae.toFixed(2)} ms   RMSE ${f.rmse.toFixed(2)} ms`,65,115,1150,48,30);
    const series=[
      {name:'Waveform',xValues:f.wave_x,values:f.wave_y,line:{fill:'#777777',width:1},marker:{symbol:'none'}},
      {name:'STE',xValues:f.centers,values:f.ste,line:{fill:'#DD8B12',width:2},marker:{symbol:'none'}},
    ];
    for (const [value,color,label] of [[f.high,'#8E44AD','HIGH'],[f.low,'#148F77','LOW']])
      series.push({name:label,xValues:[0,f.duration],values:[value,value],
        line:{fill:color,width:2},marker:{symbol:'none'}});
    for (const [boundaries,color,label] of [[f.gt,'#D62728','Ground truth'],[f.pred,'#1766BD','FINAL']])
      boundaries.forEach((boundary,k)=>series.push({name:`${label} boundary ${k+1}`,
        xValues:[boundary,boundary],values:[-1,1],line:{fill:color,width:2},marker:{symbol:'none'}}));
    // Twelve significant digits preserve chart values when opened in Excel.
    // Full-precision calculation and endpoint metrics stay in JSON and CSV.
    for (const entry of series) {
      entry.xValues=entry.xValues.map(value=>Number(value.toPrecision(12)));
      entry.values=entry.values.map(value=>Number(value.toPrecision(12)));
    }
    const chart=slide.charts.add('scatter',{position:{left:60,top:172,width:1160,height:390},series,
      scatterOptions:{style:'line'},hasLegend:false,
      xAxis:{min:0,max:f.duration,majorUnit:1,textStyle:{typeface:family,fontSize:24},numberFormatCode:'0.0'},
      yAxis:{min:-1,max:1,majorUnit:0.5,textStyle:{typeface:family,fontSize:24},numberFormatCode:'0.0'},
      chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
    applyPresentationChartFont(chart,{fontFamily:family});
    [['Waveform','#777777'],['STE','#DD8B12'],['HIGH','#8E44AD'],['LOW','#148F77'],
      ['Ground truth','#D62728'],['FINAL','#1766BD']].forEach(([value,color],k)=>
      text(slide,value,65+k*190,570,185,40,24,color));
    const explanation=index===1 ? 'TRAIN selects W. TEST was viewed historically.' :
      f.name.startsWith('phone_') ? 'Noise tails can extend speech.' : 'Framing and thresholds affect boundary error.';
    text(slide,`Time (s)\n${explanation}`,65,613,1120,65,24);
    setNotes(slide, `Sources: outputs/tables/${key}/test_metrics.csv and data/test/${f.name}.wav, ` +
      'with ground truth from the LAB file of the same name. The gray waveform is downsampled and ' +
      'normalized only for display. Feature extraction uses every original audio sample. Orange STE is ' +
      'normalized by the recording maximum. Red lines mark LAB speech-region endpoints. Blue lines mark ' +
      `the confirmed FINAL endpoints. LOW=${f.low} and HIGH=${f.high} are the actual normalized STE ` +
      `thresholds used for detection. FINAL regions=${f.region_count}. MAE=${f.mae} ms and RMSE=${f.rmse} ms. ` +
      explanation + ' ' + algorithmNotes(index,d));
  }

  slide=addSlide(presentation,'Error on four test files',7);
  const values=[['File','MAE (ms)','RMSE (ms)'],...d.files.map(f=>[f.name,f.mae.toFixed(2),f.rmse.toFixed(2)])];
  const table=slide.tables.add({rows:5,columns:3,left:65,top:155,width:1150,height:350,
    columnWidths:[550,300,300],values});
  for (let row=0;row<5;row++) for (let col=0;col<3;col++) {
    const cell=table.getCell(row,col);
    cell.text.style={typeface:family,fontSize:30,color:row===0?'#FFFFFF':'#152D43',bold:row===0};
    cell.fill=row===0?'#1766BD':row%2?'#F0F5FA':'#FFFFFF';
  }
  text(slide,`Mean MAE: ${d.mean.toFixed(2)} ms`,65,555,510,45,28,'#1766BD',true);
  text(slide,comments[index],600,555,610,45,28);
  if (index===1) text(slide,'TRAIN selects W. TEST was viewed historically.',65,615,1150,38,24);
  setNotes(slide,`Source: outputs/tables/${key}/test_metrics.csv. The summary is the arithmetic mean ` +
    'of the four per-file FINAL endpoint MAEs. Each file has one speech region and two endpoints. ' +
    'MAE averages absolute START/END errors, while RMSE takes the square root of the mean squared errors. ' +
    'Region pairs follow time order. Candidate boundaries do not enter the primary metrics. ' +
    'The assignment does not require a fixed numerical MAE cutoff. ' + algorithmNotes(index,d));

  const candidate=path.join(build,'candidate.pptx');
  await (await PresentationFile.exportPptx(presentation)).save(candidate);
  for (let item=0;item<7;item++) {
    const current=presentation.slides.getItem(item);
    const png=await presentation.export({slide:current,format:'png',scale:1});
    await fs.writeFile(path.join(build,`slide-${item+1}.png`),new Uint8Array(await png.arrayBuffer()));
    const layout=await current.export({format:'layout'});
    await fs.writeFile(path.join(build,`slide-${item+1}.layout.json`),await layout.text());
  }
  const finalPath=path.join(output,`THUAT_TOAN_${number}.pptx`);
  const receiptPath=path.join(build,'validation.json');
  for (const [existing,backupName] of [[finalPath,path.basename(finalPath)],
    [receiptPath,`${key}.validation.json`]]) {
    try {
      await fs.access(existing);
      const previous=path.join(buildRoot,'previous',backupStamp);
      await fs.mkdir(previous,{recursive:true});
      await fs.rename(existing,path.join(previous,backupName));
    } catch (error) { if (error.code!=='ENOENT') throw error; }
  }
  await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,
    pythonExecutable:runtimePython,
    integrityValidatorPath:path.join(skill,'container_tools','inspect_presentation_package_integrity.py'),
    layoutValidatorPath:path.join(skill,'container_tools','inspect_presentation_layout_geometry.py'),
    layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','7'],
    explicitTotalSlideCount:7,requiredNativeTableOwnerSlides:[7],requiredNativeChartOwnerSlides:[3,4,5,6],
    materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},
    verifyArtifactToolImport:true,receiptPath});
  console.log(`${key}: seven English slides validated`);
}
