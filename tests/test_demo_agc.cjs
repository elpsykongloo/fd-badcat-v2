const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('src/static/demo.js','utf8');
const start=source.indexOf('    const agc = new URLSearchParams');
const end=source.indexOf('    alive(s);',start);
assert.ok(start>=0&&end>start);
const snippet=source.slice(start,end).replace('await navigator.mediaDevices.getUserMedia','navigator.mediaDevices.getUserMedia');
let count=0;
for(const [query,configured,expected] of [['',undefined,true],['',false,false],['',true,true],['?agc=0',true,false],['?agc=1',false,true],['?agc=invalid',false,false],['?aec=1&agc=0',true,false]]){
 let captured;
 vm.runInNewContext(snippet,{URLSearchParams,location:{search:query},info:{browser_auto_gain_control:configured},s:{},selectedDevice:'chosen',navigator:{mediaDevices:{getUserMedia(x){captured=x;return {};}}}});
 assert.equal(captured.audio.autoGainControl,expected);
 assert.equal(captured.audio.echoCancellation,query.includes('aec=1'));
 assert.equal(captured.audio.deviceId.exact,'chosen');count++;
}
console.log(`AGC constraint cases: ${count} PASS`);
