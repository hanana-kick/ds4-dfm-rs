// Model-free HTTP double: synthetic batch/MTP counters never establish GPU qualification.
const http = require('node:http');
const crypto = require('node:crypto');
const width = Number(process.env.DS41_MAX_SEQS || 1);
const mtp = process.env.DS41_MTP_MODE === 'on';
const omitBatch = process.env.DS41_C8_MOCK_OMIT_BATCH === 'on';
const omitMtp = process.env.DS41_C8_MOCK_OMIT_MTP === 'on';
const port = Number(process.env.DS41_PORT || 18045);
const modelId = process.env.DS41_MODEL_ID || 'big';
let active = 0;
let maxSeen = 0;
if (width === 1 && mtp) console.error('DSpark drafter armed');

const server = http.createServer(async (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  if (req.method === 'GET' && req.url === '/v1/models') {
    res.end(JSON.stringify({object:'list',data:[{id:modelId}]}));
    return;
  }
  if (req.method !== 'POST' || req.url !== '/v1/chat/completions') {
    res.writeHead(404); res.end(JSON.stringify({error:{message:'not found'}})); return;
  }
  const data=[];
  for await (const chunk of req) data.push(chunk);
  let p;
  try {p=JSON.parse(Buffer.concat(data).toString('utf8'));}
  catch {res.writeHead(400);res.end(JSON.stringify({error:{message:'bad JSON'}}));return;}
  active++;
  if (active > maxSeen) {
    maxSeen = active;
    if (active === width && width > 1 && !omitBatch) {
      const verify = mtp && !omitMtp ? width * 2 : width;
      console.error(`[v41-batch] packed_decode rows=${width} steps=1 verify_rows=${verify} (synthetic mock)`);
    }
  }
  setTimeout(() => {
    const messages=p.messages||[];
    const last=messages[messages.length-1]||{};
    let finish_reason='stop', reasoning_content='', content='', tool_calls;
    if (Array.isArray(p.tools) && p.tools.length > 0) {
      finish_reason='tool_calls';
      tool_calls=[{id:`call_${Date.now()}`,type:'function',function:{name:'test_echo',arguments:'{"value":"ping"}'}}];
    } else if (messages.some(x=>x.role==='tool')) {
      content='ping';
    } else if (p.reasoning_effort==='high') {
      reasoning_content='I can see a geometric yellow face icon in the illustration.';
      content='노란색 미소 아이콘입니다.';
    } else {
      const prompt=typeof last.content==='string'?last.content:JSON.stringify(last.content);
      content='Fixed mock answer '+crypto.createHash('sha256').update(prompt).digest('hex').slice(0,16);
    }
    if (mtp && !omitMtp) console.error('[v41-mtp] request complete rounds=1 offered=5 accepted=3 (synthetic mock)');
    active--;
    res.end(JSON.stringify({id:'chatcmpl-mock',choices:[{finish_reason,message:{role:'assistant',reasoning_content,content, ...(tool_calls?{tool_calls}:{})}}],usage:{completion_tokens:50}}));
  }, 220);
});
server.listen(port,'127.0.0.1');
