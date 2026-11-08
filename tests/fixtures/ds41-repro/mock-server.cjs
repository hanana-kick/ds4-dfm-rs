// Model-free HTTP double for the seed reproduction integration test.
const http = require('node:http');

const port = Number(process.env.MOCK_PORT || process.env.DS41_PORT || 18443);
const model = process.env.DS41_MODEL_ID || 'big';

const server = http.createServer(async (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  if (req.method === 'GET' && req.url === '/v1/models') {
    res.end(JSON.stringify({object: 'list', data: [{id: model}]}));
    return;
  }
  if (req.method !== 'POST' || req.url !== '/v1/chat/completions') {
    res.writeHead(404);
    res.end(JSON.stringify({error: {message: 'not found'}}));
    return;
  }

  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  let payload;
  try {
    payload = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    res.writeHead(400);
    res.end(JSON.stringify({error: {message: 'invalid JSON'}}));
    return;
  }

  const seed = Number(payload.seed);
  const maxTokens = Number(payload.max_tokens);
  const truncated = seed === 17 && maxTokens <= 8192 && payload.temperature !== 0;
  const reasoning = seed === 17
    ? (truncated ? 'Examining the icon.' : 'Examining the icon. The shape is recognizable.')
    : 'A brief recognition of an abstract icon.';
  const answer = truncated ? '' : 'An abstract yellow icon.';
  res.end(JSON.stringify({
    id: 'chatcmpl-mock', object: 'chat.completion',
    choices: [{
      index: 0, finish_reason: truncated ? 'length' : 'stop',
      message: {role: 'assistant', reasoning_content: reasoning, content: answer},
    }],
    usage: {prompt_tokens: 12, completion_tokens: truncated ? maxTokens : 56},
  }));
});
server.listen(port, '127.0.0.1');
