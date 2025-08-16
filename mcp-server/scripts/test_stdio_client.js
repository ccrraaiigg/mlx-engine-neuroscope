import { spawn } from 'child_process';

function encode(obj) {
  const buf = Buffer.from(JSON.stringify(obj), 'utf8');
  const header = Buffer.from(`Content-Length: ${buf.length}\r\n\r\n`, 'utf8');
  return Buffer.concat([header, buf]);
}

function parseFrames(onMessage) {
  let buffer = Buffer.alloc(0);
  return (chunk) => {
    buffer = Buffer.concat([buffer, chunk]);
    while (true) {
      const headerEnd = buffer.indexOf('\r\n\r\n');
      if (headerEnd === -1) return;
      const header = buffer.slice(0, headerEnd).toString('utf8');
      const m = /Content-Length:\s*(\d+)/i.exec(header);
      if (!m) {
        buffer = buffer.slice(headerEnd + 4);
        continue;
      }
      const len = parseInt(m[1], 10);
      const total = headerEnd + 4 + len;
      if (buffer.length < total) return;
      const body = buffer.slice(headerEnd + 4, total).toString('utf8');
      buffer = buffer.slice(total);
      try { onMessage(JSON.parse(body)); } catch (e) { console.error('Bad JSON', e, body); }
    }
  };
}

async function main() {
  const proc = spawn('node', ['src/stdio_server.js'], { stdio: ['pipe', 'pipe', 'inherit'] });
  proc.on('exit', (code) => console.error('[child exit]', code));
  proc.stdout.on('data', parseFrames((msg) => {
    console.log('RECV', JSON.stringify(msg, null, 2));
  }));

  proc.stdin.write(encode({ jsonrpc: '2.0', id: 1, method: 'initialize', params: {} }));
  proc.stdin.write(encode({ jsonrpc: '2.0', id: 2, method: 'tools/list' }));

  setTimeout(() => proc.stdin.end(), 800);
}

main().catch((e) => { console.error(e); process.exit(1); });









