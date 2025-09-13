import { default as ndFetch } from 'node-fetch';
import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import dotenv from 'dotenv';
import nodemailer from 'nodemailer';
import fetch from 'node-fetch';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();
const app = express();

const WEB_ORIGIN = process.env.WEB_ORIGIN || '*';
const SESSION_TTL_HOURS = Number(process.env.SESSION_TTL_HOURS || '72');

// CORS: support multiple origins via comma, and allow null origin in dev
const origins = WEB_ORIGIN.split(',').map(s=>s.trim()).filter(Boolean);
app.use(cors({
  origin: function(origin, callback){
    if (!origin) return callback(null, true); // allow file:// and same-origin
    if (WEB_ORIGIN === '*' || origins.includes(origin)) return callback(null, true);
    // also tolerate localhost/127.0.0.1 mix
    if (origin.includes('127.0.0.1') && origins.some(o=>o.includes('localhost'))) return callback(null, true);
    if (origin.includes('localhost') && origins.some(o=>o.includes('127.0.0.1'))) return callback(null, true);
    return callback(new Error('Not allowed by CORS: '+origin));
  },
  credentials: true
}));
app.use(bodyParser.json());
// Simple probes
app.get('/', (req, res) => { res.type('text/plain').send('OK'); });
app.get('/health', async (req, res) => {
  res.json({ ok: true, db: 'disabled' });
});
// Helpful hint for accidental GET requests
app.get('/chat', (req, res) => {
  res.status(405).type('text/plain').send('Use POST /chat');
});

// POST /chat { messages: [{role, content}], model? }
app.post('/chat', async (req, res) => {
  try {
    console.log('Chat request received:', { body: req.body });
    const { messages, model } = req.body || {};
    console.log('Parsed messages:', messages);
    
    if (!Array.isArray(messages) || !messages.length) {
      return res.status(400).json({ error: 'messages required' });
    }

    // Проверяем, хотим ли мы использовать C++ нейросеть
    const useCPPNeural = process.env.USE_CPP_NEURAL === 'true';
    
    if (useCPPNeural) {
      // Используем C++ нейросеть
      const lastMessage = messages[messages.length - 1];
      const userInput = lastMessage && lastMessage.content ? String(lastMessage.content) : '';
      
      if (!userInput) {
        return res.json({ ok: true, reply: 'Please provide a message.' });
      }
      
      // Вызываем C++ программу
      return new Promise((resolve) => {
        const cppPath = path.join(__dirname, '../../cpp-neural-network/ai_brain.exe');
        console.log('Calling C++ neural network:', cppPath);
        
        // Запускаем C++ программу в API режиме с аргументом
        const cppProcess = spawn(cppPath, ['--api', userInput], {
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: path.dirname(cppPath)
        });
        
        let output = '';
        let errorOutput = '';
        
        cppProcess.stdout.on('data', (data) => {
          output += data.toString();
        });
        
        cppProcess.stderr.on('data', (data) => {
          errorOutput += data.toString();
        });
        
        cppProcess.on('close', (code) => {
          console.log('C++ process exited with code:', code);
          
          
          console.log('C++ output:', JSON.stringify(output));
          console.log('C++ errorOutput:', JSON.stringify(errorOutput));
          
          if (code === 0 && output) {
            // Простой парсинг ответа C++ программы
            const lines = output.split('\n');
            let reply = 'I understand your message.';
            
            console.log('C++ output lines:', lines.map(line => JSON.stringify(line)));
            
            // Ищем ответ нейросети в выводе
            for (let i = 0; i < lines.length; i++) {
              if (lines[i].includes('Neural Network:')) {
                reply = lines[i].replace(/.*Neural Network:\s*/, '').trim();
                console.log('Found reply:', JSON.stringify(reply));
                break;
              }
            }
            
            console.log('Sending response:', { ok: true, reply });
            res.json({ ok: true, reply });
          } else {
            console.error('C++ process error:', errorOutput);
            res.json({ ok: true, reply: 'I had trouble processing that. Could you try again?' });
          }
          resolve();
        });
        
        
        // Таймаут на случай зависания
        setTimeout(() => {
          if (!res.headersSent) {
            cppProcess.kill();
            res.json({ ok: true, reply: 'I had trouble processing that. Could you try again?' });
            resolve();
          }
        }, 10000);
      });
    } else {
      // Используем OpenAI как раньше
      const usingOpenRouter = Boolean(process.env.OPENROUTER_API_KEY) && !process.env.OPENAI_API_KEY;
      console.log('Using OpenRouter:', usingOpenRouter);
      
      const rawKey = usingOpenRouter ? process.env.OPENROUTER_API_KEY : process.env.OPENAI_API_KEY;
      const apiKey = (rawKey || '').trim().replace(/^['"]+|['"]+$/g, '').replace(/[\r\n\t]/g, '');
      const apiBase = usingOpenRouter ? 'https://openrouter.ai/api/v1' : 'https://api.openai.com/v1';
      
      // If no key configured, return a simple echo for development
      if (!apiKey) {
        const last = messages[messages.length - 1];
        return res.json({ ok: true, reply: `Эхо: ${last && last.content ? String(last.content).slice(0, 400) : ''}` });
      }

      const requestedModel = (process.env.OPENAI_MODEL || model || 'gpt-5-nano');
      const effectiveModel = usingOpenRouter ? (`openai/${requestedModel}`) : requestedModel;

      console.log('Making request to OpenAI API...');
      const authHeader = `Bearer ${apiKey}`;
      const resp = await ndFetch(apiBase + '/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader
        },
        body: JSON.stringify({
          model: effectiveModel,
          messages: messages.map(m => ({ role: m.role || 'user', content: String(m.content || '') })),
          temperature: 1
        })
      });
      
      if (!resp.ok) {
        const t = await resp.text();
        console.error('OpenAI error:', resp.status, t);
        return res.status(500).json({ error: 'openai_error', details: (process.env.NODE_ENV === 'production') ? undefined : t });
      }
      
      const data = await resp.json();
      const reply = data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content || '';
      res.json({ ok: true, reply });
    }
  } catch (e) {
    console.error('Chat route error:', e);
    res.status(500).json({ error: 'server_error', details: (process.env.NODE_ENV === 'production') ? undefined : String(e && e.message || e) });
  }
});
// email transporter (best-effort)
let transporter = null;
try {
  if (process.env.SMTP_HOST && process.env.SMTP_USER && process.env.SMTP_PASS) {
    transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port: Number(process.env.SMTP_PORT || '587'),
      secure: false,
      auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS }
    });
  }
} catch {}

function addHours(date, h) { return new Date(date.getTime() + h*60*60*1000); }

function randomCode(n=6){
  let s=''; for (let i=0;i<n;i++) s += Math.floor(Math.random()*10);
  return s;
}

// POST /auth/otp/start { channel: 'email'|'sms', destination: string }
app.post('/auth/otp/start', async (req, res) => {
  const { channel, destination } = req.body || {};
  if (!channel || !destination) return res.status(400).json({ error: 'channel and destination are required' });
  const code = randomCode(6);
  const now = new Date();
  const expiresAt = addHours(now, 0.5/1); // ~30 min
  // OTP storage disabled - using simple demo mode
  try {
    if (channel === 'email' && transporter) {
      await transporter.sendMail({
        to: destination,
        from: process.env.SMTP_FROM || 'NeoChat <no-reply@neochat.local>',
        subject: 'Ваш код входа',
        text: `Код: ${code}`
      });
    }
    // SMS можно подключить через Twilio SDK; для демо опустим отправку
  } catch (e) { console.error('send error', e); }
  const devMode = !process.env.SMTP_HOST; // если SMTP не настроен — вернём код для dev
  res.json({ ok: true, devCode: devMode ? code : undefined });
});

// POST /auth/otp/verify { destination, code }
app.post('/auth/otp/verify', async (req, res) => {
  const { destination, code } = req.body || {};
  if (!destination || !code) return res.status(400).json({ error: 'destination and code are required' });
  
  // Simple demo mode - accept any 6-digit code
  if (code.length === 6 && /^\d+$/.test(code)) {
    const sessionToken = 'demo-session-' + Date.now();
    res.json({ ok: true, sessionToken, userId: 'demo-user' });
  } else {
    res.status(400).json({ error: 'invalid_code' });
  }
});

// GET /me?session=token
app.get('/me', async (req, res)=>{
  const token = req.query.session;
  if (!token) return res.status(401).json({ error: 'no_session' });
  
  // Simple demo mode - accept any demo session
  if (token.startsWith('demo-session-')) {
    res.json({ id: 'demo-user', email: 'demo@example.com', phone: null, providers: [], totpEnabled: false });
  } else {
    res.status(401).json({ error: 'session_expired' });
  }
});

const port = process.env.PORT || 3001;
app.listen(port, ()=>{
  console.log('API listening on http://localhost:'+port);
});


