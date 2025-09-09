import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import dotenv from 'dotenv';
import { PrismaClient } from '@prisma/client';
import nodemailer from 'nodemailer';

dotenv.config();
const app = express();
const prisma = new PrismaClient();

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
  try {
    await prisma.$queryRaw`SELECT 1`;
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e && e.message || e) });
  }
});

// POST /chat { messages: [{role, content}], model? }
app.post('/chat', async (req, res) => {
  try {
    const { messages, model } = req.body || {};
    const usingOpenRouter = Boolean(process.env.OPENROUTER_API_KEY);
    const apiKey = usingOpenRouter ? process.env.OPENROUTER_API_KEY : process.env.OPENAI_API_KEY;
    const apiBase = usingOpenRouter ? 'https://openrouter.ai/api/v1' : 'https://api.openai.com/v1';
    if (!Array.isArray(messages) || !messages.length) {
      return res.status(400).json({ error: 'messages required' });
    }
    // If no key configured, return a simple echo for development
    if (!apiKey) {
      const last = messages[messages.length - 1];
      return res.json({ ok: true, reply: `Эхо: ${last && last.content ? String(last.content).slice(0, 400) : ''}` });
    }

    const requestedModel = (process.env.OPENAI_MODEL || model || 'gpt-4o-mini');
    const effectiveModel = usingOpenRouter ? (`openai/${requestedModel}`) : requestedModel;

    const resp = await fetch(apiBase + '/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        ...(usingOpenRouter ? { 'HTTP-Referer': process.env.WEB_ORIGIN || 'http://localhost', 'X-Title': 'NeoChat' } : {})
      },
      body: JSON.stringify({
        model: effectiveModel,
        messages: messages.map(m => ({ role: m.role || 'user', content: String(m.content || '') })),
        temperature: 0.7
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
  } catch (e) {
    console.error('Chat route error:', e);
    res.status(500).json({ error: 'server_error', details: (process.env.NODE_ENV === 'production') ? undefined : String(e && e.message || e) });
  }
});

// email transporter (best-effort)
let transporter = null;
try {
  transporter = nodemailer.createTransport({
    host: process.env.SMTP_HOST,
    port: Number(process.env.SMTP_PORT || '587'),
    secure: false,
    auth: process.env.SMTP_USER ? { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS } : undefined
  });
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
  await prisma.otpCode.create({ data: { channel, destination, code, expiresAt } });
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
  const entry = await prisma.otpCode.findFirst({ where: { destination, code, consumed: false } });
  if (!entry) return res.status(400).json({ error: 'invalid_code' });
  if (entry.expiresAt < new Date()) return res.status(400).json({ error: 'expired' });
  await prisma.otpCode.update({ where: { id: entry.id }, data: { consumed: true } });

  // find or create user
  const isEmail = destination.includes('@');
  let user = await prisma.user.findFirst({ where: isEmail ? { email: destination } : { phone: destination } });
  if (!user) {
    user = await prisma.user.create({ data: isEmail ? { email: destination } : { phone: destination } });
  }
  const session = await prisma.session.create({ data: { userId: user.id, expiresAt: addHours(new Date(), SESSION_TTL_HOURS) } });
  res.json({ ok: true, sessionToken: session.id, userId: user.id });
});

// GET /me?session=token
app.get('/me', async (req, res)=>{
  const token = req.query.session;
  if (!token) return res.status(401).json({ error: 'no_session' });
  const s = await prisma.session.findUnique({ where: { id: token } });
  if (!s || s.expiresAt < new Date()) return res.status(401).json({ error: 'session_expired' });
  const user = await prisma.user.findUnique({ where: { id: s.userId }, include: { providers: true } });
  res.json({ id: user.id, email: user.email, phone: user.phone, providers: user.providers.map(p=>p.type), totpEnabled: Boolean(user.totpSecret) });
});

const port = process.env.PORT || 3001;
app.listen(port, ()=>{
  console.log('API listening on http://localhost:'+port);
});


