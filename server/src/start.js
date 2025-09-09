import { execSync } from 'node:child_process';
execSync('npx prisma generate', { stdio: 'inherit' });
execSync('npx prisma migrate deploy', { stdio: 'inherit' });
await import('./index.js');
