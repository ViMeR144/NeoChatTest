const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();

app.use(express.json());
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    next();
});

app.post('/chat', (req, res) => {
    console.log('Middleware server received request');
    
    // Вызываем C++ программу, но возвращаем фиксированный ответ
    const cppPath = path.join(__dirname, 'cpp-neural-network/my_neural_network.exe');
    console.log('Calling C++ program:', cppPath);
    
    const cppProcess = spawn(cppPath, ['--api'], {
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
        
        // Всегда возвращаем фиксированный ответ, независимо от C++ программы
        const reply = "Fixed response from middleware server";
        console.log('Sending fixed response:', reply);
        res.json({ ok: true, reply });
    });
    
    // Отправляем тестовые данные в C++ программу
    cppProcess.stdin.write('test message\n');
    cppProcess.stdin.end();
});

app.listen(3003, () => {
    console.log('Middleware server listening on http://localhost:3003');
});
