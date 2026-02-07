#!/usr/bin/env node

/**
 * HTTPS server for network access
 * Enables cache API by serving over HTTPS
 */

import https from 'https';
import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import os from 'os';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 5050;
const CERT_FILE = path.join(__dirname, 'server.crt');
const KEY_FILE = path.join(__dirname, 'server.key');

// Get local network IP
function getLocalIP() {
    const interfaces = os.networkInterfaces();
    for (const name of Object.keys(interfaces)) {
        for (const iface of interfaces[name]) {
            if (iface.family === 'IPv4' && !iface.internal) {
                return iface.address;
            }
        }
    }
    return '0.0.0.0';
}

// Generate self-signed certificate if it doesn't exist
function generateCertificate() {
    return new Promise((resolve, reject) => {
        if (fs.existsSync(CERT_FILE) && fs.existsSync(KEY_FILE)) {
            console.log('✅ Using existing certificate');
            resolve();
            return;
        }

        console.log('🔐 Generating self-signed certificate...');
        const ip = getLocalIP();
        const cmd = `openssl req -x509 -newkey rsa:2048 -keyout ${KEY_FILE} -out ${CERT_FILE} -days 365 -nodes -subj "/CN=${ip}"`;

        exec(cmd, (error) => {
            if (error) {
                reject(error);
            } else {
                console.log('✅ Certificate generated!');
                resolve();
            }
        });
    });
}

// Start HTTPS server
async function startServer() {
    try {
        await generateCertificate();

        const ip = getLocalIP();

        // Use http-server for HTTPS support (better than serve for SSL)
        // Quote paths to handle spaces in directory names
        const cmd = `npx http-server web_interface -p ${PORT} -S -C "${CERT_FILE}" -K "${KEY_FILE}" --cors`;

        console.log('\n🚀 Starting HTTPS server...\n');
        console.log(`📍 Local:   https://localhost:${PORT}`);
        console.log(`📍 Network: https://${ip}:${PORT}`);
        console.log('\n⚠️  You\'ll see a certificate warning - click "Advanced" → "Proceed"\n');
        console.log('Press Ctrl+C to stop the server\n');

        const server = exec(cmd);

        server.stdout.on('data', (data) => {
            console.log(data.toString());
        });

        server.stderr.on('data', (data) => {
            console.error(data.toString());
        });

        server.on('exit', (code) => {
            console.log(`Server exited with code ${code}`);
        });

    } catch (error) {
        console.error('❌ Failed to start server:', error.message);
        process.exit(1);
    }
}

startServer();
