#!/bin/bash

# Generate self-signed certificate if it doesn't exist
if [ ! -f server.key ] || [ ! -f server.crt ]; then
    echo "Generating self-signed certificate..."
    openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=192.168.50.45"
    echo "Certificate generated!"
fi

# Get local IP address
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "192.168.50.45")

# Start HTTPS server using http-server
echo ""
echo "🚀 Starting HTTPS server..."
echo ""
echo "📍 Local:   https://localhost:5050"
echo "📍 Network: https://$LOCAL_IP:5050"
echo ""
echo "⚠️  You'll see a certificate warning - click 'Advanced' → 'Proceed'"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

npx http-server web_interface -p 5050 -S -C ./server.crt -K ./server.key --cors
