#!/bin/bash

# Script per clonare il repo direttamente nella directory corrente su RunPod
# Uso: ./clone_here.sh

echo "🚀 Cloning Starless repo direttamente nella directory corrente..."

# Metodo 1: Clone in temp e sposta tutto
echo "📦 Cloning repository..."
git clone https://github.com/SyqonCollective/Starless.git temp_starless

echo "📁 Spostando file nella directory corrente..."
mv temp_starless/* ./
mv temp_starless/.* ./ 2>/dev/null || true  # Sposta anche file nascosti

echo "🧹 Pulizia directory temporanea..."
rm -rf temp_starless

echo "✅ Repository clonato con successo nella directory corrente!"

# Verifica file
echo "📋 File presenti:"
ls -la

echo ""
echo "🎯 Pronto per il training!"
echo "1. Carica i tuoi train_tiles/ e val_tiles/"
echo "2. Esegui: ./start_training.sh"