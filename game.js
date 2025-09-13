class SurfingGame {
    constructor() {
        if (this.isMobileDevice() && localStorage.getItem('userWantsPlayFromMobile') !== 'true') {
            this.showMobileWarning();
            return;
        }

        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.inventory = document.getElementById('inventory');
        this.scoreElement = document.getElementById('score');
        this.startScreen = document.getElementById('startScreen');
        this.gameOverScreen = document.getElementById('gameOverScreen');
        
        this.sprites = {};
        this.waveAnimation = {
            frame: 0,
            frameCount: 16,
            frameDelay: 80,
            lastFrameTime: 0,
            waveOffset: 0
        };
        
        this.oceanBackground = {
            waveOffset: 0,
            foamParticles: [],
            depthLayers: [],
            waveSpeed: 0.5,
            foamCount: 50
        };
        
        this.setupCanvas();
        this.loadSprites();
        this.setupControls();
        this.setupGameState();
        this.setupEventListeners();
        this.initOceanBackground();
    }

    isMobileDevice() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
               window.innerWidth < 1024;
    }

    showMobileWarning() {
        document.body.innerHTML = `
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.9);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                z-index: 200;
            ">
                <div style="
                    text-align: center;
                    padding: 15px;
                    border: 3px solid #00ff00;
                    background: #001122;
                    border-radius: 10px;
                    max-width: 80%;
                ">
                    <div style="
                        font-size: 20px;
                        margin-bottom: 15px;
                        color: #00ff00;
                    ">🌊 AlfaCTF Mermaid 🌊</div>
                    <div style="
                        font-size: 12px;
                        margin-bottom: 15px;
                        color: #00aa00;
                    ">Эта игра предназначена для игры на компьютере с клавиатурой</div>
                    <div style="
                        font-size: 10px;
                        margin-bottom: 15px;
                        color: #008800;
                        line-height: 1.5;
                    ">
                        Держись подальше от русалки и собери флаг!<br>
                        🎮 Геймплей:<br>
                        • Перемещайся по линиям и собирай жёлтые буквы флага<br>
                        • Можно импользовапть стрелки и W/S<br>
                        • Не дай русалке тебя притянуть фиолетовыми песнями!<br>
                        • Обязательно собери флаг целиком!<br>
                    </div>
                    <div style="
                        font-size: 12px;
                        margin-bottom: 14px;
                        color: #ffaa00;
                        font-weight: bold;
                    ">
                        Пожалуйста, откройте игру на ПК или ноутбуке!
                    </div>
                    <button id="playAnywayBtn" style="
                        padding: 10px 20px;
                        font-size: 14px;
                        background: #001122;
                        color: #00ff00;
                        border: 2px solid #00ff00;
                        border-radius: 5px;
                        cursor: pointer;
                        font-family: 'Courier New', monospace;
                        transition: all 0.3s;
                        margin-top: 10px;
                    " onmouseover="this.style.background='#00ff00'; this.style.color='#000';" onmouseout="this.style.background='#001122'; this.style.color='#00ff00';">
                        Все равно играть
                    </button>
                </div>
            </div>
        `;
        
        document.getElementById('playAnywayBtn').addEventListener('click', () => {
            localStorage.setItem('userWantsPlayFromMobile', 'true');
            location.reload();
        });
    }

    initOceanBackground() {
        for (let i = 0; i < this.oceanBackground.foamCount; i++) {
            this.oceanBackground.foamParticles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 3 + 1,
                opacity: Math.random() * 0.5 + 0.1,
                life: Math.random() * 100 + 50
            });
        }
        
        this.oceanBackground.depthLayers = [
            { speed: 0.1, color: '#001a33', height: this.canvas.height * 0.8 },
            { speed: 0.2, color: '#002244', height: this.canvas.height * 0.6 },
            { speed: 0.3, color: '#003366', height: this.canvas.height * 0.4 }
        ];
    }

    generateWaveFrames() {
        this.waveFrames = [];
        const waveSize = 60;
        
        for (let frame = 0; frame < this.waveAnimation.frameCount; frame++) {
            const canvas = document.createElement('canvas');
            canvas.width = waveSize;
            canvas.height = waveSize;
            const ctx = canvas.getContext('2d');
            
            this.drawWaveFrame(ctx, waveSize, frame);
            
            this.waveFrames.push(canvas);
        }
    }

    drawWaveFrame(ctx, size, frame) {
        const phase = (frame / this.waveAnimation.frameCount) * Math.PI * 2;
        
        const gradient = ctx.createLinearGradient(0, 0, 0, size);
        gradient.addColorStop(0, '#0066aa');
        gradient.addColorStop(0.3, '#0088cc');
        gradient.addColorStop(0.7, '#00aadd');
        gradient.addColorStop(1, '#00ccff');
        
        ctx.fillStyle = gradient;
        
        ctx.beginPath();
        ctx.moveTo(0, size);
        
        for (let x = 0; x <= size; x += 1.5) {
            const wave1 = Math.sin(x * 0.08 + phase) * 10;
            const wave2 = Math.sin(x * 0.04 + phase * 0.7) * 5;
            const wave3 = Math.sin(x * 0.12 + phase * 1.3) * 7;
            const wave4 = Math.sin(x * 0.06 + phase * 0.3) * 3;
            const y = size * 0.45 + wave1 + wave2 + wave3 + wave4;
            
            ctx.lineTo(x, y);
        }
        
        ctx.lineTo(size, size);
        ctx.closePath();
        ctx.fill();
        
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        for (let i = 0; i < 8; i++) {
            const x = Math.random() * size;
            const waveY = size * 0.45 + 
                         Math.sin(x * 0.08 + phase) * 10 +
                         Math.sin(x * 0.04 + phase * 0.7) * 5 +
                         Math.sin(x * 0.12 + phase * 1.3) * 7;
            const y = waveY - Math.random() * 8;
            const foamSize = Math.random() * 3 + 1;
            ctx.fillRect(x, y, foamSize, foamSize);
        }
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let x = 0; x <= size; x += 2) {
            const wave1 = Math.sin(x * 0.08 + phase) * 10;
            const wave2 = Math.sin(x * 0.04 + phase * 0.7) * 5;
            const wave3 = Math.sin(x * 0.12 + phase * 1.3) * 7;
            const wave4 = Math.sin(x * 0.06 + phase * 0.3) * 3;
            const y = size * 0.45 + wave1 + wave2 + wave3 + wave4 - 3;
            
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x <= size; x += 4) {
            const wave1 = Math.sin(x * 0.08 + phase) * 10;
            const wave2 = Math.sin(x * 0.04 + phase * 0.7) * 5;
            const wave3 = Math.sin(x * 0.12 + phase * 1.3) * 7;
            const wave4 = Math.sin(x * 0.06 + phase * 0.3) * 3;
            const y = size * 0.45 + wave1 + wave2 + wave3 + wave4 - 1;
            
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }

    updateOceanBackground() {
        this.oceanBackground.waveOffset += this.oceanBackground.waveSpeed;
        
        for (let particle of this.oceanBackground.foamParticles) {
            particle.x += particle.vx;
            particle.y += particle.vy;
            particle.life -= 0.5;
            
            if (particle.x < 0) particle.x = this.canvas.width;
            if (particle.x > this.canvas.width) particle.x = 0;
            if (particle.y < 0) particle.y = this.canvas.height;
            if (particle.y > this.canvas.height) particle.y = 0;
            
            if (particle.life <= 0) {
                particle.x = Math.random() * this.canvas.width;
                particle.y = Math.random() * this.canvas.height;
                particle.life = Math.random() * 100 + 50;
                particle.opacity = Math.random() * 0.5 + 0.1;
            }
        }
    }

    renderOceanBackground() {
        const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
        gradient.addColorStop(0, '#001122');
        gradient.addColorStop(0.2, '#002244');
        gradient.addColorStop(0.4, '#003366');
        gradient.addColorStop(0.6, '#004488');
        gradient.addColorStop(0.8, '#0055aa');
        gradient.addColorStop(1, '#0066cc');
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.renderLightRays();
        
        for (let layer of this.oceanBackground.depthLayers) {
            this.renderDepthLayer(layer);
        }
        
        this.renderPixelWaves();
        
        this.renderFoamParticles();
        
        this.renderSurfaceReflection();
    }

    renderDepthLayer(layer) {
        const waveHeight = 20;
        const waveSpacing = 40;
        const offset = this.oceanBackground.waveOffset * layer.speed;
        
        this.ctx.fillStyle = layer.color;
        this.ctx.globalAlpha = 0.3;
        
        for (let x = -offset; x < this.canvas.width + waveSpacing; x += waveSpacing) {
            const waveY = this.canvas.height - layer.height + Math.sin(x * 0.02 + this.oceanBackground.waveOffset * 0.01) * 10;
            
            for (let px = 0; px < waveSpacing; px += 4) {
                const pixelX = x + px;
                const pixelY = waveY + Math.sin(px * 0.1 + this.oceanBackground.waveOffset * 0.02) * 5;
                
                if (pixelX >= 0 && pixelX < this.canvas.width) {
                    this.ctx.fillRect(pixelX, pixelY, 2, 2);
                }
            }
        }
        
        this.ctx.globalAlpha = 1;
    }

    renderPixelWaves() {
        const waveSpacing = 30;
        const offset = this.oceanBackground.waveOffset * 0.8;
        
        const waveLayers = [
            { color: '#0066aa', alpha: 0.4, speed: 0.015, amplitude: 15, yOffset: 0.7 },
            { color: '#0088cc', alpha: 0.3, speed: 0.02, amplitude: 10, yOffset: 0.6 },
            { color: '#00aadd', alpha: 0.2, speed: 0.025, amplitude: 8, yOffset: 0.5 }
        ];
        
        for (let layer of waveLayers) {
            this.ctx.fillStyle = layer.color;
            this.ctx.globalAlpha = layer.alpha;
            
            for (let x = -offset; x < this.canvas.width + waveSpacing; x += waveSpacing) {
                const waveY = this.canvas.height * layer.yOffset + 
                             Math.sin(x * 0.03 + this.oceanBackground.waveOffset * layer.speed) * layer.amplitude;
                
                for (let px = 0; px < waveSpacing; px += 3) {
                    const pixelX = x + px;
                    const pixelY = waveY + Math.sin(px * 0.15 + this.oceanBackground.waveOffset * 0.025) * 8;
                    
                    if (pixelX >= 0 && pixelX < this.canvas.width) {
                        const pixelSize = Math.random() > 0.7 ? 4 : 3;
                        this.ctx.fillRect(pixelX, pixelY, pixelSize, pixelSize);
                    }
                }
            }
        }
        
        this.ctx.globalAlpha = 1;
    }

    renderFoamParticles() {
        for (let particle of this.oceanBackground.foamParticles) {
            this.ctx.globalAlpha = particle.opacity;
            
            if (Math.random() > 0.7) {
                this.ctx.beginPath();
                this.ctx.arc(particle.x + particle.size/2, particle.y + particle.size/2, particle.size/2, 0, Math.PI * 2);
                this.ctx.fillStyle = '#ffffff';
                this.ctx.fill();
            } else {
                this.ctx.fillStyle = '#ffffff';
                this.ctx.fillRect(particle.x, particle.y, particle.size, particle.size);
            }
        }
        
        this.ctx.globalAlpha = 1;
    }

    renderLightRays() {
        this.ctx.globalAlpha = 0.1;
        this.ctx.fillStyle = '#ffffff';
        
        const rayCount = 5;
        const rayWidth = this.canvas.width / rayCount;
        
        for (let i = 0; i < rayCount; i++) {
            const x = i * rayWidth + rayWidth / 2;
            const gradient = this.ctx.createLinearGradient(x, 0, x, this.canvas.height);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
            
            this.ctx.fillStyle = gradient;
            this.ctx.fillRect(x - 20, 0, 40, this.canvas.height);
        }
        
        this.ctx.globalAlpha = 1;
    }

    renderSurfaceReflection() {
        const reflectionHeight = this.canvas.height * 0.1;
        const gradient = this.ctx.createLinearGradient(0, 0, 0, reflectionHeight);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 0.1)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, reflectionHeight);
    }

    setupCanvas() {
        this.canvas.width = window.innerWidth / 2;
        this.canvas.height = window.innerHeight / 2;
        
        window.addEventListener('resize', () => {
            this.canvas.width = window.innerWidth / 2;
            this.canvas.height = window.innerHeight / 2;
            this.updateLaneWidth();
            this.initOceanBackground(); 
            this.generateWaveFrames(); 
        });
    }

    loadSprites() {
        const spriteNames = ['surfer', 'swim-mermaid', 'song-mermaid'];
        
        spriteNames.forEach(name => {
            const img = new Image();
            img.onload = () => {
                this.sprites[name] = img;
            };
            img.onerror = () => {
                console.error(`Failed to load sprite: ${name}`);
            };
            img.src = `sprites/${name}.png`;
        });
        
        this.generateWaveFrames();
    }

    setupControls() {
        this.keys = {};
        this.touchStartX = 0;
        this.touchStartY = 0;
        
        document.addEventListener('keydown', (e) => {
            this.keys[e.code] = true;
            
            if (e.code === 'ArrowUp' || e.code === 'KeyW') {
                this.movePlayer(-1);
            } else if (e.code === 'ArrowDown' || e.code === 'KeyS') {
                this.movePlayer(1);
            }
        });
        
        document.addEventListener('keyup', (e) => {
            this.keys[e.code] = false;
        });
        
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.touchStartX = e.touches[0].clientX;
            this.touchStartY = e.touches[0].clientY;
        });
        
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
        });
        
        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const touchEndX = e.changedTouches[0].clientX;
            const touchEndY = e.changedTouches[0].clientY;
            
            const deltaX = touchEndX - this.touchStartX;
            const deltaY = touchEndY - this.touchStartY;
            
            if (this.gameState.isMobile) {
                if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 30) {
                    if (deltaX > 0) {
                        this.movePlayer(1); 
                    } else {
                        this.movePlayer(-1);
                    }
                }
            } else {
                if (Math.abs(deltaY) > Math.abs(deltaX) && Math.abs(deltaY) > 30) {
                    if (deltaY > 0) {
                        this.movePlayer(1); 
                    } else {
                        this.movePlayer(-1);
                    }
                }
            }
        });
        
        document.getElementById('leftBtn').addEventListener('click', () => {
            this.movePlayer(-1);
        });
        
        document.getElementById('rightBtn').addEventListener('click', () => {
            this.movePlayer(1);
        });
    }

    setupGameState() {
        this.gameState = {
            isPlaying: false,
            score: 0,
            speed: 2,
            baseSpeed: 2,
            maxSpeed: 12,
            speedIncrease: 0.2,
            sessionId: null,
            collectedLetters: [],
            currentLetter: null,
            totalLetters: 0,
            gameComplete: false,
            startTime: null,
            gameTime: 0,
            isMobile: window.innerWidth < 768
        };
        
        this.player = {
            x: this.gameState.isMobile ? this.canvas.width / 2 : this.canvas.width - 75,
            y: this.gameState.isMobile ? this.canvas.height - 75 : this.canvas.height / 2,
            width: this.gameState.isMobile ? 20 : 30, 
            height: this.gameState.isMobile ? 30 : 45, 
            lane: 1, 
            targetLane: 1,
            laneHeight: this.gameState.isMobile ? this.canvas.width / 3 : this.canvas.height / 3
        };
        
        this.mermaid = {
            x: this.gameState.isMobile ? this.canvas.width / 2 : 50,
            y: this.gameState.isMobile ? 50 : this.canvas.height / 2,
            width: this.gameState.isMobile ? 36 : 54, 
            height: this.gameState.isMobile ? 40 : 60, 
            speed: 4,
            currentLane: 1,
            moveTimer: 0,
            moveInterval: 800, 
            letterSpawnTimer: 0,
            letterSpawnInterval: 2000,
            noteSpawnTimer: 0,
            noteSpawnInterval: 4000,
            isSinging: false,
            singingTimer: 0,
            singingDuration: 3000,
            lastSpawnLane: -1,
            spawnCountOnCurrentLane: 0
        };
        
        this.letters = [];
        this.notes = [];
        this.particles = [];
        
        this.updateLaneWidth();
    }

    updateLaneWidth() {
        this.player.laneHeight = this.gameState.isMobile ? this.canvas.width / 3 : this.canvas.height / 3;
    }

    setupEventListeners() {
        document.getElementById('startBtn').addEventListener('click', () => {
            this.startGame();
        });
        
        document.getElementById('restartBtn').addEventListener('click', () => {
            this.restartGame();
        });
        
        const toggleBtn = document.getElementById('toggleInventory');
        const inventoryContainer = document.getElementById('inventory-container');
        
        if (toggleBtn && inventoryContainer) {
            toggleBtn.addEventListener('click', () => {
                inventoryContainer.classList.toggle('collapsed');
                toggleBtn.textContent = inventoryContainer.classList.contains('collapsed') ? '▲' : '▼';
            });
        }
    }

    async startGame() {
        try {
            const response = await fetch('/api/game/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.sessionId) {
                this.gameState.sessionId = data.sessionId;
                this.gameState.totalLetters = data.totalLetters;
                this.gameState.isPlaying = true;
                this.gameState.startTime = Date.now();
                this.startScreen.style.display = 'none';
                this.gameOverScreen.style.display = 'none';
                
                this.updateInventory();
                this.gameLoop();
                this.spawnNextLetter();
            }
        } catch (error) {
            console.error('Failed to start game:', error);
            alert('Failed to start game. Please try again.');
        }
    }

    async spawnNextLetter() {
        if (!this.gameState.isPlaying || this.gameState.gameComplete) return;
        
        try {
            const response = await fetch(`/api/game/next-letter/${this.gameState.sessionId}`);
            const data = await response.json();
            
            if (data.gameComplete) {
                this.gameState.gameComplete = true;
                this.endGame(true);
                return;
            }
            
            if (data.nextLetter) {
                this.gameState.currentLetter = data.nextLetter;
                
                if (this.mermaid.lastSpawnLane === this.mermaid.currentLane && this.mermaid.spawnCountOnCurrentLane >= 2) {
                    const availableLanes = [0, 1, 2].filter(lane => lane !== this.mermaid.currentLane);
                    this.mermaid.currentLane = availableLanes[Math.floor(Math.random() * availableLanes.length)];
                    this.mermaid.spawnCountOnCurrentLane = 0;
                }
                
                if (this.mermaid.lastSpawnLane === this.mermaid.currentLane) {
                    this.mermaid.spawnCountOnCurrentLane++;
                } else {
                    this.mermaid.lastSpawnLane = this.mermaid.currentLane;
                    this.mermaid.spawnCountOnCurrentLane = 1;
                }
                
                
                this.mermaid.isSinging = true;
                this.mermaid.singingTimer = 0;
                
                
                if (this.gameState.isMobile) {
                    const x = this.mermaid.currentLane * this.player.laneHeight + this.player.laneHeight / 2;
                    
                    this.letters.push({
                        x: x,
                        y: this.mermaid.y + 15,
                        width: 25,
                        height: 25,
                        letter: data.nextLetter,
                        lane: this.mermaid.currentLane,
                        speed: this.gameState.speed + 1
                    });
                } else {
                    const y = this.mermaid.currentLane * this.player.laneHeight + this.player.laneHeight / 2;
                    
                    this.letters.push({
                        x: this.mermaid.x + 15,
                        y: y,
                        width: 25,
                        height: 25,
                        letter: data.nextLetter,
                        lane: this.mermaid.currentLane,
                        speed: this.gameState.speed + 1
                    });
                }
            }
        } catch (error) {
            console.error('Failed to spawn letter:', error);
        }
    }

    async spawnNote() {
        if (!this.gameState.isPlaying || this.gameState.gameComplete) return;
        
        try {
            const response = await fetch(`/api/game/note/${this.gameState.sessionId}`);
            const data = await response.json();
            
            if (data.success && data.note) {
                
                const randomLane = Math.floor(Math.random() * 3);
                
                if (this.gameState.isMobile) {
                    const x = randomLane * this.player.laneHeight + this.player.laneHeight / 2;
                    
                    this.notes.push({
                        x: x,
                        y: this.mermaid.y + 15,
                        width: 25,
                        height: 25,
                        note: data.note,
                        lane: randomLane,
                        speed: this.gameState.speed + 1
                    });
                } else {
                    const y = randomLane * this.player.laneHeight + this.player.laneHeight / 2;
                    
                    this.notes.push({
                        x: this.mermaid.x + 15,
                        y: y,
                        width: 25,
                        height: 25,
                        note: data.note,
                        lane: randomLane,
                        speed: this.gameState.speed + 1
                    });
                }
            }
        } catch (error) {
            console.error('Failed to spawn note:', error);
        }
    }

    async collectLetter(letter) {
        if (!this.gameState.sessionId) return;
        
        try {
            const response = await fetch(`/api/game/collect-letter/${this.gameState.sessionId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ letter: letter })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.gameState.collectedLetters = data.collectedLetters;
                this.gameState.score += 100;
                this.createParticles(letter);
                this.updateInventory();
                
                if (data.gameComplete) {
                    this.gameState.gameComplete = true;
                    this.endGame(true);
                } else {
                    
                    setTimeout(() => {
                        this.spawnNextLetter();
                    }, 1000);
                }
            }
        } catch (error) {
            console.error('Failed to collect letter:', error);
        }
    }

    collectNote(note) {
        this.gameState.score += 50;
        this.createParticles(note, '#ff00ff');

        this.moveSurferTowardsMermaid();
    }

    moveSurferTowardsMermaid() {
        const moveDistance = 25;

        if (this.gameState.isMobile) {
            const targetY = Math.max(this.mermaid.y + this.mermaid.height, this.player.y - moveDistance);
            this.player.y = targetY;
        } else {
            const targetX = Math.max(this.mermaid.x + this.mermaid.width, this.player.x - moveDistance);
            this.player.x = targetX;
        }

        if (this.gameState.isMobile) {
            this.player.targetLane = Math.floor(this.player.x / this.player.laneHeight);
            this.player.targetLane = Math.max(0, Math.min(2, this.player.targetLane));
        } else {
            this.player.targetLane = Math.floor(this.player.y / this.player.laneHeight);
            this.player.targetLane = Math.max(0, Math.min(2, this.player.targetLane));
        }
    }

    async reportMissedLetter() {
        if (!this.gameState.sessionId) return;
        
        try {
            const response = await fetch(`/api/game/missed-letter/${this.gameState.sessionId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                
                if (data.nextLetter) {
                    this.gameState.currentLetter = data.nextLetter;
                    
                    
                    const y = this.mermaid.currentLane * this.player.laneHeight + this.player.laneHeight / 2;
                    
                    this.letters.push({
                        x: this.mermaid.x + 15,
                        y: y,
                        width: 25,
                        height: 25,
                        letter: data.nextLetter,
                        lane: this.mermaid.currentLane,
                        speed: this.gameState.speed + 1
                    });
                }
            } else if (data.gameComplete) {
                this.gameState.gameComplete = true;
                this.endGame(true);
            }
        } catch (error) {
            console.error('Failed to report missed letter:', error);
        }
    }

    movePlayer(direction) {
        if (!this.gameState.isPlaying) return;
        
        const newLane = this.player.targetLane + direction;
        if (newLane >= 0 && newLane <= 2) {
            this.player.targetLane = newLane;
        }
    }

    movePlayerMobile(direction) {
        if (!this.gameState.isPlaying) return;
        
        const newLane = this.player.targetLane + direction;
        if (newLane >= 0 && newLane <= 2) {
            this.player.targetLane = newLane;
        }
    }

    updatePlayer() {
        if (this.gameState.isMobile) {
            const targetX = this.player.targetLane * this.player.laneHeight + this.player.laneHeight / 2;
            this.player.x += (targetX - this.player.x) * 0.1;
        } else {
            const targetY = this.player.targetLane * this.player.laneHeight + this.player.laneHeight / 2;
            this.player.y += (targetY - this.player.y) * 0.1;
        }
    }

    updateMermaid() {
        
        if (this.mermaid.isSinging) {
            this.mermaid.singingTimer += 16;
            if (this.mermaid.singingTimer >= this.mermaid.singingDuration) {
                this.mermaid.isSinging = false;
                this.mermaid.singingTimer = 0;
            }
        }
        
        
        if (!this.mermaid.isSinging) {
            this.mermaid.moveTimer += 16; 
            
            if (this.mermaid.moveTimer >= this.mermaid.moveInterval) {
                this.mermaid.moveTimer = 0;
                this.mermaid.currentLane = Math.floor(Math.random() * 3);
            }
        }
        
        
        this.mermaid.noteSpawnTimer += 16;
        if (this.mermaid.noteSpawnTimer >= this.mermaid.noteSpawnInterval) {
            if (Math.random() < 0.3) { 
                this.spawnNote();
            }
            this.mermaid.noteSpawnTimer = 0;
        }
        
        
        if (this.gameState.isMobile) {
            const targetX = this.mermaid.currentLane * this.player.laneHeight + this.player.laneHeight / 2;
            this.mermaid.x += (targetX - this.mermaid.x) * 0.02;
        } else {
            const targetY = this.mermaid.currentLane * this.player.laneHeight + this.player.laneHeight / 2;
            this.mermaid.y += (targetY - this.mermaid.y) * 0.02;
        }
    }

    updateLetters() {
        for (let i = this.letters.length - 1; i >= 0; i--) {
            const letter = this.letters[i];
            
            if (this.gameState.isMobile) {
                letter.y += letter.speed; 
                
                if (this.checkCollision(this.player, letter) && this.checkSameLane(this.player, letter)) {
                    this.collectLetter(letter.letter);
                    this.letters.splice(i, 1);
                    continue;
                }
                
                if (letter.y > this.canvas.height + 50) {
                    this.letters.splice(i, 1);
                    this.reportMissedLetter();
                }
            } else {
                letter.x += letter.speed; 
                
                if (this.checkCollision(this.player, letter) && this.checkSameLane(this.player, letter)) {
                    this.collectLetter(letter.letter);
                    this.letters.splice(i, 1);
                    continue;
                }
                
                if (letter.x > this.canvas.width + 50) {
                    this.letters.splice(i, 1);
                    this.reportMissedLetter();
                }
            }
        }
    }

    updateNotes() {
        for (let i = this.notes.length - 1; i >= 0; i--) {
            const note = this.notes[i];
            
            if (this.gameState.isMobile) {
                note.y += note.speed; 
                
                
                if (this.checkCollision(this.player, note) && this.checkSameLane(this.player, note)) {
                    this.collectNote(note.note);
                    this.notes.splice(i, 1);
                    continue;
                }
                
                
                if (note.y > this.canvas.height + 50) {
                    this.notes.splice(i, 1);
                }
            } else {
                note.x += note.speed; 
                
                
                if (this.checkCollision(this.player, note) && this.checkSameLane(this.player, note)) {
                    this.collectNote(note.note);
                    this.notes.splice(i, 1);
                    continue;
                }
                
                
                if (note.x > this.canvas.width + 50) {
                    this.notes.splice(i, 1);
                }
            }
        }
    }

    updateParticles() {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const particle = this.particles[i];
            particle.x += particle.vx;
            particle.y += particle.vy;
            particle.life -= 1;
            
            if (particle.life <= 0) {
                this.particles.splice(i, 1);
            }
        }
    }

    checkCollision(obj1, obj2) {
        return obj1.x < obj2.x + obj2.width &&
               obj1.x + obj1.width > obj2.x &&
               obj1.y < obj2.y + obj2.height &&
               obj1.y + obj1.height > obj2.y;
    }

    checkSameLane(player, obj) {
        if (this.gameState.isMobile) {
            
            const playerLane = Math.floor(player.x / this.player.laneHeight);
            const objLane = Math.floor(obj.x / this.player.laneHeight);
            return playerLane === objLane;
        } else {
            
            const playerLane = Math.floor(player.y / this.player.laneHeight);
            const objLane = Math.floor(obj.y / this.player.laneHeight);
            return playerLane === objLane;
        }
    }

    createParticles(letter, color = '#ffff00') {
        for (let i = 0; i < 10; i++) {
            this.particles.push({
                x: this.player.x + this.player.width / 2,
                y: this.player.y + this.player.height / 2,
                vx: (Math.random() - 0.5) * 10,
                vy: (Math.random() - 0.5) * 10,
                life: 30,
                letter: letter,
                color: color
            });
        }
    }

    updateInventory() {
        this.inventory.innerHTML = '';
        
        for (let i = 0; i < this.gameState.totalLetters; i++) {
            const letterDiv = document.createElement('div');
            letterDiv.className = 'letter';
            
            if (i < this.gameState.collectedLetters.length) {
                letterDiv.textContent = this.gameState.collectedLetters[i];
                letterDiv.classList.add('collected');
            } else {
                letterDiv.textContent = '?';
            }
            
            this.inventory.appendChild(letterDiv);
        }
    }

    updateScore() {
        this.scoreElement.textContent = `Score: ${this.gameState.score}`;
    }

    updateTime() {
        const timeElement = document.getElementById('time');
        if (timeElement) {
            timeElement.textContent = `Time: ${Math.floor(this.gameState.gameTime)}s`;
        }
    }

    endGame(won = false, flag = null) {
        this.gameState.isPlaying = false;
        
        const title = document.getElementById('gameOverTitle');
        const subtitle = document.getElementById('gameOverSubtitle');
        const flagDisplay = document.getElementById('flagDisplay');
        
        if (won) {
            title.textContent = '🎉 Игра окончена 🎉';
            subtitle.textContent = 'Русалка тебя не поймала!';
            
            
            if (this.gameState.collectedLetters.length > 0) {
                const collectedText = this.gameState.collectedLetters.join('');
                flagDisplay.innerHTML = `
                    <div style="margin: 20px 0;">
                        <div style="font-size: 16px; color: #00f// Update ocean backgroundf00; margin-bottom: 10px;">Буквы флага, которые ты расслышал:</div>
                        <div style="font-size: 24px; color: #ffff00; word-break: break-all; font-family: monospace;">${collectedText}</div>
                    </div>
                `;
            } else {
                flagDisplay.textContent = '';
            }
        } else {
            title.textContent = '💀 Игра окончена 💀';
            subtitle.textContent = 'Русалка тебя поймала!';
            
            
            if (this.gameState.collectedLetters.length > 0) {
                const collectedText = this.gameState.collectedLetters.join('');
                flagDisplay.innerHTML = `
                    <div style="margin: 20px 0;">
                        <div style="font-size: 16px; color: #ffaa00; margin-bottom: 10px;">Progress:</div>
                        <div style="font-size: 20px; color: #ffff00; word-break: break-all; font-family: monospace;">${collectedText}</div>
                    </div>
                `;
            } else {
                flagDisplay.textContent = '';
            }
        }
        
        this.gameOverScreen.style.display = 'flex';
    }

    restartGame() {
        this.setupGameState();
        this.updateInventory();
        this.updateScore();
        this.startGame();
    }

    
    renderPlayer() {
        if (this.sprites.surfer) {
            
            this.ctx.drawImage(
                this.sprites.surfer,
                this.player.x - this.player.width / 2,
                this.player.y - this.player.height / 2,
                this.player.width,
                this.player.height
            );
        } else {
            
            this.ctx.fillStyle = '#00ff00';
            this.ctx.fillRect(this.player.x, this.player.y - this.player.height / 2, this.player.width, this.player.height);
            
            
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillRect(this.player.x + 2, this.player.y - 2, 5, 5);
            this.ctx.fillRect(this.player.x + 7, this.player.y + 2, 10, 15);
        }
    }

    renderMermaid() {
        const spriteName = this.mermaid.isSinging ? 'song-mermaid' : 'swim-mermaid';
        
        if (this.sprites[spriteName]) {
            
            this.ctx.drawImage(
                this.sprites[spriteName],
                this.mermaid.x - this.mermaid.width / 2,
                this.mermaid.y - this.mermaid.height / 2,
                this.mermaid.width,
                this.mermaid.height
            );
        } else {
            
            this.ctx.fillStyle = '#ff00ff';
            this.ctx.fillRect(this.mermaid.x, this.mermaid.y - this.mermaid.height / 2, this.mermaid.width, this.mermaid.height);
            
            
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillRect(this.mermaid.x + 2, this.mermaid.y - 2, 5, 5);
            this.ctx.fillRect(this.mermaid.x + 7, this.mermaid.y + 2, 10, 15);
            this.ctx.fillRect(this.mermaid.x + 10, this.mermaid.y + 17, 15, 20);
        }
    }

    renderLetters() {
        for (let letter of this.letters) {
            this.ctx.fillStyle = '#ffff00';
            this.ctx.fillRect(letter.x - letter.width / 2, letter.y - letter.height / 2, letter.width, letter.height);
            
            this.ctx.fillStyle = '#000000';
            this.ctx.font = '22px Courier New'; 
            this.ctx.textAlign = 'center';
            this.ctx.fillText(letter.letter, letter.x, letter.y + 3);
        }
    }

    renderNotes() {
        for (let note of this.notes) {
            this.ctx.fillStyle = '#ff00ff';
            this.ctx.fillRect(note.x - note.width / 2, note.y - note.height / 2, note.width, note.height);
            
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '22px Courier New';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(note.note, note.x, note.y + 4);
        }
    }

    renderParticles() {
        for (let particle of this.particles) {
            const color = particle.color || '#ffff00';
            this.ctx.fillStyle = color.replace('#', 'rgba(').replace(')', `, ${particle.life / 30})`);
            this.ctx.font = '16px Courier New';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(particle.letter, particle.x, particle.y);
        }
    }

    renderLanes() {
        const currentTime = Date.now();
        if (currentTime - this.waveAnimation.lastFrameTime > this.waveAnimation.frameDelay) {
            this.waveAnimation.frame = (this.waveAnimation.frame + 1) % this.waveAnimation.frameCount;
            this.waveAnimation.lastFrameTime = currentTime;
            this.waveAnimation.waveOffset += 2;
        }
        
        if (this.gameState.isMobile) {
            for (let i = 1; i < 3; i++) {
                const x = i * this.player.laneHeight;
                this.renderWaveLine(x, true);
            }
        } else {
            for (let i = 1; i < 3; i++) {
                const y = i * this.player.laneHeight;
                this.renderWaveLine(y, false);
            }
        }
    }

    renderWaveLine(position, isVertical) {
        const waveWidth = isVertical ? 40 : 60;
        const waveHeight = isVertical ? 40 : 60;
        const spacing = -2;
        const maxLength = isVertical ? this.canvas.height : this.canvas.width;
        
        let waveIndex = 0;
        for (let pos = 0; pos < maxLength; pos += waveHeight + spacing) {
            const waveFrame = (this.waveAnimation.frame - waveIndex) % this.waveAnimation.frameCount;
            const waveCanvas = this.waveFrames[waveFrame < 0 ? waveFrame + this.waveAnimation.frameCount : waveFrame];
            
            if (waveCanvas) {
                if (isVertical) {
                    this.ctx.drawImage(waveCanvas, position - waveWidth/2, pos, waveWidth, waveHeight);
                } else {
                    this.ctx.drawImage(waveCanvas, pos, position - waveHeight/2, waveWidth, waveHeight);
                }
            } else {
                this.ctx.strokeStyle = '#00aa00';
                this.ctx.lineWidth = 2;
                this.ctx.setLineDash([10, 10]);
                this.ctx.beginPath();
                if (isVertical) {
                    this.ctx.moveTo(position, 0);
                    this.ctx.lineTo(position, this.canvas.height);
                } else {
                    this.ctx.moveTo(0, position);
                    this.ctx.lineTo(this.canvas.width, position);
                }
                this.ctx.stroke();
                this.ctx.setLineDash([]);
                break;
            }
            waveIndex++;
        }
    }



    render() {
        this.renderOceanBackground();
        
        this.renderLanes();
        this.renderLetters();
        this.renderNotes();
        this.renderMermaid();
        this.renderPlayer();
        this.renderParticles();
    }

    update() {
        if (!this.gameState.isPlaying) return;
        
        
        this.gameState.gameTime = (Date.now() - this.gameState.startTime) / 1000; 
        
        
        this.gameState.speed = Math.min(this.gameState.maxSpeed, this.gameState.baseSpeed + this.gameState.speedIncrease * this.gameState.gameTime);
        
        
        
        this.updatePlayer();
        this.updateMermaid();
        this.updateLetters();
        this.updateNotes();
        this.updateParticles();
        this.updateOceanBackground();
        this.updateScore();
        this.updateTime();
        
        for (let letter of this.letters) {
            letter.speed = this.gameState.speed + 1;
        }
        
        for (let note of this.notes) {
            note.speed = this.gameState.speed + 1;
        }
        
        this.mermaid.speed = this.gameState.speed;
    }

    gameLoop() {
        this.update();
        this.render();
        
        if (this.gameState.isPlaying) {
            requestAnimationFrame(() => this.gameLoop());
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new SurfingGame();
});
