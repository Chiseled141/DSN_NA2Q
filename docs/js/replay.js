/**
 * =============================================================================
 * NA²Q REPLAY - Episode Replay Viewer
 * =============================================================================
 */

class EpisodeReplay {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        // Replay state
        this.currentStep = 0;
        this.maxSteps = 100;
        this.isPlaying = false;
        this.animationId = null;
        this.lastFrameTime = 0;
        this.frameInterval = 100;  // 10 FPS

        // Episode data (would be loaded from saved trajectories)
        this.episodeData = null;

        // Colors (light theme)
        this.colors = {
            background: '#f8fafc',
            grid: 'rgba(0, 0, 0, 0.08)',
            sensorBody: '#6366f1',
            sensorFov: 'rgba(99, 102, 241, 0.2)',
            sensorFovBorder: 'rgba(99, 102, 241, 0.5)',
            targetTracked: '#22c55e',
            targetUntracked: '#ef4444'
        };

        // Configuration
        this.config = {
            gridSize: 3,
            cellSize: 20.0,
            sensingRange: 18.0,
            fovAngle: Math.PI / 2,
            nSensors: 5,
            nTargets: 6
        };

        // Initialize
        this.resize();
        window.addEventListener('resize', () => this.resize());

        // Generate sample episode
        this.generateSampleEpisode();
    }

    resize() {
        const container = this.canvas.parentElement;
        const size = Math.min(container.clientWidth, container.clientHeight);
        this.canvas.width = size;
        this.canvas.height = size;
        this.fieldSize = this.config.gridSize * this.config.cellSize;
        this.scale = size / this.fieldSize;
        this.render();
    }

    generateSampleEpisode() {
        // Generate a sample episode trajectory
        this.episodeData = {
            steps: [],
            totalReward: 0,
            avgCoverage: 0,
            finalCoverage: 0
        };

        // Initialize positions
        let sensorPositions = [];
        let sensorAngles = [];
        let targetPositions = [];
        let targetVelocities = [];

        // Fixed sensor positions for scenario 1
        const cellCenters = {
            1: [0, 0], 3: [2, 0], 5: [1, 1], 7: [0, 2], 9: [2, 2]
        };

        for (const cellNum of [1, 3, 5, 7, 9]) {
            const [col, row] = cellCenters[cellNum];
            sensorPositions.push({
                x: (col + 0.5) * this.config.cellSize,
                y: (row + 0.5) * this.config.cellSize
            });
            sensorAngles.push(Math.random() * 2 * Math.PI);
        }

        // Random targets
        for (let i = 0; i < this.config.nTargets; i++) {
            targetPositions.push({
                x: Math.random() * this.fieldSize,
                y: Math.random() * this.fieldSize
            });
            const speed = 0.3 + Math.random() * 0.4;
            const angle = Math.random() * 2 * Math.PI;
            targetVelocities.push({
                x: speed * Math.cos(angle),
                y: speed * Math.sin(angle)
            });
        }

        let totalCoverage = 0;

        // Simulate steps
        for (let step = 0; step <= this.maxSteps; step++) {
            // Calculate coverage
            let tracked = 0;
            for (let j = 0; j < targetPositions.length; j++) {
                for (let i = 0; i < sensorPositions.length; i++) {
                    const dx = targetPositions[j].x - sensorPositions[i].x;
                    const dy = targetPositions[j].y - sensorPositions[i].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist <= this.config.sensingRange) {
                        const targetAngle = Math.atan2(dy, dx);
                        let angleDiff = targetAngle - sensorAngles[i];
                        angleDiff = ((angleDiff + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
                        if (Math.abs(angleDiff) <= this.config.fovAngle / 2) {
                            tracked++;
                            break;
                        }
                    }
                }
            }

            const coverage = tracked / targetPositions.length;
            totalCoverage += coverage;
            const reward = coverage + (coverage >= 0.8 ? 0.5 : 0);
            this.episodeData.totalReward += reward;

            // Save step data
            this.episodeData.steps.push({
                sensorPositions: JSON.parse(JSON.stringify(sensorPositions)),
                sensorAngles: [...sensorAngles],
                targetPositions: JSON.parse(JSON.stringify(targetPositions)),
                coverage: coverage,
                reward: reward
            });

            // Update for next step
            // Move targets
            for (let i = 0; i < targetPositions.length; i++) {
                targetPositions[i].x += targetVelocities[i].x;
                targetPositions[i].y += targetVelocities[i].y;

                // Bounce
                if (targetPositions[i].x < 0 || targetPositions[i].x > this.fieldSize) {
                    targetVelocities[i].x *= -1;
                    targetPositions[i].x = Math.max(0, Math.min(this.fieldSize, targetPositions[i].x));
                }
                if (targetPositions[i].y < 0 || targetPositions[i].y > this.fieldSize) {
                    targetVelocities[i].y *= -1;
                    targetPositions[i].y = Math.max(0, Math.min(this.fieldSize, targetPositions[i].y));
                }
            }

            // Rotate sensors toward targets (simple AI)
            for (let i = 0; i < sensorPositions.length; i++) {
                // Find nearest target
                let minDist = Infinity;
                let nearestAngle = 0;
                for (let j = 0; j < targetPositions.length; j++) {
                    const dx = targetPositions[j].x - sensorPositions[i].x;
                    const dy = targetPositions[j].y - sensorPositions[i].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < minDist) {
                        minDist = dist;
                        nearestAngle = Math.atan2(dy, dx);
                    }
                }

                // Rotate toward target
                let angleDiff = nearestAngle - sensorAngles[i];
                angleDiff = ((angleDiff + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
                sensorAngles[i] += Math.sign(angleDiff) * Math.min(Math.abs(angleDiff), Math.PI / 36);
            }
        }

        this.episodeData.avgCoverage = totalCoverage / (this.maxSteps + 1);
        this.episodeData.finalCoverage = this.episodeData.steps[this.maxSteps].coverage;
        this.episodeData.totalReward = this.episodeData.totalReward.toFixed(2);

        // Update UI
        this.updateInfo();
    }

    render() {
        if (!this.episodeData || !this.episodeData.steps[this.currentStep]) {
            return;
        }

        const ctx = this.ctx;
        const scale = this.scale;
        const step = this.episodeData.steps[this.currentStep];

        // Clear
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw grid
        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 1;
        for (let i = 0; i <= this.config.gridSize; i++) {
            const pos = i * this.config.cellSize * scale;
            ctx.beginPath();
            ctx.moveTo(pos, 0);
            ctx.lineTo(pos, this.canvas.height);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, pos);
            ctx.lineTo(this.canvas.width, pos);
            ctx.stroke();
        }

        // Draw sensor FoVs
        for (let i = 0; i < step.sensorPositions.length; i++) {
            const sensor = step.sensorPositions[i];
            const angle = step.sensorAngles[i];
            const x = sensor.x * scale;
            const y = this.canvas.height - sensor.y * scale;
            const range = this.config.sensingRange * scale;

            ctx.fillStyle = this.colors.sensorFov;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.arc(x, y, range, -(angle + this.config.fovAngle / 2), -(angle - this.config.fovAngle / 2));
            ctx.closePath();
            ctx.fill();

            ctx.strokeStyle = this.colors.sensorFovBorder;
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Check which targets are tracked
        const trackedTargets = new Set();
        for (let j = 0; j < step.targetPositions.length; j++) {
            for (let i = 0; i < step.sensorPositions.length; i++) {
                const dx = step.targetPositions[j].x - step.sensorPositions[i].x;
                const dy = step.targetPositions[j].y - step.sensorPositions[i].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist <= this.config.sensingRange) {
                    const targetAngle = Math.atan2(dy, dx);
                    let angleDiff = targetAngle - step.sensorAngles[i];
                    angleDiff = ((angleDiff + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
                    if (Math.abs(angleDiff) <= this.config.fovAngle / 2) {
                        trackedTargets.add(j);
                        break;
                    }
                }
            }
        }

        // Draw targets
        for (let j = 0; j < step.targetPositions.length; j++) {
            const target = step.targetPositions[j];
            const x = target.x * scale;
            const y = this.canvas.height - target.y * scale;
            const tracked = trackedTargets.has(j);

            ctx.fillStyle = tracked ? this.colors.targetTracked : this.colors.targetUntracked;
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, 2 * Math.PI);
            ctx.fill();
        }

        // Draw sensor bodies
        for (let i = 0; i < step.sensorPositions.length; i++) {
            const sensor = step.sensorPositions[i];
            const angle = step.sensorAngles[i];
            const x = sensor.x * scale;
            const y = this.canvas.height - sensor.y * scale;

            ctx.fillStyle = this.colors.sensorBody;
            ctx.beginPath();
            ctx.arc(x, y, 10, 0, 2 * Math.PI);
            ctx.fill();

            ctx.strokeStyle = '#1e293b';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + 15 * Math.cos(-angle), y + 15 * Math.sin(-angle));
            ctx.stroke();
        }
    }

    setStep(step) {
        this.currentStep = Math.max(0, Math.min(step, this.maxSteps));
        this.updateStepUI();
        this.render();
    }

    play() {
        if (this.isPlaying) return;
        this.isPlaying = true;
        this.lastFrameTime = performance.now();
        this.animate();
    }

    pause() {
        this.isPlaying = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    toggle() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    reset() {
        this.pause();
        this.currentStep = 0;
        this.updateStepUI();
        this.render();
    }

    stepForward() {
        this.pause();
        if (this.currentStep < this.maxSteps) {
            this.currentStep++;
            this.updateStepUI();
            this.render();
        }
    }

    animate() {
        if (!this.isPlaying) return;

        const now = performance.now();
        const elapsed = now - this.lastFrameTime;

        if (elapsed >= this.frameInterval) {
            if (this.currentStep >= this.maxSteps) {
                this.pause();
                return;
            }

            this.currentStep++;
            this.updateStepUI();
            this.render();
            this.lastFrameTime = now;
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    updateStepUI() {
        const stepValue = document.getElementById('replay-step-value');
        const timelineSlider = document.getElementById('timeline-slider');

        if (stepValue) stepValue.textContent = `${this.currentStep} / ${this.maxSteps}`;
        if (timelineSlider) timelineSlider.value = this.currentStep;
    }

    updateInfo() {
        const rewardEl = document.getElementById('episode-reward');
        const coverageEl = document.getElementById('episode-coverage');
        const finalEl = document.getElementById('episode-final');

        if (rewardEl) rewardEl.textContent = this.episodeData.totalReward;
        if (coverageEl) coverageEl.textContent = `${Math.round(this.episodeData.avgCoverage * 100)}%`;
        if (finalEl) finalEl.textContent = `${Math.round(this.episodeData.finalCoverage * 100)}%`;
    }

    loadEpisode(episodeNum) {
        // In production, this would fetch actual saved trajectory data
        // For now, regenerate sample data
        this.generateSampleEpisode();
        this.reset();
    }
}

// Global instance
let replayViewer = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('replay-canvas');
    if (canvas) {
        replayViewer = new EpisodeReplay(canvas);
        replayViewer.render();

        // Controls
        const playBtn = document.getElementById('replay-play-btn');
        const stepBtn = document.getElementById('replay-step-btn');
        const resetBtn = document.getElementById('replay-reset-btn');
        const timelineSlider = document.getElementById('timeline-slider');
        const episodeSlider = document.getElementById('episode-slider');

        if (playBtn) {
            playBtn.addEventListener('click', () => {
                replayViewer.toggle();
                playBtn.textContent = replayViewer.isPlaying ? '⏸' : '▶';
            });
        }

        if (stepBtn) {
            stepBtn.addEventListener('click', () => {
                replayViewer.stepForward();
                if (playBtn) playBtn.textContent = '▶';
            });
        }

        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                replayViewer.reset();
                if (playBtn) playBtn.textContent = '▶';
            });
        }

        if (timelineSlider) {
            timelineSlider.addEventListener('input', (e) => {
                replayViewer.pause();
                if (playBtn) playBtn.textContent = '▶';
                replayViewer.setStep(parseInt(e.target.value));
            });
        }

        if (episodeSlider) {
            episodeSlider.addEventListener('input', (e) => {
                document.getElementById('episode-value').textContent = e.target.value;
            });

            episodeSlider.addEventListener('change', (e) => {
                replayViewer.loadEpisode(parseInt(e.target.value));
            });
        }
    }
});
