/**
 * =============================================================================
 * NA²Q REPLAY - Episode Replay Viewer
 * =============================================================================
 */

class EpisodeReplay {
    constructor(canvas, customColors = null) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        // Replay state
        this.currentStep = 0;
        this.maxSteps = 100;
        this.isPlaying = false;
        this.animationId = null;
        this.lastFrameTime = 0;
        this.frameInterval = 100;  // 10 FPS
        this.onStepChange = null;  // Callback for step changes

        // Episode data (would be loaded from saved trajectories)
        this.episodeData = null;

        // Colors (light theme)
        this.colors = {
            background: '#fafafa',
            grid: 'rgba(0, 0, 0, 0.06)',
            sensorBody: '#6366f1',
            sensorFov: 'rgba(99, 102, 241, 0.22)',
            sensorFovMid: 'rgba(99, 102, 241, 0.06)',
            sensorFovBorder: 'rgba(99, 102, 241, 0.4)',
            sensorGlow: 'rgba(99, 102, 241, 0.14)',
            targetTracked: '#f59e0b',
            targetTrackedGlow: 'rgba(245, 158, 11, 0.25)',
            targetUntracked: '#64748b'
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

        // Apply custom colors if provided
        if (customColors) {
            Object.assign(this.colors, customColors);
        }

        // Initialize
        this.resize();
        window.addEventListener('resize', () => this.resize());

        // Generate sample episode
        this.generateSampleEpisode();
    }

    resize() {
        // Get dimensions from CSS-sized canvas or container
        const rect = this.canvas.getBoundingClientRect();
        const container = this.canvas.parentElement;

        // Use the CSS dimensions if available, otherwise use container dimensions
        const width = rect.width || container.clientWidth || 400;
        const height = rect.height || container.clientHeight || 400;

        // Set canvas internal resolution to match display size
        this.canvas.width = width;
        this.canvas.height = height;

        // Calculate logical size: grid + padding for sensing range on all sides
        this.fieldSize = this.config.gridSize * this.config.cellSize;
        // Add padding equal to sensing range (0.3x) to zoom in more
        const padding = this.config.sensingRange * 0.3;
        const logicalSize = this.fieldSize + (padding * 2);

        // Calculate scale to fit the logical size into the canvas
        const minDim = Math.min(this.canvas.width, this.canvas.height);
        this.scale = minDim / logicalSize;

        // Center the field within the canvas
        this.offsetX = (this.canvas.width - (this.fieldSize * this.scale)) / 2;
        this.offsetY = (this.canvas.height - (this.fieldSize * this.scale)) / 2;

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
        // Keep totalReward as number (format on display)
    }

    render() {
        if (!this.episodeData || !this.episodeData.steps[this.currentStep]) {
            return;
        }

        const ctx = this.ctx;
        const scale = this.scale;
        const offsetX = this.offsetX || 0;
        const offsetY = this.offsetY || 0;
        const step = this.episodeData.steps[this.currentStep];

        // Clear with subtle radial gradient background
        const bgGrad = ctx.createRadialGradient(
            this.canvas.width / 2, this.canvas.height / 2, 0,
            this.canvas.width / 2, this.canvas.height / 2, Math.max(this.canvas.width, this.canvas.height) * 0.7
        );
        bgGrad.addColorStop(0, '#f8faff');
        bgGrad.addColorStop(1, '#f0f2f8');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        const gridSizePx = this.fieldSize * scale;

        // Field background panel
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.beginPath();
        ctx.roundRect(offsetX - 2, offsetY - 2, gridSizePx + 4, gridSizePx + 4, 6);
        ctx.fill();
        ctx.strokeStyle = 'rgba(100,116,139,0.18)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Grid lines (dots at intersections instead of full lines)
        for (let i = 0; i <= this.config.gridSize; i++) {
            const pos = i * this.config.cellSize * scale;

            // Subtle lines
            ctx.strokeStyle = 'rgba(148,163,184,0.25)';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 6]);

            ctx.beginPath();
            ctx.moveTo(offsetX + pos, offsetY);
            ctx.lineTo(offsetX + pos, offsetY + gridSizePx);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(offsetX, offsetY + pos);
            ctx.lineTo(offsetX + gridSizePx, offsetY + pos);
            ctx.stroke();
        }
        ctx.setLineDash([]);

        // Border frame (solid)
        ctx.strokeStyle = 'rgba(100,116,139,0.35)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(offsetX, offsetY, gridSizePx, gridSizePx);

        // Axis labels
        ctx.fillStyle = '#475569';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        // X Axis labels
        for (let i = 0; i <= this.config.gridSize; i++) {
            const val = i * this.config.cellSize;
            const px = offsetX + val * scale;
            const py = offsetY + gridSizePx + 6;

            ctx.strokeStyle = 'rgba(100,116,139,0.4)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(px, offsetY + gridSizePx);
            ctx.lineTo(px, offsetY + gridSizePx + 4);
            ctx.stroke();

            ctx.fillText(val.toString(), px, py);
        }

        // Y Axis labels
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i <= this.config.gridSize; i++) {
            const val = i * this.config.cellSize;
            const px = offsetX - 8;
            const py = offsetY + gridSizePx - (val * scale);

            ctx.strokeStyle = 'rgba(100,116,139,0.4)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(offsetX, py);
            ctx.lineTo(offsetX - 4, py);
            ctx.stroke();

            ctx.fillText(val.toString(), px, py);
        }

        // Draw sensor FoVs — radial gradient cones
        for (let i = 0; i < step.sensorPositions.length; i++) {
            const sensor = step.sensorPositions[i];
            const angle = step.sensorAngles[i];
            const x = offsetX + sensor.x * scale;
            const y = offsetY + (this.fieldSize * scale) - sensor.y * scale;
            const range = this.config.sensingRange * scale;
            const startAngle = -(angle + this.config.fovAngle / 2);
            const endAngle   = -(angle - this.config.fovAngle / 2);

            // Radial gradient fill: opaque at center, transparent at edge
            const grad = ctx.createRadialGradient(x, y, 0, x, y, range);
            grad.addColorStop(0,   this.colors.sensorFov);
            grad.addColorStop(0.6, this.colors.sensorFovMid || 'rgba(99,102,241,0.06)');
            grad.addColorStop(1,   'rgba(0,0,0,0)');

            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.arc(x, y, range, startAngle, endAngle);
            ctx.closePath();
            ctx.fillStyle = grad;
            ctx.fill();

            // Dashed arc boundary
            ctx.save();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = this.colors.sensorFovBorder;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.arc(x, y, range, startAngle, endAngle);
            ctx.stroke();
            ctx.restore();

            // Edge radii lines (thin)
            ctx.strokeStyle = this.colors.sensorFovBorder;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + range * Math.cos(startAngle), y + range * Math.sin(startAngle));
            ctx.moveTo(x, y);
            ctx.lineTo(x + range * Math.cos(endAngle),   y + range * Math.sin(endAngle));
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
            const x = offsetX + target.x * scale;
            const y = offsetY + (this.fieldSize * scale) - target.y * scale;
            const tracked = trackedTargets.has(j);
            const color = tracked ? this.colors.targetTracked : this.colors.targetUntracked;
            const r = 6;

            // Outer glow ring (tracked only)
            if (tracked) {
                ctx.beginPath();
                ctx.arc(x, y, r + 6, 0, 2 * Math.PI);
                ctx.fillStyle = this.colors.targetTrackedGlow || 'rgba(34,197,94,0.2)';
                ctx.fill();
            }

            // Main filled circle
            ctx.beginPath();
            ctx.arc(x, y, r, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();

            // White outline ring
            ctx.strokeStyle = 'rgba(255,255,255,0.85)';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Inner white dot
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(255,255,255,0.9)';
            ctx.fill();
        }

        // Draw sensor bodies
        for (let i = 0; i < step.sensorPositions.length; i++) {
            const sensor = step.sensorPositions[i];
            const angle = step.sensorAngles[i];
            const x = offsetX + sensor.x * scale;
            const y = offsetY + (this.fieldSize * scale) - sensor.y * scale;
            const r = 8;
            const dirLen = 18;

            // Soft glow halo
            ctx.beginPath();
            ctx.arc(x, y, r + 5, 0, 2 * Math.PI);
            ctx.fillStyle = this.colors.sensorGlow || 'rgba(99,102,241,0.14)';
            ctx.fill();

            // White body
            ctx.beginPath();
            ctx.arc(x, y, r, 0, 2 * Math.PI);
            ctx.fillStyle = '#ffffff';
            ctx.fill();

            // Colored ring
            ctx.strokeStyle = this.colors.sensorBody;
            ctx.lineWidth = 2.5;
            ctx.stroke();

            // Inner colored dot
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fillStyle = this.colors.sensorBody;
            ctx.fill();

            // Direction line
            const tipX = x + dirLen * Math.cos(-angle);
            const tipY = y + dirLen * Math.sin(-angle);
            ctx.strokeStyle = this.colors.sensorBody;
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(tipX, tipY);
            ctx.stroke();

            // Arrowhead
            const aw = 5, aa = 0.5;
            ctx.beginPath();
            ctx.moveTo(tipX, tipY);
            ctx.lineTo(tipX - aw * Math.cos(-angle - aa), tipY - aw * Math.sin(-angle - aa));
            ctx.moveTo(tipX, tipY);
            ctx.lineTo(tipX - aw * Math.cos(-angle + aa), tipY - aw * Math.sin(-angle + aa));
            ctx.stroke();
            ctx.lineCap = 'butt';
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

            // Fire callback if set
            if (this.onStepChange) {
                this.onStepChange(this.currentStep);
            }
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    updateStepUI() {
        const stepValue = document.getElementById('replay-step-value');
        const timelineSlider = document.getElementById('timeline-slider');

        if (stepValue) stepValue.textContent = this.currentStep;
        if (timelineSlider) timelineSlider.value = this.currentStep;
    }

    loadEpisode(episodeNum) {
        // In production, this would fetch actual saved trajectory data
        // For now, regenerate sample data
        this.generateSampleEpisode();
        this.reset();
    }
}

// Global instances for dual replay
let na2qReplay = null;
let hitmacReplay = null;

// Wrapper class to synchronize two replays
class DualReplayController {
    constructor(replay1, replay2) {
        this.na2q = replay1;
        this.hitmac = replay2;
        this.isPlaying = false;
        this.frameInterval = 100;
    }

    play() {
        if (this.isPlaying) return;
        this.isPlaying = true;
        this.na2q.play();
        this.hitmac.play();
    }

    pause() {
        this.isPlaying = false;
        this.na2q.pause();
        this.hitmac.pause();
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
        this.na2q.reset();
        this.hitmac.reset();
        this.updateStats();
    }

    stepForward() {
        this.pause();
        this.na2q.stepForward();
        this.hitmac.stepForward();
        this.updateStats();
    }

    setStep(step) {
        this.na2q.setStep(step);
        this.hitmac.setStep(step);
        this.updateStats();
    }

    setSpeed(intervalMs) {
        this.frameInterval = intervalMs;
        this.na2q.frameInterval = intervalMs;
        this.hitmac.frameInterval = intervalMs;
    }

    loadEpisode(episodeNum) {
        this.na2q.loadEpisode(episodeNum);
        this.hitmac.loadEpisode(episodeNum);
        this.updateStats();
    }

    updateStats() {
        // Get current step coverage (real-time during playback)
        const na2qStep = this.na2q.currentStep;
        const hitmacStep = this.hitmac.currentStep;

        // NA²Q stats
        const na2qReward = document.getElementById('na2q-reward');
        const na2qCoverage = document.getElementById('na2q-coverage');
        let na2qCoverageValue = 0;
        if (na2qReward && this.na2q.episodeData) {
            na2qReward.textContent = this.na2q.episodeData.totalReward.toFixed(1);
        }
        if (na2qCoverage && this.na2q.episodeData && this.na2q.episodeData.steps[na2qStep]) {
            // Use current step's coverage for gauge
            na2qCoverageValue = Math.round(this.na2q.episodeData.steps[na2qStep].coverage * 100);
            na2qCoverage.textContent = `${na2qCoverageValue}%`;
        }

        // HiT-MAC stats
        const hitmacReward = document.getElementById('hitmac-reward');
        const hitmacCoverage = document.getElementById('hitmac-coverage');
        let hitmacCoverageValue = 0;
        if (hitmacReward && this.hitmac.episodeData) {
            hitmacReward.textContent = this.hitmac.episodeData.totalReward.toFixed(1);
        }
        if (hitmacCoverage && this.hitmac.episodeData && this.hitmac.episodeData.steps[hitmacStep]) {
            // Use current step's coverage for gauge
            hitmacCoverageValue = Math.round(this.hitmac.episodeData.steps[hitmacStep].coverage * 100);
            hitmacCoverage.textContent = `${hitmacCoverageValue}%`;
        }

        // Update gauges if available
        if (this.na2qGauge && this.na2qGauge.series[0]) {
            this.na2qGauge.series[0].points[0].update(na2qCoverageValue);
        }
        if (this.hitmacGauge && this.hitmacGauge.series[0]) {
            this.hitmacGauge.series[0].points[0].update(hitmacCoverageValue);
        }
    }
}

let dualController = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    const na2qCanvas = document.getElementById('replay-canvas-na2q');
    const hitmacCanvas = document.getElementById('replay-canvas-hitmac');

    if (na2qCanvas && hitmacCanvas) {
        // Dual replay mode — standalone palette, independent of chart colors
        const na2qColors = {
            sensorBody: '#0ea5e9',           // Sky blue
            sensorFov: 'rgba(14, 165, 233, 0.20)',
            sensorFovMid: 'rgba(14, 165, 233, 0.05)',
            sensorFovBorder: 'rgba(14, 165, 233, 0.35)',
            sensorGlow: 'rgba(14, 165, 233, 0.18)',
            targetTracked: '#10b981',         // Emerald
            targetTrackedGlow: 'rgba(16, 185, 129, 0.28)',
            targetUntracked: '#94a3b8'        // Slate
        };

        const hitmacColors = {
            sensorBody: '#a855f7',           // Purple
            sensorFov: 'rgba(168, 85, 247, 0.20)',
            sensorFovMid: 'rgba(168, 85, 247, 0.05)',
            sensorFovBorder: 'rgba(168, 85, 247, 0.35)',
            sensorGlow: 'rgba(168, 85, 247, 0.18)',
            targetTracked: '#10b981',         // Emerald
            targetTrackedGlow: 'rgba(16, 185, 129, 0.28)',
            targetUntracked: '#94a3b8'        // Slate
        };

        na2qReplay = new EpisodeReplay(na2qCanvas, na2qColors);
        hitmacReplay = new EpisodeReplay(hitmacCanvas, hitmacColors);

        dualController = new DualReplayController(na2qReplay, hitmacReplay);
        dualController.updateStats();

        // Set up callbacks to update stats (and gauges) during animation
        na2qReplay.onStepChange = () => dualController.updateStats();
        hitmacReplay.onStepChange = () => dualController.updateStats();

        // Coverage Rate Gauge Options
        const gaugeOptions = {
            chart: {
                type: 'solidgauge',
                backgroundColor: 'transparent',
                height: 150
            },
            title: null,
            pane: {
                center: ['50%', '85%'],
                size: '140%',
                startAngle: -90,
                endAngle: 90,
                background: {
                    backgroundColor: '#f0f0f0',
                    borderRadius: 5,
                    innerRadius: '60%',
                    outerRadius: '100%',
                    shape: 'arc'
                }
            },
            exporting: { enabled: false },
            tooltip: { enabled: false },
            credits: { enabled: false },
            yAxis: {
                min: 0,
                max: 100,
                stops: [
                    [0.3, '#ef4444'],  // red (low coverage)
                    [0.6, '#eab308'],  // yellow (medium)
                    [0.9, '#22c55e']   // green (high coverage)
                ],
                lineWidth: 0,
                tickWidth: 0,
                minorTickInterval: null,
                tickAmount: 2,
                title: { text: 'Coverage', y: -50, style: { fontSize: '11px' } },
                labels: { y: 16, style: { fontSize: '10px' } }
            },
            plotOptions: {
                solidgauge: {
                    borderRadius: 3,
                    dataLabels: {
                        y: -20,
                        borderWidth: 0,
                        useHTML: true
                    }
                }
            }
        };

        // Create NA²Q Coverage Gauge (green theme) - Only if element exists
        let na2qGauge = null;
        if (document.getElementById('gauge-na2q')) {
            na2qGauge = Highcharts.chart('gauge-na2q', Highcharts.merge(gaugeOptions, {
                yAxis: {
                    stops: [
                        [0.3, 'rgba(22, 163, 74, 0.4)'],
                        [0.6, 'rgba(22, 163, 74, 0.7)'],
                        [1.0, '#16a34a']
                    ]
                },
                series: [{
                    name: 'Coverage',
                    data: [0],
                    dataLabels: {
                        format: '<div style="text-align:center"><span style="font-size:18px;font-weight:600;color:#16a34a">{y}%</span></div>'
                    }
                }]
            }));
        }

        // Create HiT-MAC Coverage Gauge (red theme) - Only if element exists
        let hitmacGauge = null;
        if (document.getElementById('gauge-hitmac')) {
            hitmacGauge = Highcharts.chart('gauge-hitmac', Highcharts.merge(gaugeOptions, {
                yAxis: {
                    stops: [
                        [0.3, 'rgba(220, 38, 38, 0.4)'],
                        [0.6, 'rgba(220, 38, 38, 0.7)'],
                        [1.0, '#dc2626']
                    ]
                },
                series: [{
                    name: 'Coverage',
                    data: [0],
                    dataLabels: {
                        format: '<div style="text-align:center"><span style="font-size:18px;font-weight:600;color:#dc2626">{y}%</span></div>'
                    }
                }]
            }));
        }

        // Store gauges in controller for updates
        dualController.na2qGauge = na2qGauge;
        dualController.hitmacGauge = hitmacGauge;

        // Shared Controls
        const playBtn = document.getElementById('replay-play-btn');
        const stepBtn = document.getElementById('replay-step-btn');
        const resetBtn = document.getElementById('replay-reset-btn');
        const timelineSlider = document.getElementById('timeline-slider');
        const speedSelect = document.getElementById('replay-speed');

        if (playBtn) {
            playBtn.addEventListener('click', () => {
                dualController.toggle();
                playBtn.textContent = dualController.isPlaying ? 'Pause' : 'Play';
            });
        }

        if (stepBtn) {
            stepBtn.addEventListener('click', () => {
                dualController.stepForward();
                if (playBtn) playBtn.textContent = 'Play';
            });
        }

        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                dualController.reset();
                if (playBtn) playBtn.textContent = 'Play';
            });
        }

        if (timelineSlider) {
            timelineSlider.addEventListener('input', (e) => {
                dualController.pause();
                if (playBtn) playBtn.textContent = 'Play';
                dualController.setStep(parseInt(e.target.value));
            });
        }

        if (speedSelect) {
            speedSelect.addEventListener('change', (e) => {
                dualController.setSpeed(parseInt(e.target.value));
            });
        }

        // Trigger initial resize and render after layout is complete
        requestAnimationFrame(() => {
            na2qReplay.resize();
            hitmacReplay.resize();
            na2qReplay.render();
            hitmacReplay.render();
        });

        // Expose instances globally for debugging
        window.na2qReplay = na2qReplay;
        window.hitmacReplay = hitmacReplay;
        window.dualController = dualController;
        window.na2qGauge = na2qGauge;
        window.hitmacGauge = hitmacGauge;

    }
});
