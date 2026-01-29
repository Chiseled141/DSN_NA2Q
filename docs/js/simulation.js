/**
 * =============================================================================
 * NA²Q SIMULATION - JavaScript Port of DSNEnv
 * =============================================================================
 * 
 * This is a browser-based implementation of the Directional Sensor Network
 * environment. It allows real-time visualization and interaction.
 */

class DSNSimulation {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // Configuration (defaults to Scenario 1)
    this.config = {
      gridSize: 3,
      nSensors: 5,
      nTargets: 6,
      cellSize: 20.0,
      sensingRange: 18.0,
      fovAngle: Math.PI / 2,  // 90 degrees
      rotationStep: Math.PI / 36,  // 5 degrees
      maxSteps: 100,
      targetSpeedMin: 0.3,
      targetSpeedMax: 0.7
    };

    // State
    this.sensorPositions = [];
    this.sensorAngles = [];
    this.targetPositions = [];
    this.targetVelocities = [];
    this.goalMap = [];
    this.currentStep = 0;
    this.isRunning = false;
    this.animationId = null;
    this.lastFrameTime = 0;
    this.frameInterval = 1000 / 10;  // 10 FPS default
    this.speedMultiplier = 1;

    // Policy: 'ai' or 'random'
    this.policy = 'ai';

    // Colors - Light Theme
    this.colors = {
      background: '#f8fafc',
      grid: 'rgba(0, 0, 0, 0.08)',
      sensorBody: '#2563eb',
      sensorFov: 'rgba(37, 99, 235, 0.12)',
      sensorFovBorder: 'rgba(37, 99, 235, 0.5)',
      targetTracked: '#16a34a',
      targetUntracked: '#dc2626',
      text: '#1e293b'
    };

    // Initialize
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  // ==========================================================================
  // CONFIGURATION
  // ==========================================================================

  setConfig(config) {
    Object.assign(this.config, config);
    this.reset();
  }

  setScenario(scenario) {
    const presets = {
      1: { gridSize: 3, nSensors: 5, nTargets: 6 },
      2: { gridSize: 10, nSensors: 50, nTargets: 60 },
      3: { gridSize: 5, nSensors: 15, nTargets: 20 }
    };

    if (presets[scenario]) {
      this.setConfig(presets[scenario]);
    }
  }

  setSpeed(multiplier) {
    this.speedMultiplier = multiplier;
    this.frameInterval = 1000 / (10 * multiplier);
  }

  setPolicy(policy) {
    this.policy = policy;
  }

  // ==========================================================================
  // INITIALIZATION
  // ==========================================================================

  resize() {
    const container = this.canvas.parentElement;
    // Use container dimensions, keeping square aspect ratio
    const size = Math.min(container.clientWidth, container.clientHeight);
    this.canvas.width = size;
    this.canvas.height = size;
    this.scale = size / (this.config.gridSize * this.config.cellSize);
    this.render();
  }

  reset() {
    this.currentStep = 0;

    // Calculate field size
    this.fieldSize = this.config.gridSize * this.config.cellSize;

    // Initialize sensors
    this.initializeSensors();

    // Initialize targets
    this.initializeTargets();

    // Update tracking
    this.updateGoalMap();

    // Resize canvas
    this.resize();

    // Update UI
    this.updateStats();

    // Render
    this.render();
  }

  initializeSensors() {
    this.sensorPositions = [];
    this.sensorAngles = [];

    if (this.config.gridSize === 3 && this.config.nSensors === 5) {
      // Scenario 1: Fixed positions at cells 1, 3, 5, 7, 9
      const cellCenters = {
        1: [0, 0], 3: [2, 0], 5: [1, 1], 7: [0, 2], 9: [2, 2]
      };

      for (const cellNum of [1, 3, 5, 7, 9]) {
        const [col, row] = cellCenters[cellNum];
        const x = (col + 0.5) * this.config.cellSize;
        const y = (row + 0.5) * this.config.cellSize;
        this.sensorPositions.push({ x, y });
      }
    } else {
      // Random placement
      for (let i = 0; i < this.config.nSensors; i++) {
        const col = Math.floor(Math.random() * this.config.gridSize);
        const row = Math.floor(Math.random() * this.config.gridSize);
        const x = (col + 0.5) * this.config.cellSize;
        const y = (row + 0.5) * this.config.cellSize;
        this.sensorPositions.push({ x, y });
      }
    }

    // Initialize angles pointing at nearest target (or random if no targets yet)
    for (let i = 0; i < this.sensorPositions.length; i++) {
      this.sensorAngles.push(Math.random() * 2 * Math.PI);
    }
  }

  initializeTargets() {
    this.targetPositions = [];
    this.targetVelocities = [];

    const margin = this.config.cellSize * 0.1;

    for (let i = 0; i < this.config.nTargets; i++) {
      // Random position
      const x = margin + Math.random() * (this.fieldSize - 2 * margin);
      const y = margin + Math.random() * (this.fieldSize - 2 * margin);
      this.targetPositions.push({ x, y });

      // Random velocity
      const speed = this.config.targetSpeedMin +
        Math.random() * (this.config.targetSpeedMax - this.config.targetSpeedMin);
      const angle = Math.random() * 2 * Math.PI;
      this.targetVelocities.push({
        x: speed * Math.cos(angle),
        y: speed * Math.sin(angle)
      });
    }

    // Point sensors at nearest targets
    for (let i = 0; i < this.sensorPositions.length; i++) {
      let minDist = Infinity;
      let nearestIdx = 0;

      for (let j = 0; j < this.targetPositions.length; j++) {
        const dx = this.targetPositions[j].x - this.sensorPositions[i].x;
        const dy = this.targetPositions[j].y - this.sensorPositions[i].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < minDist) {
          minDist = dist;
          nearestIdx = j;
        }
      }

      const dx = this.targetPositions[nearestIdx].x - this.sensorPositions[i].x;
      const dy = this.targetPositions[nearestIdx].y - this.sensorPositions[i].y;
      this.sensorAngles[i] = Math.atan2(dy, dx);
    }
  }

  // ==========================================================================
  // SIMULATION STEP
  // ==========================================================================

  step() {
    if (this.currentStep >= this.config.maxSteps) {
      this.stop();
      return;
    }

    // Get actions based on policy
    const actions = this.getActions();

    // Apply actions (rotate sensors)
    for (let i = 0; i < this.sensorAngles.length; i++) {
      if (actions[i] === 0) {
        this.sensorAngles[i] -= this.config.rotationStep;
      } else if (actions[i] === 2) {
        this.sensorAngles[i] += this.config.rotationStep;
      }
      // Normalize angle
      this.sensorAngles[i] = ((this.sensorAngles[i] % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    }

    // Move targets
    this.updateTargets();

    // Update tracking
    this.updateGoalMap();

    // Increment step
    this.currentStep++;

    // Update UI
    this.updateStats();

    // Render
    this.render();
  }

  getActions() {
    const actions = [];

    if (this.policy === 'random') {
      // Random policy
      for (let i = 0; i < this.config.nSensors; i++) {
        actions.push(Math.floor(Math.random() * 3));
      }
    } else {
      // Simple AI: point at nearest untracked target, or nearest target if all tracked
      for (let i = 0; i < this.config.nSensors; i++) {
        const sensorPos = this.sensorPositions[i];
        const sensorAngle = this.sensorAngles[i];

        // Find best target to track
        let bestTarget = -1;
        let bestScore = -Infinity;

        for (let j = 0; j < this.targetPositions.length; j++) {
          const targetPos = this.targetPositions[j];
          const dx = targetPos.x - sensorPos.x;
          const dy = targetPos.y - sensorPos.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          // Skip if out of range
          if (dist > this.config.sensingRange) continue;

          // Calculate angle to target
          const targetAngle = Math.atan2(dy, dx);
          const angleDiff = this.normalizeAngle(targetAngle - sensorAngle);

          // Score: prefer untracked targets, then close targets, then centered targets
          const isTracked = this.isTargetTracked(j);
          const trackBonus = isTracked ? 0 : 100;
          const distScore = (this.config.sensingRange - dist) / this.config.sensingRange * 10;
          const angleScore = (1 - Math.abs(angleDiff) / Math.PI) * 5;

          const score = trackBonus + distScore + angleScore;

          if (score > bestScore) {
            bestScore = score;
            bestTarget = j;
          }
        }

        if (bestTarget >= 0) {
          // Calculate action to point at target
          const targetPos = this.targetPositions[bestTarget];
          const dx = targetPos.x - sensorPos.x;
          const dy = targetPos.y - sensorPos.y;
          const targetAngle = Math.atan2(dy, dx);
          const angleDiff = this.normalizeAngle(targetAngle - sensorAngle);

          if (Math.abs(angleDiff) < this.config.rotationStep) {
            actions.push(1);  // Stay
          } else if (angleDiff > 0) {
            actions.push(2);  // Turn right
          } else {
            actions.push(0);  // Turn left
          }
        } else {
          actions.push(1);  // Stay if no target in range
        }
      }
    }

    return actions;
  }

  updateTargets() {
    for (let i = 0; i < this.targetPositions.length; i++) {
      // Add speed variation
      const speedVar = 1 + 0.2 * Math.random();

      // Update position
      this.targetPositions[i].x += this.targetVelocities[i].x * speedVar;
      this.targetPositions[i].y += this.targetVelocities[i].y * speedVar;

      // Bounce off walls
      if (this.targetPositions[i].x < 0) {
        this.targetPositions[i].x = -this.targetPositions[i].x;
        this.targetVelocities[i].x = -this.targetVelocities[i].x;
      }
      if (this.targetPositions[i].x > this.fieldSize) {
        this.targetPositions[i].x = 2 * this.fieldSize - this.targetPositions[i].x;
        this.targetVelocities[i].x = -this.targetVelocities[i].x;
      }
      if (this.targetPositions[i].y < 0) {
        this.targetPositions[i].y = -this.targetPositions[i].y;
        this.targetVelocities[i].y = -this.targetVelocities[i].y;
      }
      if (this.targetPositions[i].y > this.fieldSize) {
        this.targetPositions[i].y = 2 * this.fieldSize - this.targetPositions[i].y;
        this.targetVelocities[i].y = -this.targetVelocities[i].y;
      }

      // Random direction change (10% chance)
      if (Math.random() < 0.1) {
        const speed = Math.sqrt(
          this.targetVelocities[i].x ** 2 + this.targetVelocities[i].y ** 2
        );
        const newAngle = Math.random() * 2 * Math.PI;
        this.targetVelocities[i].x = speed * Math.cos(newAngle);
        this.targetVelocities[i].y = speed * Math.sin(newAngle);
      }
    }
  }

  // ==========================================================================
  // TRACKING LOGIC
  // ==========================================================================

  updateGoalMap() {
    this.goalMap = [];

    for (let i = 0; i < this.config.nSensors; i++) {
      const row = [];
      for (let j = 0; j < this.targetPositions.length; j++) {
        row.push(this.isTargetInFov(i, j) ? 1 : 0);
      }
      this.goalMap.push(row);
    }
  }

  isTargetInFov(sensorIdx, targetIdx) {
    if (sensorIdx >= this.sensorPositions.length) return false;
    if (targetIdx >= this.targetPositions.length) return false;

    const sensor = this.sensorPositions[sensorIdx];
    const target = this.targetPositions[targetIdx];

    // Distance check
    const dx = target.x - sensor.x;
    const dy = target.y - sensor.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist > this.config.sensingRange) return false;

    // Angle check
    const targetAngle = Math.atan2(dy, dx);
    const angleDiff = this.normalizeAngle(targetAngle - this.sensorAngles[sensorIdx]);

    return Math.abs(angleDiff) <= this.config.fovAngle / 2;
  }

  isTargetTracked(targetIdx) {
    for (let i = 0; i < this.goalMap.length; i++) {
      if (this.goalMap[i] && this.goalMap[i][targetIdx]) return true;
    }
    return false;
  }

  normalizeAngle(angle) {
    return ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  }

  getCoverageRate() {
    let tracked = 0;
    for (let j = 0; j < this.targetPositions.length; j++) {
      if (this.isTargetTracked(j)) tracked++;
    }
    return this.targetPositions.length > 0 ? tracked / this.targetPositions.length : 0;
  }

  getTrackedCount() {
    let tracked = 0;
    for (let j = 0; j < this.targetPositions.length; j++) {
      if (this.isTargetTracked(j)) tracked++;
    }
    return tracked;
  }

  // ==========================================================================
  // RENDERING
  // ==========================================================================

  render() {
    const ctx = this.ctx;
    const scale = this.scale;

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
    for (let i = 0; i < this.sensorPositions.length; i++) {
      const sensor = this.sensorPositions[i];
      const angle = this.sensorAngles[i];
      const x = sensor.x * scale;
      const y = this.canvas.height - sensor.y * scale;  // Flip Y
      const range = this.config.sensingRange * scale;

      // FoV wedge
      ctx.fillStyle = this.colors.sensorFov;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.arc(x, y, range,
        -(angle + this.config.fovAngle / 2),
        -(angle - this.config.fovAngle / 2));
      ctx.closePath();
      ctx.fill();

      // FoV border
      ctx.strokeStyle = this.colors.sensorFovBorder;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Draw targets
    for (let j = 0; j < this.targetPositions.length; j++) {
      const target = this.targetPositions[j];
      const x = target.x * scale;
      const y = this.canvas.height - target.y * scale;  // Flip Y
      const tracked = this.isTargetTracked(j);

      ctx.fillStyle = tracked ? this.colors.targetTracked : this.colors.targetUntracked;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 2 * Math.PI);
      ctx.fill();

      // Glow effect
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 15);
      gradient.addColorStop(0, tracked ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)');
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 15, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Draw sensor bodies
    for (let i = 0; i < this.sensorPositions.length; i++) {
      const sensor = this.sensorPositions[i];
      const angle = this.sensorAngles[i];
      const x = sensor.x * scale;
      const y = this.canvas.height - sensor.y * scale;  // Flip Y

      // Sensor body
      ctx.fillStyle = this.colors.sensorBody;
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, 2 * Math.PI);
      ctx.fill();

      // Direction indicator
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + 15 * Math.cos(-angle), y + 15 * Math.sin(-angle));
      ctx.stroke();
    }
  }

  // ==========================================================================
  // PLAYBACK CONTROLS
  // ==========================================================================

  play() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.lastFrameTime = performance.now();
    this.animate();
  }

  stop() {
    this.isRunning = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  toggle() {
    if (this.isRunning) {
      this.stop();
    } else {
      this.play();
    }
  }

  animate() {
    if (!this.isRunning) return;

    const now = performance.now();
    const elapsed = now - this.lastFrameTime;

    if (elapsed >= this.frameInterval) {
      this.step();
      this.lastFrameTime = now;
    }

    this.animationId = requestAnimationFrame(() => this.animate());
  }

  updateStats() {
    const coverage = this.getCoverageRate();
    const tracked = this.getTrackedCount();

    // Update badges
    const coverageBadge = document.getElementById('coverage-badge');
    const stepBadge = document.getElementById('step-badge');

    if (coverageBadge) {
      coverageBadge.textContent = `Coverage: ${Math.round(coverage * 100)}%`;
    }
    if (stepBadge) {
      stepBadge.textContent = `Step: ${this.currentStep} / ${this.config.maxSteps}`;
    }

    // Update stat cards
    const statSensors = document.getElementById('stat-sensors');
    const statTargets = document.getElementById('stat-targets');
    const statTracked = document.getElementById('stat-tracked');
    const statCoverage = document.getElementById('stat-coverage');

    if (statSensors) statSensors.textContent = this.config.nSensors;
    if (statTargets) statTargets.textContent = this.config.nTargets;
    if (statTracked) statTracked.textContent = tracked;
    if (statCoverage) statCoverage.textContent = `${Math.round(coverage * 100)}%`;
  }
}

// Global instance
let simulation = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('simulation-canvas');
  if (canvas) {
    simulation = new DSNSimulation(canvas);
    simulation.reset();
  }
});
