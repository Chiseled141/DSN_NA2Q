/**
 * =============================================================================
 * NA²Q CONTROLS - UI Control Handlers
 * =============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
    if (!simulation) return;

    // ==========================================================================
    // SCENARIO SELECTION
    // ==========================================================================

    const scenarioBtns = document.querySelectorAll('.mode-btn[data-scenario]');
    scenarioBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            scenarioBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const scenario = parseInt(btn.dataset.scenario);
            simulation.setScenario(scenario);

            // Update sliders to match scenario
            const presets = {
                1: { sensors: 5, targets: 6, grid: 3 },
                2: { sensors: 50, targets: 60, grid: 10 }
            };

            if (presets[scenario]) {
                updateSlider('sensors', presets[scenario].sensors);
                updateSlider('targets', presets[scenario].targets);
                updateSlider('grid', presets[scenario].grid);
            }
        });
    });

    // ==========================================================================
    // SLIDER CONTROLS
    // ==========================================================================

    function updateSlider(name, value) {
        const slider = document.getElementById(`${name}-slider`);
        const valueEl = document.getElementById(`${name}-value`);

        if (slider) slider.value = value;
        if (valueEl) {
            if (name === 'grid') {
                valueEl.textContent = `${value}×${value}`;
            } else if (name === 'speed') {
                valueEl.textContent = `${value}×`;
            } else {
                valueEl.textContent = value;
            }
        }
    }

    // Sensors slider
    const sensorsSlider = document.getElementById('sensors-slider');
    if (sensorsSlider) {
        sensorsSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            document.getElementById('sensors-value').textContent = value;
            simulation.setConfig({ nSensors: value });
        });
    }

    // Targets slider
    const targetsSlider = document.getElementById('targets-slider');
    if (targetsSlider) {
        targetsSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            document.getElementById('targets-value').textContent = value;
            simulation.setConfig({ nTargets: value });
        });
    }

    // Grid slider
    const gridSlider = document.getElementById('grid-slider');
    if (gridSlider) {
        gridSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            document.getElementById('grid-value').textContent = `${value}×${value}`;
            simulation.setConfig({ gridSize: value });
        });
    }

    // Speed slider
    const speedSlider = document.getElementById('speed-slider');
    if (speedSlider) {
        speedSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            document.getElementById('speed-value').textContent = `${value}×`;
            simulation.setSpeed(value);
        });
    }

    // ==========================================================================
    // POLICY TOGGLE
    // ==========================================================================

    const policyBtns = document.querySelectorAll('.policy-btn');
    policyBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            policyBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            simulation.setPolicy(btn.dataset.policy);
        });
    });

    // ==========================================================================
    // PLAYBACK CONTROLS
    // ==========================================================================

    const playBtn = document.getElementById('play-btn');
    const stepBtn = document.getElementById('step-btn');
    const resetBtn = document.getElementById('reset-btn');

    if (playBtn) {
        playBtn.addEventListener('click', () => {
            simulation.toggle();
            playBtn.textContent = simulation.isRunning ? '⏸' : '▶';
        });
    }

    if (stepBtn) {
        stepBtn.addEventListener('click', () => {
            simulation.stop();
            if (playBtn) playBtn.textContent = '▶';
            simulation.step();
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            simulation.stop();
            if (playBtn) playBtn.textContent = '▶';
            simulation.reset();
        });
    }

    // ==========================================================================
    // KEYBOARD SHORTCUTS
    // ==========================================================================

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;

        switch (e.key) {
            case ' ':
                e.preventDefault();
                simulation.toggle();
                if (playBtn) playBtn.textContent = simulation.isRunning ? '⏸' : '▶';
                break;
            case 'r':
                simulation.stop();
                if (playBtn) playBtn.textContent = '▶';
                simulation.reset();
                break;
            case 'ArrowRight':
                simulation.stop();
                if (playBtn) playBtn.textContent = '▶';
                simulation.step();
                break;
        }
    });
});
