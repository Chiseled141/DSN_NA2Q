/**
 * =============================================================================
 * NA²Q CHARTS - Training Dashboard Visualizations
 * Loads real training data from JSON when available
 * =============================================================================
 */

// Training data storage
let trainingData = {
    scenario1: { episodes: [], rewards: [], coverage: [], loss: [], epsilon: [] },
    scenario2: { episodes: [], rewards: [], coverage: [], loss: [], epsilon: [] }
};

let metadata = {
    scenario1: { total_episodes: 0, final_coverage: 0, best_reward: 0 },
    scenario2: { total_episodes: 0, final_coverage: 0, best_reward: 0 }
};

// Chart.js default config
Chart.defaults.color = '#64748b';
Chart.defaults.borderColor = 'rgba(0, 0, 0, 0.08)';
Chart.defaults.font.family = 'Inter, sans-serif';

// Chart instances
let rewardChart, coverageChart, lossChart, epsilonChart;
let currentScenario = 'scenario1';
let dataLoaded = false;

// Load training data from JSON
async function loadTrainingData() {
    try {
        const response = await fetch('data/training_data.json');
        if (response.ok) {
            const data = await response.json();

            if (data.scenario1) {
                trainingData.scenario1 = data.scenario1;
                metadata.scenario1 = data.metadata || {};
            }
            if (data.scenario2) {
                trainingData.scenario2 = data.scenario2;
            }

            dataLoaded = true;
            console.log('✅ Loaded real training data');
            return true;
        }
    } catch (e) {
        console.log('ℹ️ No training data found, using sample data');
    }

    // Generate sample data if no real data
    generateSampleData();
    return false;
}

// Generate sample data for preview
function generateSampleData() {
    const numPoints = 300;

    for (let i = 0; i < numPoints; i++) {
        const episode = i * 100;
        trainingData.scenario1.episodes.push(episode);
        trainingData.scenario2.episodes.push(episode);

        const progress1 = Math.min(1, i / 150);
        const progress2 = Math.min(1, i / 100);

        trainingData.scenario1.rewards.push(
            0.5 + progress1 * 2 + (Math.random() - 0.5) * 0.3
        );
        trainingData.scenario2.rewards.push(
            0.3 + progress2 * 1.8 + (Math.random() - 0.5) * 0.4
        );

        trainingData.scenario1.coverage.push(
            30 + progress1 * 55 + (Math.random() - 0.5) * 8
        );
        trainingData.scenario2.coverage.push(
            25 + progress2 * 50 + (Math.random() - 0.5) * 10
        );

        trainingData.scenario1.loss.push(
            2 * Math.exp(-i / 50) + 0.1 + Math.random() * 0.05
        );
        trainingData.scenario2.loss.push(
            2.5 * Math.exp(-i / 40) + 0.15 + Math.random() * 0.08
        );

        trainingData.scenario1.epsilon.push(
            Math.max(0.05, 1 - episode / 7500)
        );
        trainingData.scenario2.epsilon.push(
            Math.max(0.05, 1 - episode / 1000)
        );
    }

    metadata.scenario1 = { total_episodes: 30000, final_coverage: 85, best_reward: 2.8, training_time: '4.2h' };
    metadata.scenario2 = { total_episodes: 10000, final_coverage: 75, best_reward: 2.1, training_time: '6.8h' };
}

// Common chart options
function getCommonOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600, easing: 'easeOutQuart' },
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                titleColor: '#1e293b',
                bodyColor: '#64748b',
                borderColor: 'rgba(0, 0, 0, 0.1)',
                borderWidth: 1,
                cornerRadius: 6,
                padding: 10
            }
        },
        scales: {
            x: {
                grid: { display: false },
                ticks: { maxTicksLimit: 6, color: '#94a3b8' }
            },
            y: {
                grid: { color: 'rgba(0, 0, 0, 0.05)' },
                ticks: { color: '#94a3b8' }
            }
        }
    };
}

// Create all charts
function createCharts() {
    const commonOptions = getCommonOptions();

    // Reward Chart
    const rewardCtx = document.getElementById('reward-chart');
    if (rewardCtx) {
        rewardChart = new Chart(rewardCtx, {
            type: 'line',
            data: {
                labels: trainingData[currentScenario].episodes,
                datasets: [{
                    label: 'Episode Reward',
                    data: trainingData[currentScenario].rewards,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: commonOptions
        });
    }

    // Coverage Chart
    const coverageCtx = document.getElementById('coverage-chart');
    if (coverageCtx) {
        coverageChart = new Chart(coverageCtx, {
            type: 'line',
            data: {
                labels: trainingData[currentScenario].episodes,
                datasets: [{
                    label: 'Coverage %',
                    data: trainingData[currentScenario].coverage,
                    borderColor: '#16a34a',
                    backgroundColor: 'rgba(22, 163, 74, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: 0, max: 100 } } }
        });
    }

    // Loss Chart
    const lossCtx = document.getElementById('loss-chart');
    if (lossCtx) {
        lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: trainingData[currentScenario].episodes,
                datasets: [{
                    label: 'Loss',
                    data: trainingData[currentScenario].loss,
                    borderColor: '#ca8a04',
                    backgroundColor: 'rgba(202, 138, 4, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: commonOptions
        });
    }

    // Epsilon Chart
    const epsilonCtx = document.getElementById('epsilon-chart');
    if (epsilonCtx) {
        epsilonChart = new Chart(epsilonCtx, {
            type: 'line',
            data: {
                labels: trainingData[currentScenario].episodes,
                datasets: [{
                    label: 'Epsilon',
                    data: trainingData[currentScenario].epsilon,
                    borderColor: '#7c3aed',
                    backgroundColor: 'rgba(124, 58, 237, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }]
            },
            options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: 0, max: 1 } } }
        });
    }
}

// Update charts for scenario
function updateChartsForScenario(scenario) {
    currentScenario = scenario;

    const charts = [
        { chart: rewardChart, data: trainingData[scenario].rewards },
        { chart: coverageChart, data: trainingData[scenario].coverage },
        { chart: lossChart, data: trainingData[scenario].loss },
        { chart: epsilonChart, data: trainingData[scenario].epsilon }
    ];

    charts.forEach(({ chart, data }) => {
        if (chart && data) {
            chart.data.labels = trainingData[scenario].episodes;
            chart.data.datasets[0].data = data;
            chart.update('none');
        }
    });

    updateStats(scenario);
}

// Update stats cards
function updateStats(scenario) {
    const m = metadata[scenario] || metadata.scenario1;

    document.getElementById('total-episodes')?.textContent = m.total_episodes?.toLocaleString() || '30,000';
    document.getElementById('final-coverage')?.textContent = `${m.final_coverage || 85}%`;
    document.getElementById('best-reward')?.textContent = m.best_reward || '2.8';
    document.getElementById('training-time')?.textContent = m.training_time || '4.2h';
}

// Show data source notice
function showDataNotice(isReal) {
    const notice = document.createElement('div');
    notice.style.cssText = 'position:fixed;bottom:1rem;right:1rem;padding:0.5rem 1rem;border-radius:6px;font-size:0.8rem;z-index:1000;';

    if (isReal) {
        notice.style.background = '#dcfce7';
        notice.style.color = '#166534';
        notice.textContent = '✓ Showing real training data';
    } else {
        notice.style.background = '#fef3c7';
        notice.style.color = '#92400e';
        notice.innerHTML = '⚠️ Showing sample data. <a href="#" style="color:#92400e">Export real data</a>';
    }

    document.body.appendChild(notice);
    setTimeout(() => notice.remove(), 5000);
}

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    const hasRealData = await loadTrainingData();
    createCharts();
    updateStats('scenario1');
    showDataNotice(hasRealData);

    // Tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const scenario = btn.dataset.scenario;
            if (scenario === 'compare') {
                // Show comparison (both datasets)
                [rewardChart, coverageChart].forEach(chart => {
                    if (chart) {
                        chart.data.datasets = [
                            { ...chart.data.datasets[0], label: 'Scenario 1', borderColor: '#2563eb', backgroundColor: 'transparent' },
                            { label: 'Scenario 2', data: scenario === 'compare' ? trainingData.scenario2.rewards || trainingData.scenario2.coverage : [], borderColor: '#16a34a', backgroundColor: 'transparent', borderWidth: 2, tension: 0.4, pointRadius: 0 }
                        ];
                        chart.options.plugins.legend.display = true;
                        chart.update();
                    }
                });
            } else {
                const scenarioKey = `scenario${scenario}`;
                // Reset to single dataset
                [rewardChart, coverageChart, lossChart, epsilonChart].forEach(chart => {
                    if (chart) chart.options.plugins.legend.display = false;
                });
                updateChartsForScenario(scenarioKey);
            }
        });
    });
});
