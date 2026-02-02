/**
 * Comparison Chart - Animated NA²Q vs HiT-MAC Performance (Line Race)
 * Shows training progress with animated line race chart
 */

(function () {
    const playBtn = document.getElementById('chart-play-btn');
    const slider = document.getElementById('episode-slider');
    const display = document.getElementById('episode-display');
    const container = document.getElementById('comparison-chart');

    if (!container || !playBtn || !slider) {
        console.error('Comparison Chart: Missing required DOM elements');
        return;
    }

    // --- DATA LOADING LOGIC ---
    let trainingData = { na2q: [], hitmac: [] };
    let totalEpisodes = 10000;

    // Attempt to read from global window.trainingData
    let realData = window.trainingData;
    let usingRealData = false;

    // Helper: Apply moving average to smooth data
    // Data is in [[episode, value], ...] format
    function applyMovingAverage(data, windowSize) {
        if (data.length < windowSize) return data;
        const result = [];
        for (let i = 0; i < data.length; i++) {
            const start = Math.max(0, i - Math.floor(windowSize / 2));
            const end = Math.min(data.length, i + Math.floor(windowSize / 2) + 1);
            let sum = 0;
            for (let j = start; j < end; j++) {
                sum += data[j][1];
            }
            result.push([data[i][0], sum / (end - start)]);
        }
        return result;
    }

    if (realData && realData.scenario1 && realData.hitmac_scenario1) {
        try {
            const d1 = realData.scenario1; // NA2Q
            const d2 = realData.hitmac_scenario1; // HiT-MAC

            if (d1.coverage && d1.coverage.length > 0 && d1.episodes && d1.episodes.length > 0) {
                // NA²Q: Build [episode, coverage] pairs
                let rawData = [];
                for (let i = 0; i < d1.coverage.length; i++) {
                    const ep = d1.episodes[i] || i;
                    rawData.push([ep, d1.coverage[i]]);
                }
                // Apply 50-point moving average (~500 episodes) to smooth out random spikes
                trainingData.na2q = applyMovingAverage(rawData, 50);
                totalEpisodes = d1.episodes[d1.episodes.length - 1] || d1.coverage.length;
                usingRealData = true;
            }

            if (d2.coverage && d2.coverage.length > 0 && d2.episodes && d2.episodes.length > 0) {
                // HiT-MAC: Build [episode, coverage] pairs
                let rawData = [];
                for (let i = 0; i < d2.coverage.length; i++) {
                    const ep = d2.episodes[i] || i;
                    rawData.push([ep, d2.coverage[i]]);
                }
                // Apply 50-point moving average (~500 episodes) to smooth out random spikes
                trainingData.hitmac = applyMovingAverage(rawData, 50);
            }

            if (usingRealData) {
                const na2qMax = trainingData.na2q.length > 0 ? trainingData.na2q[trainingData.na2q.length - 1][0] : 0;
                const hitmacMax = trainingData.hitmac.length > 0 ? trainingData.hitmac[trainingData.hitmac.length - 1][0] : 0;
                totalEpisodes = Math.max(na2qMax, hitmacMax, totalEpisodes);
                console.log(`Comparison Chart: NA²Q ${trainingData.na2q.length} pts (0-${na2qMax}), HiT-MAC ${trainingData.hitmac.length} pts (0-${hitmacMax})`);
                if (display) display.textContent = `Episode: 0`;
            }
        } catch (err) {
            console.error('Comparison Chart: Error processing real data.', err);
        }
    }

    // Fallback: Generate simulated data if real data failed
    if (!usingRealData) {
        console.warn('Comparison Chart: Using simulated fallback data.');
        if (display) display.textContent = 'Using Simulated Data';

        const maxSim = 10000;
        const step = 100;

        for (let ep = 0; ep <= maxSim; ep += step) {
            let na2qSum = 0, hitmacSum = 0;
            const samples = step;

            for (let j = 0; j < samples; j++) {
                const i = ep + j;
                // Simulation math
                const nP = 1 - Math.exp(-i / 3000);
                const nN = (Math.random() - 0.5) * 0.03;
                na2qSum += Math.min(0.95, 0.25 + (0.70 * nP) + nN);

                const hP = 1 - Math.exp(-i / 2000);
                const hN = (Math.random() - 0.5) * 0.04;
                hitmacSum += Math.min(0.88, 0.30 + (0.55 * hP) + hN);
            }
            // Push averages
            trainingData.na2q.push(Math.round((na2qSum / samples) * 1000) / 10);
            trainingData.hitmac.push(Math.round((hitmacSum / samples) * 1000) / 10);
        }
        totalEpisodes = maxSim;
    }

    // --- CHART SETUP ---
    const totalPoints = trainingData.na2q.length;
    // Calculate step size for slider interaction
    // If we have 100 points covering 10000 episodes, step size is ~100
    // If we have 100 points covering 100 episodes, step size is ~1
    const computedStep = totalPoints > 1 ? Math.floor(totalEpisodes / (totalPoints - 1)) : 1;
    const stepSize = Math.max(1, computedStep);

    // Update Slider limits
    slider.min = 0;
    slider.max = totalEpisodes;
    slider.step = stepSize;
    slider.value = 0;

    // Initialize Highchart
    const chart = Highcharts.chart('comparison-chart', {
        chart: {
            type: 'line',
            backgroundColor: 'transparent',
            height: 350,
            animation: { duration: 200 }
        },
        title: { text: null },
        credits: { enabled: false },
        xAxis: {
            min: 0,
            max: totalEpisodes,
            title: { text: 'Episodes' },
            labels: {
                formatter: function () {
                    return this.value.toLocaleString();
                }
            }
        },
        yAxis: {
            min: 0,
            max: 100,
            title: { text: 'Coverage (%)' },
            labels: { format: '{value}%' }
        },
        legend: { enabled: true, verticalAlign: 'top' },
        plotOptions: {
            line: {
                marker: { enabled: false },
                lineWidth: 3
            }
        },
        series: [{
            name: 'NA²Q',
            data: [trainingData.na2q[0]], // Start with just first point
            color: '#16a34a'
        }, {
            name: 'HiT-MAC',
            data: [trainingData.hitmac[0]],
            color: '#dc2626'
        }]
    });

    // Animate/Update function
    let interval = null;
    let currentEpisode = 0;

    function updateToEpisode(episode, animate) {
        currentEpisode = episode;

        // Update display text
        if (display) display.textContent = `Episode: ${episode.toLocaleString()}`;

        // Filter data points up to current episode
        // Data is in [episode, coverage] format
        const d1 = trainingData.na2q.filter(p => p[0] <= episode);
        const d2 = trainingData.hitmac.filter(p => p[0] <= episode);

        chart.series[0].setData(d1, false);
        chart.series[1].setData(d2, false);
        chart.xAxis[0].setExtremes(0, Math.max(episode, 100), false);
        chart.redraw(animate);
    }

    function stop() {
        if (interval) clearInterval(interval);
        interval = null;
        playBtn.textContent = '▶';
    }

    // Listeners
    playBtn.onclick = function () {
        if (interval) {
            stop();
        } else {
            playBtn.textContent = '⏸';
            // Start animation - increment by stepSize episodes each frame
            interval = setInterval(() => {
                const nextEp = currentEpisode + stepSize;
                if (nextEp > totalEpisodes) {
                    stop();
                    return;
                }
                slider.value = nextEp;
                updateToEpisode(nextEp, true);
            }, 100);
        }
    };

    slider.oninput = function () {
        stop();
        updateToEpisode(parseInt(this.value), false);
    };

    // Initial render
    updateToEpisode(0, false);

})();
