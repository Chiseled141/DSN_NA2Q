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

    if (realData && realData.scenario1 && realData.hitmac_scenario1) {
        try {
            const d1 = realData.scenario1; // NA2Q
            const d2 = realData.hitmac_scenario1; // HiT-MAC

            if (d1.coverage && d1.coverage.length > 0 && d2.coverage && d2.coverage.length > 0) {
                // Determine length
                const len = Math.min(d1.coverage.length, d2.coverage.length);

                trainingData.na2q = d1.coverage.slice(0, len);
                trainingData.hitmac = d2.coverage.slice(0, len);

                // Get max episode from the data if available
                if (d1.episodes && d1.episodes.length >= len) {
                    totalEpisodes = d1.episodes[len - 1];
                } else {
                    totalEpisodes = len; // Fallback if episodes array missing
                }

                usingRealData = true;

                // Show debug success
                if (display) display.textContent = `Loaded ${len} data points`;
                console.log(`Comparison Chart: Successfully loaded ${len} points from real data.`);
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
            max: totalPoints - 1,
            title: { text: 'Episodes' },
            labels: {
                formatter: function () {
                    return (this.value * stepSize).toLocaleString();
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
    let currentIdx = 0;

    function updateToEpisode(episode, animate) {
        // Find index corresponding to episode
        // idx = episode / stepSize
        let idx = Math.floor(episode / stepSize);
        if (idx < 0) idx = 0;
        if (idx >= totalPoints) idx = totalPoints - 1;
        currentIdx = idx;

        // Update display text
        if (display) display.textContent = `Episode: ${episode.toLocaleString()}`;

        // Slice data
        const d1 = trainingData.na2q.slice(0, idx + 1);
        const d2 = trainingData.hitmac.slice(0, idx + 1);

        chart.series[0].setData(d1, false);
        chart.series[1].setData(d2, false);
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
            // Start animation
            interval = setInterval(() => {
                const nextEp = (currentIdx + 1) * stepSize;
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
