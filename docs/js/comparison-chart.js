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
            // Round to 1 decimal place
            const avg = sum / (end - start);
            result.push([data[i][0], Math.round(avg * 10) / 10]);
        }
        return result;
    }

    if (realData && realData.scenario1 && realData.hitmac_scenario1) {
        try {
            const d1 = realData.scenario1; // NA2Q
            const d2 = realData.hitmac_scenario1; // HiT-MAC

            // NA²Q Data
            if (d1.coverage && d1.coverage.length > 0 && d1.episodes && d1.episodes.length > 0) {
                let rawData = [];
                for (let i = 0; i < d1.coverage.length; i++) {
                    const ep = d1.episodes[i] || i;
                    if (ep > 10000) break; // Strict 10k limit
                    // Clamp to 100% max
                    const val = Math.min(100, d1.coverage[i]);
                    rawData.push([ep, val]);
                }
                // Apply lighter smoothing (window 5) for better detail on 200 data points
                trainingData.na2q = applyMovingAverage(rawData, 5);
                totalEpisodes = Math.max(totalEpisodes, rawData[rawData.length - 1][0]);
                usingRealData = true;
            }

            // HiT-MAC Data
            if (d2.coverage && d2.coverage.length > 0 && d2.episodes && d2.episodes.length > 0) {
                let rawData = [];
                for (let i = 0; i < d2.coverage.length; i++) {
                    const ep = d2.episodes[i] || i;
                    if (ep > 10000) break; // Strict 10k limit
                    const val = Math.min(100, d2.coverage[i]);
                    rawData.push([ep, val]);
                }
                // Apply lighter smoothing (window 5)
                trainingData.hitmac = applyMovingAverage(rawData, 5);
                totalEpisodes = Math.max(totalEpisodes, rawData[rawData.length - 1][0]);
            }

            // Final clamp to ensure we don't accidentally exceed chart max
            totalEpisodes = Math.min(10000, totalEpisodes);

            if (usingRealData) {
                console.log(`Comparison Chart: Loaded Real Data. Episodes: ${totalEpisodes}`);
                if (display) display.textContent = `Episode: 0`;
            }
        } catch (err) {
            console.error('Comparison Chart: Error processing real data.', err);
        }
    }

    // Fallback: Generate simulated data if real data failed
    if (!usingRealData) {
        console.warn('Comparison Chart: Using simulated fallback data.');
        if (display) display.textContent = 'Using Simulated Data (Fallback)';

        // Default 10k Simulation
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
        tooltip: {
            valueDecimals: 1,
            valueSuffix: '%'
        },
        legend: { enabled: true, verticalAlign: 'top' },
        plotOptions: {
            line: {
                marker: { enabled: false },
                lineWidth: 3,
                dataLabels: {
                    enabled: true,
                    formatter: function () {
                        // Only show label for the last point
                        if (this.point.index === this.series.data.length - 1) {
                            return this.series.name + ': ' + Highcharts.numberFormat(this.y, 1) + '%';
                        }
                        return null;
                    },
                    style: {
                        fontSize: '12px',
                        fontWeight: 'bold',
                        textOutline: 'none',
                        color: 'contrast' // Highcharts will auto-contrast or we can inherit
                    },
                    // Allow color to persist from series
                    color: undefined
                }
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
