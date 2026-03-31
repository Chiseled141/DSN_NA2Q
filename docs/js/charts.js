/**
 * =============================================================================
 * NA²Q CHARTS - Training Dashboard Visualizations using Highcharts
 * Animation style EXACTLY like hardware-charts.js:
 * - Generate all data upfront
 * - Load all data into chart at once
 * - Use Highcharts built-in animation to draw smoothly
 * =============================================================================
 */

document.addEventListener('DOMContentLoaded', function () {    // Load training data from global object (set by training_data.js)
    const initCharts = () => {
        const jsonData = window.trainingData;

        if (!jsonData) {
            console.warn('No training data found in window.trainingData');
            return;
        }

        // Process data for Scenario 1 (default)
        // Use _full data for accurate dashboard charts (all episodes)
        // Use non-full data for comparison charts on index page (sampled for performance)

        const rawData = jsonData.scenario1_full || jsonData.scenario1 || {};
        const metadata = jsonData.metadata || {};

        const processSeries = (dataObj) => {
            if (!dataObj || !dataObj.episodes) return [];

            return dataObj.episodes.map((ep, i) => ({
                episode: ep,
                reward: dataObj.rewards ? dataObj.rewards[i] : null,
                coverage: dataObj.coverage ? dataObj.coverage[i] : null,
                loss: dataObj.loss ? dataObj.loss[i] : null,
                epsilon: dataObj.epsilon ? dataObj.epsilon[i] : null,
                time: dataObj.time ? dataObj.time[i] : null
            }));
        };

        // Smooth data using moving average (reduces noise for cleaner charts)
        const smoothData = (data, field, windowSize = 100) => {
            if (!data || data.length === 0) return [];

            const result = [];
            for (let i = 0; i < data.length; i += windowSize) {
                const chunk = data.slice(i, Math.min(i + windowSize, data.length));
                const avgValue = chunk.reduce((sum, d) => sum + (d[field] || 0), 0) / chunk.length;
                result.push([chunk[0].episode, Math.round(avgValue * 10) / 10]);
            }
            return result;
        };

        const na2qData = processSeries(rawData);
        const hitmacData = processSeries(jsonData.hitmac_scenario1_full || jsonData.hitmac_scenario1);

        // Update stats
        const totalEpisodesEl = document.getElementById('total-episodes');
        if (totalEpisodesEl) {
            totalEpisodesEl.textContent = "50,000";
        }

        /**
         * STATS COMPARISON CARD
         */
        const populateStats = (data, prefix) => {
            if (!data || data.length === 0) return;

            // Coverage values are already in 0–100 scale
            const cov = data.map(d => d.coverage || 0);
            const n = cov.length;

            // Mean coverage (last 10k episodes ~ last 1000 data points)
            const tail = cov.slice(Math.max(0, n - 1000));
            const mean = tail.reduce((s, v) => s + v, 0) / tail.length;

            // % of all episodes with coverage >= 50%
            const aboveHalf = (cov.filter(v => v >= 50).length / n * 100);

            // Std deviation (all episodes)
            const allMean = cov.reduce((s, v) => s + v, 0) / n;
            const std = Math.sqrt(cov.reduce((s, v) => s + (v - allMean) ** 2, 0) / n);

            // Episodes to first reach 50% (100-ep rolling avg, values in 0-100)
            let conv = null;
            for (let i = 99; i < n; i++) {
                const win = cov.slice(i - 99, i + 1);
                const avg = win.reduce((s, v) => s + v, 0) / win.length;
                if (avg >= 50) { conv = data[i].episode; break; }
            }

            // Final stability: std of last 5k episodes (~500 points)
            const finalSlice = cov.slice(Math.max(0, n - 500));
            const finalMean = finalSlice.reduce((s, v) => s + v, 0) / finalSlice.length;
            const finalStd = Math.sqrt(finalSlice.reduce((s, v) => s + (v - finalMean) ** 2, 0) / finalSlice.length);

            const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
            set(`${prefix}-stat-mean`,   mean.toFixed(1) + '%');
            set(`${prefix}-stat-best`,   aboveHalf.toFixed(1) + '%');
            set(`${prefix}-stat-std`,    '±' + std.toFixed(1) + '%');
            set(`${prefix}-stat-conv`,   conv ? conv.toLocaleString() : 'N/A');
            set(`${prefix}-stat-stable`, '±' + finalStd.toFixed(1) + '%');
        };

        populateStats(na2qData,   'na2q');
        populateStats(hitmacData, 'hitmac');

        /**
         * CHART 1: Coverage Rate (was Episode Rewards)
         */
        if (document.getElementById('reward-chart')) {
            Highcharts.chart('reward-chart', {
                chart: {
                    type: 'area',
                    height: 420,
                    backgroundColor: 'transparent',
                    zooming: { type: 'x' },
                    animation: { duration: 1500, easing: 'easeOutBounce' }
                },
                title: { text: null },
                credits: { enabled: false },
                xAxis: {
                    type: 'linear',
                    title: { text: 'Episode' },
                    max: 50000
                },
                yAxis: {
                    title: { text: 'Reward' },
                    gridLineColor: 'rgba(0,0,0,0.05)'
                },
                tooltip: {
                    shared: true,
                    valueDecimals: 2,
                    formatter: function () {
                        let s = `<b>Episode: ${this.x}</b>`;
                        this.points.forEach(point => {
                            s += `<br/><span style="color:${point.color}">\u25CF</span> ${point.series.name}: <b>${point.y}</b>`;
                        });
                        // find time for this episode if available
                        const pt = na2qData.find(d => d.episode === this.x);
                        if (pt && pt.time !== undefined) {
                            s += `<br/>Time: ${pt.time}s`;
                        }
                        return s;
                    }
                },
                plotOptions: {
                    area: {
                        marker: { radius: 2 },
                        lineWidth: 1,
                        states: { hover: { lineWidth: 1 } },
                        threshold: null,
                        fillOpacity: 0.5
                    }
                },
                legend: {
                    enabled: true,
                    align: 'center',
                    verticalAlign: 'bottom'
                },
                series: [{
                    name: 'NA²Q',
                    data: na2qData.map(d => [d.episode, d.reward]),
                    color: {
                        linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
                        stops: [
                            [0, 'rgb(79, 70, 229)'], // Indigo
                            [0.7, 'rgba(79, 70, 229, 0.1)']
                        ]
                    }
                }, {
                    name: 'HiT-MAC',
                    data: hitmacData.map(d => [d.episode, d.reward]),
                    visible: hitmacData.length > 0,
                    color: {
                        linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
                        stops: [
                            [0, 'rgb(217, 119, 6)'], // Amber
                            [0.7, 'rgba(217, 119, 6, 0.1)']
                        ]
                    }
                }]
            });
        }

        /**
         * CHART 2: Coverage Rate (Area Chart for better visibility)
         */
        if (document.getElementById('coverage-chart')) {
            Highcharts.chart('coverage-chart', {
                chart: {
                    type: 'area',
                    height: 360,
                    backgroundColor: 'transparent',
                    animation: { duration: 1500, easing: 'easeOutBounce' }
                },
                title: { text: null },
                credits: { enabled: false },
                xAxis: {
                    allowDecimals: false,
                    title: { text: 'Episode' },
                    max: 50000
                },
                yAxis: {
                    title: { text: 'Coverage %' },
                    gridLineColor: 'rgba(0,0,0,0.05)',
                    max: 100
                },
                legend: { enabled: true, align: 'center', verticalAlign: 'bottom' },
                tooltip: {
                    valueSuffix: '%',
                    shared: true
                },
                plotOptions: {
                    area: {
                        marker: {
                            enabled: false,
                            symbol: 'circle',
                            radius: 2,
                            states: {
                                hover: { enabled: true }
                            }
                        },
                        lineWidth: 1,
                        fillOpacity: 0.5
                    }
                },
                series: [{
                    name: 'NA²Q (avg)',
                    data: smoothData(na2qData, 'coverage', 100),
                    color: {
                        linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
                        stops: [
                            [0, 'rgba(79, 70, 229, 0.8)'],
                            [1, 'rgba(79, 70, 229, 0.1)']
                        ]
                    },
                    lineColor: '#4f46e5'
                }, {
                    name: 'HiT-MAC (avg)',
                    data: smoothData(hitmacData, 'coverage', 100),
                    visible: hitmacData.length > 0,
                    color: {
                        linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
                        stops: [
                            [0, 'rgba(217, 119, 6, 0.8)'],
                            [1, 'rgba(217, 119, 6, 0.1)']
                        ]
                    },
                    lineColor: '#d97706'
                }]
            });
        }


    };

    // Check if data is already loaded or wait for it
    if (window.trainingData) {
        initCharts();
    } else {
        // Simple polling if script loads async (though we'll put it before in HTML)
        window.addEventListener('load', initCharts);
    }
});
