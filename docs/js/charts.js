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
         * CHART 1: Episode Rewards
         */
        if (document.getElementById('reward-chart')) {
            Highcharts.chart('reward-chart', {
                chart: {
                    type: 'area',
                    height: 280,
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
                            [0, 'rgb(22, 163, 74)'], // Green
                            [0.7, 'rgba(22, 163, 74, 0.1)']
                        ]
                    }
                }, {
                    name: 'HiT-MAC',
                    data: hitmacData.map(d => [d.episode, d.reward]),
                    visible: hitmacData.length > 0, // Hide if no data
                    color: {
                        linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
                        stops: [
                            [0, 'rgb(220, 38, 38)'], // Red
                            [0.7, 'rgba(220, 38, 38, 0.1)']
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
                    height: 280,
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
                            [0, 'rgba(22, 163, 74, 0.8)'],
                            [1, 'rgba(22, 163, 74, 0.1)']
                        ]
                    },
                    lineColor: '#16a34a'
                }, {
                    name: 'HiT-MAC (avg)',
                    data: smoothData(hitmacData, 'coverage', 100),
                    visible: hitmacData.length > 0,
                    color: {
                        linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
                        stops: [
                            [0, 'rgba(220, 38, 38, 0.8)'],
                            [1, 'rgba(220, 38, 38, 0.1)']
                        ]
                    },
                    lineColor: '#dc2626'
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
