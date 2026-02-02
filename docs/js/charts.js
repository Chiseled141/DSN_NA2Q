/**
 * =============================================================================
 * NA²Q CHARTS - Training Dashboard Visualizations using Highcharts
 * Animation style EXACTLY like hardware-charts.js:
 * - Generate all data upfront
 * - Load all data into chart at once
 * - Use Highcharts built-in animation to draw smoothly
 * =============================================================================
 */

document.addEventListener('DOMContentLoaded', function () {
    // Algorithm configurations (Updated Colors: Green vs Red)
    const algorithmConfigs = {
        na2q: {
            name: 'NA²Q',
            rewardBase: 0.3,
            rewardMax: 2.8,
            coverageBase: 30,
            coverageMax: 88,
            lossBase: 2.0,
            lossMin: 0.05,
            epsilonStart: 1.0,
            epsilonEnd: 0.05,
            epsilonDecay: 5000,
            color: '#16a34a', // Green
            ref1ColorStop: 'rgb(34, 197, 94)' // Lighter green
        },
        hitmac: {
            name: 'HiT-MAC',
            rewardBase: 0.2,
            rewardMax: 2.4,
            coverageBase: 25,
            coverageMax: 77,
            lossBase: 2.2,
            lossMin: 0.2,
            epsilonStart: 1.0,
            epsilonEnd: 0.1,
            epsilonDecay: 8000,
            color: '#dc2626', // Red
            ref1ColorStop: 'rgb(239, 68, 68)' // Lighter red
        }
    };

    // Load training data from global object (set by training_data.js)
    const initCharts = () => {
        const jsonData = window.trainingData;

        if (!jsonData) {
            console.warn('No training data found in window.trainingData');
            return;
        }

        // Process data for Scenario 1 (default)
        // Note: Currently mapping scenario1 data to NA2Q. 
        // TODO: Update export script to support multiple algorithms if comparison is needed.

        const rawData = jsonData.scenario1 || {};
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

        const na2qData = processSeries(rawData);
        const hitmacData = processSeries(jsonData.hitmac_scenario1);

        // Update stats
        const totalEpisodesEl = document.getElementById('total-episodes');
        if (totalEpisodesEl && metadata.total_episodes) {
            totalEpisodesEl.textContent = metadata.total_episodes.toLocaleString();
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
                    title: { text: 'Episode' }
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
         * CHART 2: Coverage Rate
         */
        if (document.getElementById('coverage-chart')) {
            Highcharts.chart('coverage-chart', {
                chart: {
                    type: 'spline',
                    height: 280,
                    backgroundColor: 'transparent',
                    animation: { duration: 1500, easing: 'easeOutBounce' }
                },
                title: { text: null },
                credits: { enabled: false },
                xAxis: {
                    title: { text: 'Episode' }
                },
                yAxis: {
                    title: { text: 'Coverage %' },
                    gridLineColor: 'rgba(0,0,0,0.05)',
                    max: 100
                },
                legend: { enabled: true, align: 'center', verticalAlign: 'bottom' },
                tooltip: { valueSuffix: '%' },
                plotOptions: {
                    spline: {
                        lineWidth: 4,
                        marker: { enabled: false }
                    }
                },
                series: [{
                    name: 'NA²Q',
                    data: na2qData.map(d => [d.episode, d.coverage]),
                    color: {
                        linearGradient: { x1: 0, x2: 1, y1: 0, y2: 0 },
                        stops: [
                            [0, '#16a34a'],
                            [1, '#4ade80']
                        ]
                    }
                }, {
                    name: 'HiT-MAC',
                    data: hitmacData.map(d => [d.episode, d.coverage]),
                    visible: hitmacData.length > 0,
                    color: {
                        linearGradient: { x1: 0, x2: 1, y1: 0, y2: 0 },
                        stops: [
                            [0, '#dc2626'],
                            [1, '#f87171']
                        ]
                    }
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
