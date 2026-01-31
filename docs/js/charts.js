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

    // Generate smooth training data (like hardware-charts.js generateData)
    function generateData(algorithm, points = 40) {
        const config = algorithmConfigs[algorithm];
        const data = [];

        for (let i = 0; i < points; i++) {
            const progress = i / points;
            const episode = i * 250; // 0 to 10,000 episodes

            // Smooth learning curve with some variation
            let rewardMod, coverageMod;
            if (algorithm === 'na2q') {
                // NA²Q: Faster learning, higher final rewards
                rewardMod = Math.sin(progress * Math.PI * 2) * 0.15 +
                    Math.sin(progress * Math.PI * 5) * 0.05;
                coverageMod = Math.sin(progress * Math.PI * 3) * 3;
            } else {
                // HiT-MAC: Slower learning curve
                rewardMod = Math.sin(progress * Math.PI * 2.5) * 0.12 +
                    Math.sin(progress * Math.PI * 6) * 0.04;
                coverageMod = Math.sin(progress * Math.PI * 4) * 4;
            }

            // Learning curve (starts low, converges high)
            const learningProgress = 1 - Math.exp(-progress * 3);

            // Loss curve (starts high, decays exponentially)
            // NA2Q converges faster (higher decay rate)
            const lossDecay = algorithm === 'na2q' ? 5 : 3;
            const lossProgress = Math.exp(-progress * lossDecay);
            const lossNoise = (Math.random()) * 0.1;

            // Epsilon Decay (Linear decay usually, or exponential)
            // Starts at epsilonStart, decays to epsilonEnd over epsilonDecay steps
            // Here taking simple exponential decay for visualization
            const epsilonVal = Math.max(config.epsilonEnd,
                config.epsilonStart * Math.exp(-episode / config.epsilonDecay));

            data.push({
                episode: episode,
                reward: config.rewardBase + learningProgress * (config.rewardMax - config.rewardBase) +
                    rewardMod + (Math.random() - 0.5) * 0.1,
                coverage: config.coverageBase + learningProgress * (config.coverageMax - config.coverageBase) +
                    coverageMod + (Math.random() - 0.5) * 2,
                loss: config.lossMin + (config.lossBase - config.lossMin) * lossProgress + lossNoise,
                epsilon: epsilonVal
            });
        }
        return data;
    }

    // Generate data for both algorithms
    const na2qData = generateData('na2q');
    const hitmacData = generateData('hitmac');

    /**
     * CHART 1: Episode Rewards -> Reference "Line chart1.js" (Area with Gradient)
     */
    if (document.getElementById('reward-chart')) {
        Highcharts.chart('reward-chart', {
            chart: {
                type: 'area', // Reference 1 uses 'area'
                height: 280,
                backgroundColor: 'transparent',
                zooming: { type: 'x' }, // Ref 1 feature
                animation: { duration: 1500, easing: 'easeOutBounce' } // Keeping smooth load
            },
            title: { text: null },
            credits: { enabled: false },
            xAxis: {
                type: 'linear', // Using linear for episodes instead of datetime
                title: { text: 'Episode' }
            },
            yAxis: {
                title: { text: 'Reward' },
                gridLineColor: 'rgba(0,0,0,0.05)'
            },
            tooltip: {
                shared: true,
                valueDecimals: 2
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
                enabled: true, // Use legend to distinguish algorithms
                align: 'center',
                verticalAlign: 'bottom'
            },
            series: [{
                name: 'NA²Q',
                data: na2qData.map(d => [d.episode, Math.round(d.reward * 100) / 100]),
                color: {
                    linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
                    stops: [
                        [0, 'rgb(22, 163, 74)'], // Green
                        [0.7, 'rgba(22, 163, 74, 0.1)']
                    ]
                }
            }, {
                name: 'HiT-MAC',
                data: hitmacData.map(d => [d.episode, Math.round(d.reward * 100) / 100]),
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
     * CHART 2: Coverage Rate -> Reference "Line chart3.js" (Gradient Stroke Line)
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
                    lineWidth: 4, // Bold line like chart3
                    marker: { enabled: false }
                }
            },
            series: [{
                name: 'NA²Q',
                data: na2qData.map(d => [d.episode, Math.round(d.coverage)]),
                // Reference 3 style: Gradient ON THE LINE itself
                color: {
                    linearGradient: { x1: 0, x2: 1, y1: 0, y2: 0 }, // Left to right gradient
                    stops: [
                        [0, '#16a34a'], // Green start
                        [1, '#4ade80']  // Lighter green end
                    ]
                }
            }, {
                name: 'HiT-MAC',
                data: hitmacData.map(d => [d.episode, Math.round(d.coverage)]),
                color: {
                    linearGradient: { x1: 0, x2: 1, y1: 0, y2: 0 },
                    stops: [
                        [0, '#dc2626'], // Red start
                        [1, '#f87171']  // Lighter red end
                    ]
                }
            }]
        });
    }

    /**
     * CHART 3: Loss Comparison -> Reference "Line chart 4.js" (Solar Employment - Clean Multi-line)
     */
    if (document.getElementById('loss-chart')) {
        Highcharts.chart('loss-chart', {
            chart: {
                type: 'spline', // Line chart 4 uses standard line, spline looks smoother for loss
                height: 280,
                backgroundColor: 'transparent',
                animation: { duration: 1500, easing: 'easeOutBounce' }
            },
            title: { text: null },
            credits: { enabled: false },
            yAxis: {
                title: { text: 'Loss' },
                gridLineColor: 'rgba(0,0,0,0.05)'
            },
            xAxis: {
                accessibility: { rangeDescription: 'Range: 0 to 10000' }
            },
            legend: {
                layout: 'vertical',
                align: 'right',
                verticalAlign: 'middle' // Legend on right as per Ref 4
            },
            plotOptions: {
                series: {
                    label: { connectorAllowed: false },
                    marker: { enabled: false }
                }
            },
            series: [{
                name: 'NA²Q',
                data: na2qData.map(d => d.loss), // Ref 4 uses simple array data
                pointStart: 0,
                pointInterval: 250, // Matches our episode steps
                color: '#16a34a' // Solid Green
            }, {
                name: 'HiT-MAC',
                data: hitmacData.map(d => d.loss),
                pointStart: 0,
                pointInterval: 250,
                color: '#dc2626' // Solid Red
            }]
        });
    }

    /**
     * CHART 4: Epsilon Decay -> Reference "Line chart 2.js" (Basic Line)
     */
    if (document.getElementById('epsilon-chart')) {
        Highcharts.chart('epsilon-chart', {
            chart: {
                type: 'line', // Ref 2 uses 'line'
                height: 280,
                backgroundColor: 'transparent',
                animation: { duration: 1500, easing: 'easeOutBounce' }
            },
            title: { text: null },
            credits: { enabled: false },
            xAxis: {
                categories: null, // Using linear scale, Ref 2 used categories but linear is better here
                title: { text: 'Episode' }
            },
            yAxis: {
                title: { text: 'Epsilon' },
                gridLineColor: 'rgba(0,0,0,0.05)',
                max: 1.0
            },
            plotOptions: {
                line: {
                    dataLabels: { enabled: false }, // Ref 2 enabled them, but too crowded for 40 points
                    enableMouseTracking: true
                }
            },
            legend: { enabled: true, align: 'center', verticalAlign: 'bottom' },
            series: [{
                name: 'NA²Q',
                data: na2qData.map(d => d.epsilon),
                color: '#16a34a'
            }, {
                name: 'HiT-MAC',
                data: hitmacData.map(d => d.epsilon),
                color: '#dc2626'
            }]
        });
    }

    // Update stats
    const totalEpisodesEl = document.getElementById('total-episodes');
    if (totalEpisodesEl) totalEpisodesEl.textContent = '10,000';
});
