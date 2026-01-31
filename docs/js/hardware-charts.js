/**
 * Hardware Performance Dashboard
 * Shows CPU and GPU usage comparison between NA2Q and HiT-MAC algorithms
 * CPU: Spline chart / GPU: Areaspline chart with filled areas
 */

document.addEventListener('DOMContentLoaded', function () {
    // Algorithm configurations
    const algorithmConfigs = {
        na2q: {
            name: 'NA²Q',
            gpuBase: 82,
            gpuVariance: 10,
            cpuBase: 45,
            cpuVariance: 15,
            color: '#16a34a' // Green
        },
        hitmac: {
            name: 'HiT-MAC',
            gpuBase: 52,
            gpuVariance: 15,
            cpuBase: 58,
            cpuVariance: 18,
            color: '#3b82f6' // Blue
        }
    };

    // Generate smooth data with noise
    function generateData(algorithm, points = 120) {
        const config = algorithmConfigs[algorithm];
        const data = [];

        for (let i = 0; i < points; i++) {
            const progress = i / points;

            let gpuMod, cpuMod;
            if (algorithm === 'na2q') {
                // NA2Q: Higher GPU, starts high and varies smoothly
                gpuMod = Math.sin(progress * Math.PI * 3) * config.gpuVariance +
                    Math.sin(progress * Math.PI * 8) * 3;
                cpuMod = Math.sin(progress * Math.PI * 2) * config.cpuVariance;
            } else {
                // HiT-MAC: Lower GPU, more volatile pattern
                gpuMod = Math.sin(progress * Math.PI * 4) * config.gpuVariance +
                    Math.sin(progress * Math.PI * 12) * 5 - progress * 15;
                cpuMod = Math.sin(progress * Math.PI * 2.5) * config.cpuVariance;
            }

            data.push({
                time: i * 30 * 1000, // 30 second intervals (60 minutes total)
                gpu: Math.min(98, Math.max(35, config.gpuBase + gpuMod + (Math.random() - 0.5) * 8)),
                cpu: Math.min(85, Math.max(25, config.cpuBase + cpuMod + (Math.random() - 0.5) * 6))
            });
        }
        return data;
    }

    const na2qData = generateData('na2q');
    const hitmacData = generateData('hitmac');

    // CPU Usage Comparison - Spline Chart
    if (document.getElementById('cpu-chart')) {
        Highcharts.chart('cpu-chart', {
            chart: {
                type: 'spline',
                height: 280,
                backgroundColor: 'transparent',
                animation: { duration: 1500, easing: 'easeOutBounce' }
            },
            title: { text: null },
            credits: { enabled: false },
            xAxis: {
                type: 'datetime',
                title: { text: 'Training Time' },
                labels: {
                    formatter: function () {
                        return Math.round(this.value / 60000) + 'm';
                    }
                }
            },
            yAxis: {
                min: 0,
                max: 100,
                title: { text: 'Utilization (%)' },
                gridLineColor: 'rgba(0,0,0,0.05)'
            },
            tooltip: {
                shared: true,
                valueSuffix: '%',
                xDateFormat: '%M:%S'
            },
            plotOptions: {
                spline: {
                    marker: { enabled: false },
                    lineWidth: 2.5
                },
                series: {
                    animation: { duration: 1500, easing: 'easeOutBounce' }
                }
            },
            legend: {
                align: 'center',
                verticalAlign: 'bottom',
                layout: 'horizontal'
            },
            series: [{
                name: 'NA²Q',
                data: na2qData.map(d => [d.time, Math.round(d.cpu)]),
                color: algorithmConfigs.na2q.color
            }, {
                name: 'HiT-MAC',
                data: hitmacData.map(d => [d.time, Math.round(d.cpu)]),
                color: algorithmConfigs.hitmac.color
            }]
        });
    }

    // GPU Usage Comparison - Areaspline Chart with filled areas
    if (document.getElementById('gpu-chart')) {
        Highcharts.chart('gpu-chart', {
            chart: {
                type: 'areaspline',
                height: 280,
                backgroundColor: 'transparent',
                animation: { duration: 1800, easing: 'easeOutBounce' }
            },
            title: { text: null },
            credits: { enabled: false },
            xAxis: {
                type: 'datetime',
                title: { text: 'Training Time' },
                labels: {
                    formatter: function () {
                        return Math.round(this.value / 60000) + 'm';
                    }
                }
            },
            yAxis: {
                min: 0,
                max: 100,
                title: { text: 'Utilization (%)' },
                gridLineColor: 'rgba(0,0,0,0.05)',
                plotBands: [{
                    from: 80,
                    to: 100,
                    color: 'rgba(22, 163, 74, 0.1)',
                    label: { text: 'Optimal', style: { color: '#16a34a', fontSize: '10px' } }
                }]
            },
            tooltip: {
                shared: true,
                valueSuffix: '%',
                xDateFormat: '%M:%S'
            },
            plotOptions: {
                areaspline: {
                    fillOpacity: 0.3,
                    marker: { enabled: false },
                    lineWidth: 2.5
                },
                series: {
                    animation: { duration: 1800, easing: 'easeOutBounce' }
                }
            },
            legend: {
                align: 'center',
                verticalAlign: 'bottom',
                layout: 'horizontal'
            },
            series: [{
                name: 'NA²Q',
                data: na2qData.map(d => [d.time, Math.round(d.gpu)]),
                color: algorithmConfigs.na2q.color,
                fillColor: {
                    linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
                    stops: [
                        [0, 'rgba(22, 163, 74, 0.4)'],
                        [1, 'rgba(22, 163, 74, 0.05)']
                    ]
                }
            }, {
                name: 'HiT-MAC',
                data: hitmacData.map(d => [d.time, Math.round(d.gpu)]),
                color: algorithmConfigs.hitmac.color,
                fillColor: {
                    linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
                    stops: [
                        [0, 'rgba(59, 130, 246, 0.4)'],
                        [1, 'rgba(59, 130, 246, 0.05)']
                    ]
                }
            }]
        });
    }
});
