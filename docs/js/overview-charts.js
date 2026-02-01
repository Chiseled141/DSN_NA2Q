/**
 * Overview Section Charts
 * Animated Highcharts for the DSN Overview cards
 */

document.addEventListener('DOMContentLoaded', function () {

    // ========================================
    // 1. THE PROBLEM - Semi-circle Donut
    // Shows camera's limited 60° FoV vs blind spots
    // ========================================
    if (document.getElementById('problem-chart')) {
        Highcharts.chart('problem-chart', {
            chart: {
                plotBackgroundColor: null,
                plotBorderWidth: 0,
                plotShadow: false,
                backgroundColor: 'transparent',
                height: 200
            },
            title: {
                text: 'Camera<br>FoV',
                align: 'center',
                verticalAlign: 'middle',
                y: 40,
                style: {
                    fontSize: '14px',
                    fontWeight: '600',
                    color: 'var(--text-primary)'
                }
            },
            tooltip: {
                pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
            },
            credits: { enabled: false },
            accessibility: {
                point: { valueSuffix: '%' }
            },
            plotOptions: {
                pie: {
                    dataLabels: {
                        enabled: true,
                        distance: -30,
                        style: {
                            fontWeight: 'bold',
                            color: 'white',
                            fontSize: '11px'
                        }
                    },
                    startAngle: -90,
                    endAngle: 90,
                    center: ['50%', '75%'],
                    size: '110%',
                    animation: {
                        duration: 1500
                    }
                }
            },
            series: [{
                type: 'pie',
                name: 'View',
                innerSize: '50%',
                data: [
                    { name: 'Visible', y: 60, color: '#3b82f6' },
                    { name: 'Blind Spot', y: 300, color: '#e5e7eb' }
                ]
            }]
        });
    }

    // ========================================
    // 2. MULTI-AGENT APPROACH - Venn Diagram
    // Shows sensor cooperation and overlap
    // ========================================
    if (document.getElementById('multiagent-chart')) {
        Highcharts.chart('multiagent-chart', {
            chart: {
                backgroundColor: 'transparent',
                height: 200
            },
            accessibility: {
                point: {
                    valueDescriptionFormat: '{point.name}'
                }
            },
            credits: { enabled: false },
            title: { text: null },
            tooltip: {
                headerFormat: '<span style="font-size: 14px"><b>{point.point.name}</b></span><br/>',
                pointFormat: '{point.description}'
            },
            series: [{
                type: 'venn',
                data: [{
                    sets: ['A'],
                    value: 3,
                    name: 'Sensor 1',
                    color: 'rgba(59, 130, 246, 0.7)',
                    description: 'Individual sensor view'
                }, {
                    sets: ['B'],
                    value: 3,
                    name: 'Sensor 2',
                    color: 'rgba(22, 163, 74, 0.7)',
                    description: 'Individual sensor view'
                }, {
                    sets: ['C'],
                    value: 3,
                    name: 'Sensor 3',
                    color: 'rgba(168, 85, 247, 0.7)',
                    description: 'Individual sensor view'
                }, {
                    sets: ['A', 'B'],
                    value: 1,
                    name: 'Shared Coverage',
                    color: 'rgba(14, 165, 233, 0.8)',
                    description: 'Coordinated tracking zone'
                }, {
                    sets: ['B', 'C'],
                    value: 1,
                    name: 'Shared Coverage',
                    color: 'rgba(34, 197, 94, 0.8)',
                    description: 'Coordinated tracking zone'
                }, {
                    sets: ['A', 'B', 'C'],
                    value: 0.5,
                    name: 'Full Coordination',
                    color: 'rgba(251, 191, 36, 0.9)',
                    description: 'All sensors coordinating'
                }]
            }]
        });
    }

    // ========================================
    // 3. KEY METRICS - Solid Gauge
    // Shows animated coverage rate
    // ========================================
    if (document.getElementById('metrics-chart')) {
        Highcharts.chart('metrics-chart', {
            chart: {
                type: 'solidgauge',
                backgroundColor: 'transparent',
                height: 200
            },
            title: null,
            credits: { enabled: false },
            pane: {
                center: ['50%', '75%'],
                size: '120%',
                startAngle: -90,
                endAngle: 90,
                background: {
                    backgroundColor: '#e5e7eb',
                    innerRadius: '60%',
                    outerRadius: '100%',
                    shape: 'arc',
                    borderWidth: 0
                }
            },
            tooltip: {
                enabled: false
            },
            yAxis: {
                min: 0,
                max: 100,
                stops: [
                    [0.3, '#dc2626'],  // Red for low
                    [0.6, '#f59e0b'],  // Yellow for medium
                    [0.9, '#16a34a']   // Green for high
                ],
                lineWidth: 0,
                tickWidth: 0,
                minorTickInterval: null,
                tickAmount: 2,
                labels: {
                    y: 16,
                    style: {
                        fontSize: '12px'
                    }
                }
            },
            plotOptions: {
                solidgauge: {
                    dataLabels: {
                        y: -25,
                        borderWidth: 0,
                        useHTML: true,
                        format: '<div style="text-align:center"><span style="font-size:24px;font-weight:700;color:#16a34a">{y}%</span><br/><span style="font-size:11px;color:#64748b">Coverage</span></div>'
                    },
                    animation: {
                        duration: 2000
                    }
                }
            },
            series: [{
                name: 'Coverage',
                data: (function () {
                    if (window.trainingData && window.trainingData.scenario1) {
                        const cov = window.trainingData.scenario1.coverage;
                        // use the last value of NA2Q as a representative metric
                        if (cov && cov.length > 0) return [Math.round(cov[cov.length - 1])];
                    }
                    return [85]; // fallback
                })(),
                innerRadius: '60%'
            }]
        });
    }

    // ========================================
    // 4. APPLICATIONS - Arc Diagram
    // Shows connections between use cases
    // ========================================
    if (document.getElementById('applications-chart')) {
        Highcharts.chart('applications-chart', {
            chart: {
                backgroundColor: 'transparent',
                height: 200
            },
            title: { text: null },
            credits: { enabled: false },
            accessibility: {
                point: {
                    valueDescriptionFormat: '{point.from} to {point.to}'
                }
            },
            series: [{
                keys: ['from', 'to', 'weight'],
                type: 'arcdiagram',
                name: 'Applications',
                linkWeight: 2,
                centeredLinks: true,
                dataLabels: {
                    rotation: 0,
                    y: 20,
                    align: 'center',
                    color: 'var(--text-primary)',
                    style: {
                        fontSize: '9px',
                        fontWeight: '500'
                    }
                },
                offset: '60%',
                data: [
                    ['DSN', 'Security', 1],
                    ['DSN', 'Traffic', 1],
                    ['DSN', 'Wildlife', 1],
                    ['DSN', 'Smart City', 1],
                    ['Security', 'Surveillance', 1],
                    ['Traffic', 'Monitoring', 1],
                    ['Smart City', 'IoT', 1]
                ],
                nodes: [
                    { id: 'DSN', color: '#3b82f6', marker: { radius: 15 } },
                    { id: 'Security', color: '#16a34a' },
                    { id: 'Traffic', color: '#f59e0b' },
                    { id: 'Wildlife', color: '#8b5cf6' },
                    { id: 'Smart City', color: '#ec4899' },
                    { id: 'Surveillance', color: '#14b8a6' },
                    { id: 'Monitoring', color: '#f97316' },
                    { id: 'IoT', color: '#6366f1' }
                ]
            }]
        });
    }

});
