/**
 * NA²Q Attention Network Visualization
 * Shows how agents communicate through attention mechanism
 */

document.addEventListener('DOMContentLoaded', function () {
    const container = document.getElementById('na2q-chart');
    if (!container) return;

    // Color scheme matching website
    const colors = Highcharts.getOptions().colors;

    Highcharts.chart('na2q-chart', {
        chart: {
            type: 'networkgraph',
            height: 420,
            backgroundColor: 'transparent',
            marginTop: 40,
            marginBottom: 50
        },
        title: {
            text: null
        },
        credits: {
            enabled: true,
            text: 'Powered by Highcharts',
            href: 'https://www.highcharts.com',
            style: { fontSize: '9px', color: '#999' }
        },
        plotOptions: {
            networkgraph: {
                keys: ['from', 'to'],
                layoutAlgorithm: {
                    enableSimulation: true,
                    friction: -0.9,
                    gravitationalConstant: 0.01,
                    maxIterations: 500,
                    integration: 'verlet',
                    linkLength: 160,
                    initialPositions: 'circle'
                },
                link: {
                    color: 'rgba(100, 100, 100, 0.4)',
                    width: 1.5
                }
            }
        },
        series: [{
            accessibility: { enabled: false },
            dataLabels: {
                enabled: true,
                linkFormat: '',
                allowOverlap: false,
                style: {
                    fontSize: '11px',
                    fontWeight: '600',
                    textOutline: '2px white'
                },
                y: 0,
                verticalAlign: 'middle',
                align: 'center'
            },
            id: 'na2q-network',
            marker: {
                radius: 22
            },
            data: [
                // Center connects to all agents
                ['Team Q', 'Agent 1'],
                ['Team Q', 'Agent 2'],
                ['Team Q', 'Agent 3'],
                ['Team Q', 'Agent 4'],
                ['Team Q', 'Agent 5'],
                // Attention connections between agents
                ['Agent 1', 'Agent 2'],
                ['Agent 2', 'Agent 3'],
                ['Agent 3', 'Agent 4'],
                ['Agent 4', 'Agent 5'],
                ['Agent 5', 'Agent 1']
            ],
            nodes: [
                {
                    id: 'Team Q',
                    name: 'Team Q',
                    marker: { radius: 35 },
                    color: '#3b82f6',
                    dataLabels: {
                        style: {
                            fontSize: '12px',
                            fontWeight: '700'
                        }
                    }
                },
                {
                    id: 'Agent 1',
                    name: 'S1',
                    color: '#16a34a'
                },
                {
                    id: 'Agent 2',
                    name: 'S2',
                    color: '#16a34a'
                },
                {
                    id: 'Agent 3',
                    name: 'S3',
                    color: '#16a34a'
                },
                {
                    id: 'Agent 4',
                    name: 'S4',
                    color: '#16a34a'
                },
                {
                    id: 'Agent 5',
                    name: 'S5',
                    color: '#16a34a'
                }
            ]
        }],
        tooltip: {
            formatter: function () {
                if (this.point.id === 'Team Q') {
                    return '<b>Team Q-Value</b><br>Sum of all sensor contributions';
                }
                return '<b>Sensor ' + this.point.id.slice(-1) + '</b><br>Individual agent Q-value';
            }
        }
    });
});
