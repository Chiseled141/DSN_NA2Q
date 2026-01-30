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
            height: 280,
            backgroundColor: 'transparent'
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
                    gravitationalConstant: 0.08,
                    maxIterations: 500,
                    integration: 'verlet',
                    linkLength: 50,
                    initialPositions: 'circle'
                },
                link: {
                    color: 'rgba(100, 100, 100, 0.5)',
                    width: 1.5
                }
            }
        },
        series: [{
            accessibility: { enabled: false },
            dataLabels: {
                enabled: true,
                linkFormat: '',
                style: {
                    fontSize: '10px',
                    fontWeight: '600',
                    textOutline: '2px white'
                }
            },
            id: 'na2q-network',
            marker: {
                radius: 15
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
                ['Agent 5', 'Agent 1'],
                ['Agent 1', 'Agent 3'],
                ['Agent 2', 'Agent 4']
            ],
            nodes: [
                {
                    id: 'Team Q',
                    name: 'Team Q-Value',
                    marker: { radius: 25 },
                    color: '#3b82f6'
                },
                {
                    id: 'Agent 1',
                    name: 'Sensor 1',
                    color: '#16a34a'
                },
                {
                    id: 'Agent 2',
                    name: 'Sensor 2',
                    color: '#16a34a'
                },
                {
                    id: 'Agent 3',
                    name: 'Sensor 3',
                    color: '#16a34a'
                },
                {
                    id: 'Agent 4',
                    name: 'Sensor 4',
                    color: '#16a34a'
                },
                {
                    id: 'Agent 5',
                    name: 'Sensor 5',
                    color: '#16a34a'
                }
            ]
        }],
        tooltip: {
            formatter: function () {
                if (this.point.id === 'Team Q') {
                    return '<b>Team Q-Value</b><br>Sum of all agent contributions';
                }
                return '<b>' + this.point.name + '</b><br>Individual agent Q-value';
            }
        }
    });
});
