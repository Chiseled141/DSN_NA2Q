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
            height: 380,
            backgroundColor: 'transparent',
            marginTop: 30,
            marginBottom: 30
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
                    gravitationalConstant: 0.06,
                    maxIterations: 72
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
                ['Team Q', 'S1'],
                ['Team Q', 'S2'],
                ['Team Q', 'S3'],
                ['Team Q', 'S4'],
                ['Team Q', 'S5'],
                ['S1', 'S2'],
                ['S2', 'S3'],
                ['S3', 'S4'],
                ['S4', 'S5'],
                ['S5', 'S1']
            ],
            nodes: [
                {
                    id: 'Team Q',
                    name: 'Team Q',
                    marker: { radius: 32 },
                    color: '#3b82f6',
                    dataLabels: {
                        style: { fontSize: '12px', fontWeight: '700' }
                    }
                },
                { id: 'S1', name: 'S1', color: '#16a34a' },
                { id: 'S2', name: 'S2', color: '#16a34a' },
                { id: 'S3', name: 'S3', color: '#16a34a' },
                { id: 'S4', name: 'S4', color: '#16a34a' },
                { id: 'S5', name: 'S5', color: '#16a34a' }
            ]
        }],
        tooltip: {
            useHTML: true,
            formatter: function () {
                if (this.point.id === 'Team Q') {
                    return '<b>Team Q-Value</b><br>Combines all sensor Q-values using attention';
                }
                return '<b>Sensor ' + this.point.id.replace('S', '') + '</b><br>Individual agent with local observations';
            }
        }
    });
});
