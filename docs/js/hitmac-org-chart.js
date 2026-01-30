/**
 * HiT-MAC Hierarchy Organization Chart
 * Shows Coordinator → Executor hierarchy structure
 */

document.addEventListener('DOMContentLoaded', function () {
    const container = document.getElementById('hitmac-chart');
    if (!container) return;

    Highcharts.chart('hitmac-chart', {
        chart: {
            height: 280,
            inverted: true,
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
        accessibility: {
            point: {
                descriptionFormat: '{toNode.name} reports to {fromNode.name}'
            }
        },
        series: [{
            type: 'organization',
            name: 'HiT-MAC',
            keys: ['from', 'to'],
            data: [
                ['Coordinator', 'Executor 1'],
                ['Coordinator', 'Executor 2'],
                ['Coordinator', 'Executor 3'],
                ['Coordinator', 'Executor 4'],
                ['Executor 1', 'Action 1'],
                ['Executor 2', 'Action 2'],
                ['Executor 3', 'Action 3'],
                ['Executor 4', 'Action 4']
            ],
            levels: [{
                level: 0,
                color: '#dc2626',
                dataLabels: { color: 'white' },
                height: 30
            }, {
                level: 1,
                color: '#3b82f6',
                dataLabels: { color: 'white' },
                height: 25
            }, {
                level: 2,
                color: '#16a34a',
                dataLabels: { color: 'white' },
                height: 22
            }],
            nodes: [{
                id: 'Coordinator',
                title: 'BOSS',
                name: 'Coordinator',
                info: 'Assigns targets to executors'
            }, {
                id: 'Executor 1',
                title: 'Worker',
                name: 'Executor 1'
            }, {
                id: 'Executor 2',
                title: 'Worker',
                name: 'Executor 2'
            }, {
                id: 'Executor 3',
                title: 'Worker',
                name: 'Executor 3'
            }, {
                id: 'Executor 4',
                title: 'Worker',
                name: 'Executor 4'
            }, {
                id: 'Action 1',
                name: 'Rotate'
            }, {
                id: 'Action 2',
                name: 'Pan'
            }, {
                id: 'Action 3',
                name: 'Track'
            }, {
                id: 'Action 4',
                name: 'Zoom'
            }],
            colorByPoint: false,
            dataLabels: {
                color: 'white',
                style: {
                    fontSize: '10px',
                    fontWeight: '600'
                }
            },
            borderColor: 'white',
            nodeWidth: 60
        }],
        tooltip: {
            outside: true,
            formatter: function () {
                if (this.point.id === 'Coordinator') {
                    return '<b>Coordinator</b><br>Assigns targets to executors';
                } else if (this.point.id.startsWith('Executor')) {
                    return '<b>' + this.point.name + '</b><br>Controls camera movement';
                }
                return '<b>' + this.point.name + '</b><br>Camera action';
            }
        }
    });
});
