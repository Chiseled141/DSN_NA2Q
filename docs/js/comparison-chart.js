/**
 * Comparison Chart - Animated NA²Q vs HiT-MAC Performance (Line Race)
 * Shows training progress with animated line race chart
 */

(function () {
    const playBtn = document.getElementById('chart-play-btn');
    const slider = document.getElementById('episode-slider');
    const display = document.getElementById('episode-display');
    const container = document.getElementById('comparison-chart');

    if (!container || !playBtn || !slider) return;

    const maxEpisode = 10000;
    const stepSize = 100;
    const animationDuration = 200;
    let sequenceTimer = null;
    let currentStep = 0;

    // Generate fake training data (averaged every 100 episodes)
    const trainingData = {
        na2q: [],
        hitmac: []
    };

    for (let ep = 0; ep <= maxEpisode; ep += stepSize) {
        let na2qSum = 0, hitmacSum = 0;
        const samples = stepSize;

        for (let j = 0; j < samples; j++) {
            const i = ep + j;
            const na2qProgress = 1 - Math.exp(-i / 3000);
            const na2qNoise = (Math.random() - 0.5) * 0.03;
            na2qSum += Math.min(0.95, 0.25 + (0.70 * na2qProgress) + na2qNoise);

            const hitmacProgress = 1 - Math.exp(-i / 2000);
            const hitmacNoise = (Math.random() - 0.5) * 0.04;
            hitmacSum += Math.min(0.88, 0.30 + (0.55 * hitmacProgress) + hitmacNoise);
        }

        trainingData.na2q.push(Math.round((na2qSum / samples) * 1000) / 10);
        trainingData.hitmac.push(Math.round((hitmacSum / samples) * 1000) / 10);
    }

    const totalSteps = trainingData.na2q.length;

    // Create Highcharts line race chart
    const chart = Highcharts.chart('comparison-chart', {
        chart: {
            type: 'line',
            backgroundColor: 'transparent',
            marginRight: 130,
            style: {
                fontFamily: 'Inter, sans-serif'
            },
            animation: {
                duration: animationDuration
            },
            height: 350
        },
        title: {
            text: null
        },
        credits: {
            enabled: false
        },
        xAxis: {
            allowDecimals: false,
            min: 0,
            max: totalSteps - 1,
            title: {
                text: 'Training Episode',
                style: { color: '#64748b' }
            },
            labels: {
                style: { color: '#64748b' },
                formatter: function () {
                    return (this.value * stepSize).toLocaleString();
                }
            },
            gridLineColor: 'rgba(0,0,0,0.05)'
        },
        yAxis: {
            min: 0,
            max: 100,
            title: {
                text: 'Coverage Rate (%)',
                style: { color: '#64748b' }
            },
            labels: {
                format: '{value}%',
                style: { color: '#64748b' }
            },
            gridLineColor: 'rgba(0,0,0,0.08)'
        },
        tooltip: {
            shared: true,
            headerFormat: '<b>Episode {point.x:.0f}00</b><br/>',
            pointFormat: '<span style="color:{series.color}">●</span> {series.name}: <b>{point.y}%</b><br/>',
            backgroundColor: 'rgba(255,255,255,0.95)',
            borderColor: '#e2e8f0',
            borderRadius: 8,
            shadow: true
        },
        legend: {
            enabled: false
        },
        plotOptions: {
            line: {
                lineWidth: 3,
                marker: {
                    enabled: false
                },
                states: {
                    hover: {
                        lineWidth: 4
                    }
                },
                dataLabels: {
                    enabled: true,
                    allowOverlap: true,
                    crop: false,
                    overflow: 'allow',
                    formatter: function () {
                        // Only show label on the last point
                        if (this.point.index === this.series.data.length - 1) {
                            return '<span style="color:' + this.series.color + '">' + this.series.name + ': ' + this.y + '%</span>';
                        }
                        return null;
                    },
                    style: {
                        fontWeight: '600',
                        fontSize: '11px',
                        textOutline: '2px white'
                    },
                    align: 'left',
                    x: 5
                }
            }
        },
        series: [
            {
                name: 'NA²Q',
                data: [trainingData.na2q[0]],
                color: '#16a34a',
                dataLabels: {
                    y: 12
                }
            },
            {
                name: 'HiT-MAC',
                data: [trainingData.hitmac[0]],
                color: '#dc2626',
                dataLabels: {
                    y: -12
                }
            }
        ]
    });

    function updateChart(step, animate) {
        currentStep = step;
        const episode = step * stepSize;
        slider.value = episode;
        display.textContent = `Episode: ${episode.toLocaleString()}`;

        const na2qSlice = trainingData.na2q.slice(0, step + 1);
        const hitmacSlice = trainingData.hitmac.slice(0, step + 1);

        // Get current values to determine label positions
        const na2qValue = na2qSlice[na2qSlice.length - 1];
        const hitmacValue = hitmacSlice[hitmacSlice.length - 1];

        // Position labels based on which line is higher
        // Higher line gets label above (-15), lower line gets label below (+15)
        const na2qAbove = na2qValue >= hitmacValue;

        chart.series[0].update({
            dataLabels: { y: na2qAbove ? -15 : 15 }
        }, false);
        chart.series[1].update({
            dataLabels: { y: na2qAbove ? 15 : -15 }
        }, false);

        chart.series[0].setData(na2qSlice, false);
        chart.series[1].setData(hitmacSlice, false);
        chart.redraw(animate);
    }

    function pause() {
        playBtn.textContent = '▶';
        playBtn.title = 'Play';
        if (sequenceTimer) {
            clearInterval(sequenceTimer);
            sequenceTimer = null;
        }
    }

    function play() {
        if (currentStep >= totalSteps - 1) {
            currentStep = 0;
        }
        playBtn.textContent = '⏸';
        playBtn.title = 'Pause';

        sequenceTimer = setInterval(() => {
            currentStep++;
            if (currentStep >= totalSteps) {
                currentStep = totalSteps - 1;
                updateChart(currentStep, true);
                pause();
                return;
            }
            updateChart(currentStep, true);
        }, animationDuration);
    }

    // Event listeners
    playBtn.addEventListener('click', () => {
        if (sequenceTimer) {
            pause();
        } else {
            play();
        }
    });

    slider.addEventListener('input', () => {
        pause();
        const episode = parseInt(slider.value, 10);
        const step = Math.floor(episode / stepSize);
        updateChart(step, false);
    });

    // Initialize with first data point
    updateChart(1, false);
})();
