/**
 * =============================================================================
 * NA²Q CHARTS - Training Dashboard Visualizations using Highcharts
 * =============================================================================
 */

document.addEventListener('DOMContentLoaded', function () {
    const initCharts = () => {
        const jsonData = window.trainingData;
        if (!jsonData) {
            console.warn('No training data found in window.trainingData');
            return;
        }

        const rawData = jsonData.scenario1_full || jsonData.scenario1 || {};

        const processSeries = (dataObj) => {
            if (!dataObj || !dataObj.episodes) return [];
            return dataObj.episodes.map((ep, i) => ({
                episode:  ep,
                reward:   dataObj.rewards  ? dataObj.rewards[i]  : null,
                coverage: dataObj.coverage ? dataObj.coverage[i] : null,
            }));
        };

        const rollingAvg = (data, field, window) => {
            if (!data || data.length === 0) return [];
            return data.map((d, i) => {
                const start = Math.max(0, i - window + 1);
                const win = data.slice(start, i + 1);
                const avg = win.reduce((s, x) => s + (x[field] || 0), 0) / win.length;
                return [d.episode, Math.round(avg * 10) / 10];
            });
        };

        // Phase 1 only: first half of raw worker episodes (~25k)
        const hitmacData = (() => {
            const raw = processSeries(jsonData.hitmac_scenario1_full || jsonData.hitmac_scenario1);
            if (raw.length === 0) return raw;
            const phase2Start = (jsonData.metadata && jsonData.metadata.hitmac_phase2_start != null)
                ? jsonData.metadata.hitmac_phase2_start
                : Math.round((Math.max(...raw.map(d => d.episode)) + 1) / 2);
            return raw.filter(d => d.episode < phase2Start);
        })();
        const hitmacVisible = hitmacData.length > 0;

        // Cap NA2Q to same episode count as HiT-MAC Phase 1 for fair comparison
        const na2qData = processSeries(rawData).slice(0, hitmacData.length);

        const phase2PlotLines = [];


        // x-axis spans both datasets
        const allEps = [...na2qData.map(d => d.episode), ...hitmacData.map(d => d.episode)];
        const xMax = allEps.length > 0 ? Math.max(...allEps) : 50000;

        // Update total episodes — actual max episode in the capped comparison window
        const epCount = na2qData.length > 0
            ? Math.max(...na2qData.map(d => d.episode)).toLocaleString()
            : '—';
        const el = document.getElementById('total-episodes');
        if (el) el.textContent = epCount;
        const el2 = document.getElementById('na2q-ep-text');
        if (el2) el2.textContent = epCount;

        /**
         * STATS COMPARISON CARD
         */
        const populateStats = (data, prefix) => {
            if (!data || data.length === 0) return;
            const cov = data.map(d => d.coverage || 0);
            const n = cov.length;
            const tail = cov.slice(Math.max(0, n - 1000));
            const mean = tail.reduce((s, v) => s + v, 0) / tail.length;
            const best = Math.max(...cov);
            const allMean = cov.reduce((s, v) => s + v, 0) / n;
            const std = Math.sqrt(cov.reduce((s, v) => s + (v - allMean) ** 2, 0) / n);
            let conv = null;
            for (let i = 99; i < n; i++) {
                const win = cov.slice(i - 99, i + 1);
                if (win.reduce((s, v) => s + v, 0) / win.length >= 50) { conv = data[i].episode; break; }
            }
            const finalSlice = cov.slice(Math.max(0, n - 500));
            const finalMean  = finalSlice.reduce((s, v) => s + v, 0) / finalSlice.length;
            const finalStd   = Math.sqrt(finalSlice.reduce((s, v) => s + (v - finalMean) ** 2, 0) / finalSlice.length);
            const set = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
            set(`${prefix}-stat-mean`,   mean.toFixed(1) + '%');
            set(`${prefix}-stat-best`,   best.toFixed(1) + '%');
            set(`${prefix}-stat-std`,    '±' + std.toFixed(1) + '%');
            set(`${prefix}-stat-conv`,   conv ? conv.toLocaleString() : 'N/A');
            set(`${prefix}-stat-stable`, '±' + finalStd.toFixed(1) + '%');
        };
        populateStats(na2qData,   'na2q');
        populateStats(hitmacData, 'hitmac');

        // Populate performance table cells (final 500 episodes, mean ± std)
        const populateTableCell = (data, cellId) => {
            const el = document.getElementById(cellId);
            if (!el || !data || data.length === 0) return;
            const cov = data.map(d => d.coverage || 0);
            const n = cov.length;
            const slice = cov.slice(Math.max(0, n - 500));
            const mean = slice.reduce((s, v) => s + v, 0) / slice.length;
            const std  = Math.sqrt(slice.reduce((s, v) => s + (v - mean) ** 2, 0) / slice.length);
            el.innerHTML = `<strong>${mean.toFixed(1)}</strong> <span class="std">± ${std.toFixed(1)}</span>`;
        };
        populateTableCell(na2qData,   'table-na2q-cov');
        populateTableCell(hitmacData, 'table-hitmac-cov');

        // -----------------------------------------------------------------------
        // Series layout (same indices for both charts):
        //   RAW series (visible by default):
        //     0 - NA²Q raw      (thin, faint)
        //     1 - NA²Q trend    (bold, linkedTo 0)
        //     2 - HiT-MAC raw   (thin, faint)
        //     3 - HiT-MAC trend (bold, linkedTo 2)
        //   SMOOTH series (hidden by default):
        //     4 - NA²Q smooth   (bold)
        //     5 - HiT-MAC smooth(bold)
        // Toggle just flips visibility — no data mutation.
        // -----------------------------------------------------------------------

        const makeSeriesSet = (na2qField, hitmacField) => [
            // --- RAW ---
            {
                name: 'NA²Q',
                data: na2qData.map(d => [d.episode, d[na2qField]]),
                lineWidth: 1,
                color: 'rgba(79, 70, 229, 0.35)',
                marker: { enabled: false },
                visible: true,
            },
            {
                name: 'NA²Q trend',
                data: rollingAvg(na2qData, na2qField, 20),
                lineWidth: 2,
                color: '#4f46e5',
                marker: { enabled: false },
                linkedTo: ':previous',
                visible: true,
            },
            {
                name: 'HiT-MAC',
                data: hitmacData.map(d => [d.episode, d[hitmacField]]),
                lineWidth: 1,
                color: 'rgba(217, 119, 6, 0.35)',
                marker: { enabled: false },
                visible: hitmacVisible,
            },
            {
                name: 'HiT-MAC trend',
                data: rollingAvg(hitmacData, hitmacField, 20),
                lineWidth: 2,
                color: '#d97706',
                marker: { enabled: false },
                linkedTo: ':previous',
                visible: hitmacVisible,
            },
            // --- SMOOTH (hidden until toggled) ---
            {
                name: 'NA²Q',
                data: rollingAvg(na2qData, na2qField, 50),
                lineWidth: 2,
                color: '#4f46e5',
                marker: { enabled: false },
                visible: false,
                showInLegend: false,
            },
            {
                name: 'HiT-MAC',
                data: rollingAvg(hitmacData, hitmacField, 50),
                lineWidth: 2,
                color: '#d97706',
                marker: { enabled: false },
                visible: false,
                showInLegend: false,
            },
        ];

        const baseChart = () => ({
            backgroundColor: 'transparent',
            zooming: { type: 'x' },
            animation: { duration: 1200 },
        });

        let rewardChart, coverageChart;

        if (document.getElementById('reward-chart')) {
            rewardChart = Highcharts.chart('reward-chart', {
                chart: { ...baseChart(), type: 'line', height: 420 },
                title: { text: null },
                credits: { enabled: false },
                xAxis: { type: 'linear', title: { text: 'Episode' }, max: xMax, plotLines: phase2PlotLines },
                yAxis: { title: { text: 'Reward' }, gridLineColor: 'rgba(0,0,0,0.05)' },
                tooltip: { shared: true, valueDecimals: 1 },
                plotOptions: { line: { marker: { enabled: false } } },
                legend: { enabled: false },
                series: makeSeriesSet('reward', 'reward'),
            });
        }

        if (document.getElementById('coverage-chart')) {
            coverageChart = Highcharts.chart('coverage-chart', {
                chart: { ...baseChart(), type: 'line', height: 360 },
                title: { text: null },
                credits: { enabled: false },
                xAxis: { allowDecimals: false, title: { text: 'Episode' }, max: xMax, plotLines: phase2PlotLines },
                yAxis: { title: { text: 'Coverage %' }, gridLineColor: 'rgba(0,0,0,0.05)', max: 100 },
                tooltip: { valueSuffix: '%', shared: true },
                plotOptions: { line: { marker: { enabled: false } } },
                legend: { enabled: false },
                series: makeSeriesSet('coverage', 'coverage'),
            });
        }

        // Toggle: flip raw (0-3) vs smooth (4-5) visibility
        // Both toggles (#chart-smooth-toggle and #coverage-smooth-toggle) stay in sync
        const toggleReward   = document.getElementById('chart-smooth-toggle');
        const toggleCoverage = document.getElementById('coverage-smooth-toggle');

        const applySmooth = (isSmooth) => {
            [rewardChart, coverageChart].forEach(chart => {
                if (!chart) return;
                chart.series[0].setVisible(!isSmooth, false);
                chart.series[1].setVisible(!isSmooth, false);
                chart.series[2].setVisible(!isSmooth && hitmacVisible, false);
                chart.series[3].setVisible(!isSmooth && hitmacVisible, false);
                chart.series[4].setVisible(isSmooth, false);
                chart.series[5].setVisible(isSmooth && hitmacVisible, false);
                chart.redraw();
            });
        };

        if (toggleReward) {
            toggleReward.addEventListener('change', () => {
                if (toggleCoverage) toggleCoverage.checked = toggleReward.checked;
                applySmooth(toggleReward.checked);
            });
        }

        if (toggleCoverage) {
            toggleCoverage.addEventListener('change', () => {
                if (toggleReward) toggleReward.checked = toggleCoverage.checked;
                applySmooth(toggleCoverage.checked);
            });
        }
    };

    if (window.trainingData) {
        initCharts();
    } else {
        window.addEventListener('load', initCharts);
    }
});
