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
                episode:       ep,
                reward:        dataObj.rewards       ? dataObj.rewards[i]       : null,
                coverage:      dataObj.coverage      ? dataObj.coverage[i]      : null,
                coverage_std:  dataObj.coverage_std  ? dataObj.coverage_std[i]  : null,
                rewards_std:   dataObj.rewards_std   ? dataObj.rewards_std[i]   : null,
            }));
        };

        const hasStd = (data, field) => data.length > 0 && data[0][field] != null;

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

        // COMA and QPLEX — render when data is available
        const comaData  = processSeries(jsonData.coma_scenario1  || {});
        const qplexData = processSeries(jsonData.qplex_scenario1 || {});

        const phase2PlotLines = [];

        // x-axis spans all datasets
        const allEps = [
            ...na2qData.map(d => d.episode),
            ...hitmacData.map(d => d.episode),
            ...comaData.map(d => d.episode),
            ...qplexData.map(d => d.episode),
        ];
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
        populateStats(comaData,   'coma');
        populateStats(qplexData,  'qplex');

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
        populateTableCell(comaData,   'table-coma-cov');
        populateTableCell(qplexData,  'table-qplex-cov');

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

        const makeSeriesSet = (na2qField, hitmacField) => {
            const stdField = na2qField === 'coverage' ? 'coverage_std' : 'rewards_std';
            const na2qHasStd   = hasStd(na2qData, stdField);
            const hitmacHasStd = false; // show HiT-MAC as clean mean line, not a dense std band

            const isCov = na2qField === 'coverage';
            const bandSeries = (data, name, color, stdF, visible) => [
                {
                    name,
                    type: 'arearange',
                    data: data.map(d => {
                        const v = d[isCov ? 'coverage' : 'reward'];
                        const s = d[stdF] || 0;
                        const lo = Math.round((v - s) * 10) / 10;
                        const hi = Math.round((v + s) * 10) / 10;
                        return [d.episode,
                            isCov ? Math.max(0,   lo) : lo,
                            isCov ? Math.min(100, hi) : hi,
                        ];
                    }),
                    color, fillOpacity: 0.12, lineWidth: 0,
                    marker: { enabled: false }, enableMouseTracking: false,
                    showInLegend: false, visible,
                },
                {
                    name,
                    data: data.map(d => [d.episode, d[na2qField === 'coverage' ? 'coverage' : 'reward']]),
                    lineWidth: 2, color, marker: { enabled: false }, visible,
                },
            ];

            const rawSeries = (data, name, color, field, visible) => [
                {
                    name, data: data.map(d => [d.episode, d[field]]),
                    lineWidth: 1, color: color.replace(')', ', 0.35)').replace('rgb', 'rgba'),
                    marker: { enabled: false }, visible,
                },
                {
                    name: name + ' trend',
                    data: rollingAvg(data, field, 20),
                    lineWidth: 2, color, marker: { enabled: false }, visible,
                },
            ];

            const na2qSeries   = na2qHasStd
                ? bandSeries(na2qData,   'NA²Q',   '#4f46e5', stdField, true)
                : rawSeries(na2qData,    'NA²Q',   'rgb(79, 70, 229)', na2qField, true);

            const hitmacSeries = hitmacHasStd
                ? bandSeries(hitmacData, 'HiT-MAC', '#d97706', stdField, hitmacVisible)
                : rawSeries(hitmacData,  'HiT-MAC', 'rgb(217, 119, 6)', hitmacField, hitmacVisible);

            const comaSeries  = comaData.length  > 0
                ? rawSeries(comaData,  'COMA',  'rgb(22, 163, 74)',  na2qField, true) : [];
            const qplexSeries = qplexData.length > 0
                ? rawSeries(qplexData, 'QPLEX', 'rgb(220, 38, 38)', na2qField, true) : [];

            return [...na2qSeries, ...hitmacSeries, ...comaSeries, ...qplexSeries];
        };

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
                yAxis: { title: { text: 'Coverage %' }, gridLineColor: 'rgba(0,0,0,0.05)', min: 0, max: 100 },
                tooltip: { valueSuffix: '%', shared: true },
                plotOptions: { line: { marker: { enabled: false } } },
                legend: { enabled: false },
                series: makeSeriesSet('coverage', 'coverage'),
            });
        }

        // Toggle: Raw = faint raw line + bold trend; Smooth = trend only
        const applySmooth = (isSmooth, chart) => {
            if (!chart) return;
            chart.series.forEach(s => {
                const isTrend = s.name.endsWith(' trend');
                s.setVisible(isTrend ? true : !isSmooth, false);
            });
            chart.redraw();
        };

        const syncToggle = (checked) => {
            applySmooth(checked, rewardChart);
            applySmooth(checked, coverageChart);
            document.getElementById('chart-smooth-toggle').checked   = checked;
            document.getElementById('coverage-smooth-toggle').checked = checked;
        };

        document.getElementById('chart-smooth-toggle')   ?.addEventListener('change', e => syncToggle(e.target.checked));
        document.getElementById('coverage-smooth-toggle') ?.addEventListener('change', e => syncToggle(e.target.checked));
    };

    if (window.trainingData) {
        initCharts();
    } else {
        window.addEventListener('load', initCharts);
    }

    // ── Scenario 2 charts ──────────────────────────────────────────────────
    let s2Chart = null;

    const buildS2Chart = () => {
        const container = document.getElementById('s2-coverage-chart');
        const s2 = window.trainingDataS2;
        if (!container || !s2) return;

        const smooth = (data, w) => data.map((_, i) => {
            const sl = data.slice(Math.max(0, i - w + 1), i + 1);
            return [data[i][0], Math.round(sl.reduce((s, p) => s + p[1], 0) / sl.length * 10) / 10];
        });

        const makePts = (key) => {
            const d = s2[key];
            if (!d || !d.episodes || d.episodes.length === 0) return null;
            return d.episodes.map((ep, i) => [ep, d.coverage[i]]);
        };

        const hitmacPts = makePts('hitmac_scenario2');
        const na2qPts   = makePts('na2q_scenario2');

        const w = container.closest('#scenario-2-content')
            ? (document.querySelector('.main-container') || document.body).offsetWidth - 80
            : container.offsetWidth;

        const series = [];
        if (hitmacPts) {
            series.push({ name: 'HiT-MAC raw', data: hitmacPts,             lineWidth: 1,   color: 'rgba(217,119,6,0.25)',  marker: { enabled: false }, showInLegend: false });
            series.push({ name: 'HiT-MAC',     data: smooth(hitmacPts, 50), lineWidth: 2.5, color: '#d97706',               marker: { enabled: false } });
        }
        if (na2qPts) {
            series.push({ name: 'NA²Q raw', data: na2qPts,             lineWidth: 1,   color: 'rgba(79,70,229,0.25)',  marker: { enabled: false }, showInLegend: false });
            series.push({ name: 'NA²Q',     data: smooth(na2qPts, 50), lineWidth: 2.5, color: '#4f46e5',               marker: { enabled: false } });
        }

        s2Chart = Highcharts.chart(container, {
            chart: { backgroundColor: 'transparent', type: 'line',
                     width: w > 100 ? w : 900, height: 360,
                     zooming: { type: 'x' }, animation: { duration: 800 } },
            title:   { text: null },
            credits: { enabled: false },
            xAxis: { title: { text: 'Episode' } },
            yAxis: {
                title: { text: 'Coverage %' },
                gridLineColor: 'rgba(128,128,128,0.15)',
                min: 0, max: 100,
            },
            tooltip: { valueSuffix: '%', shared: true },
            plotOptions: { line: { marker: { enabled: false } } },
            legend: { enabled: false },
            series,
        });
    };

    window.initS2Charts = () => {
        if (s2Chart) { s2Chart.reflow(); return; }
        buildS2Chart();
    };

    // Pre-render on load (hidden container — uses fallback width), reflow on tab click
    setTimeout(buildS2Chart, 300);
});
