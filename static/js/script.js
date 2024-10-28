document.addEventListener('DOMContentLoaded', function(){
    const trace1 = {
        x: [1, 2, 3, 4],
        y: [10, 15, 13, 17],
        mode: 'markers',
        type: 'scatter'
    };

    const data = [trace1];

    Plotly.newPlot('graph', data);
});