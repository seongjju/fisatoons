let ratingChart = new Chart(ctx, {
    type: "line",
    data: {
        labels: episodeNumbers,
        datasets: [
            {
                label: "실제 별점",
                borderColor: "#ff4757", /* 네온 레드 */
                backgroundColor: "rgba(255, 71, 87, 0.2)",
                data: actualRatings,
                fill: false
            },
            {
                label: "모델 1 예측 별점",
                borderColor: "#1e90ff", /* 네온 블루 */
                backgroundColor: "rgba(30, 144, 255, 0.2)",
                data: model1Predictions,
                fill: false
            },
            {
                label: "모델 2 예측 별점",
                borderColor: "#2ed573", /* 네온 그린 */
                backgroundColor: "rgba(46, 213, 115, 0.2)",
                data: model2Predictions,
                fill: false
            },
            {
                label: "모델 3 예측 별점",
                borderColor: "#ffa502", /* 네온 오렌지 */
                backgroundColor: "rgba(255, 165, 2, 0.2)",
                data: model3Predictions,
                fill: false
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: false,
                suggestedMin: 5,
                suggestedMax: 10,
                grid: { color: "#444" } /* 눈금선 다크 스타일 */
            },
            x: {
                grid: { color: "#444" }
            }
        }
    }
});