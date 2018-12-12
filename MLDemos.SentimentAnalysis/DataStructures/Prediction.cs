using Microsoft.ML.Runtime.Api;

namespace MLDemos.SentimentAnalysis.DataStructures
{
    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public bool Value { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }
}
