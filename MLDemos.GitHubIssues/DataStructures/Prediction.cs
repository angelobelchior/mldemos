using Microsoft.ML.Runtime.Api;

namespace MLDemos.GitHubIssues.DataStructures
{
    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;
    }
}
