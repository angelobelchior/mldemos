using System;
using System.IO;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Core.Data;
using Microsoft.ML;

using Microsoft.ML.Transforms.Text;

namespace MLDemos.SentimentAnalysis
{
    class Program
    {
        private const string DATA_FILE_PATH = @"C:\git\MLDemos\MLDemos\Datasets\wikipedia-detox-250-line-data.tsv";
        private const string TEST_FILE_PATH = @"C:\git\MLDemos\MLDemos\Datasets\wikipedia-detox-250-line-test.tsv";
        private const string MODEL_FILE_PATH = @"C:\git\MLDemos\MLDemos\TrainedModels\SentimentModel.zip";

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);
            var trainedModel = Train(mlContext);

            Test(mlContext, trainedModel);

            Predict(mlContext, "This is a very rude movie");

            Console.ReadLine();
        }

        private static ITransformer Train(MLContext mlContext)
        {
            var trainingDataView = ReadData(mlContext, DATA_FILE_PATH);

            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Text", DefaultColumnNames.Features);

            var trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumn: DefaultColumnNames.Label, featureColumn: DefaultColumnNames.Features);
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            using (var fs = new FileStream(MODEL_FILE_PATH, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainedModel, fs);

            return trainedModel;
        }

        private static void Test(MLContext mlContext, ITransformer trainedModel)
        {
            var testDataView = ReadData(mlContext, TEST_FILE_PATH);
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label", "Score");
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*            Auc: {metrics.Auc:P2}");
            Console.WriteLine($"*        F1Score: {metrics.F1Score:P2}");
            Console.WriteLine($"************************************************************");
        }

        private static void Predict(MLContext mlContext, string text, ITransformer trainedModel = null)
        {
            var issue = new DataStructures.Issue { Text = text };

            if (trainedModel == null)
            {
                using (var stream = new FileStream(MODEL_FILE_PATH, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    trainedModel = mlContext.Model.Load(stream);
                }
            }

            var predFunction = trainedModel.MakePredictionFunction<DataStructures.Issue, DataStructures.Prediction>(mlContext);
            var resultprediction = predFunction.Predict(issue);
            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"Text: {issue.Text}");
            Console.WriteLine($"Prediction: {(Convert.ToBoolean(resultprediction.Value) ? "Toxic" : "Nice")} sentiment | Probability: {resultprediction.Probability} ");
            Console.WriteLine($"==================================================");
        }

        private static IDataView ReadData(MLContext mlContext, string filePath)
        {
            var textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Bool, 0),
                    new TextLoader.Column("Text", DataKind.Text, 1)
                }
            });
            var dataview = textLoader.Read(filePath);
            return dataview;
        }
    }
}
