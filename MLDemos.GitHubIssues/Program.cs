using System;
using System.IO;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Runtime.Data;

namespace MLDemos.GitHubIssues
{
    class Program
    {
        private const string DATA_FILE_PATH = @"C:\git\MLDemos\MLDemos\Datasets\corefx-issues-train-data.tsv";
        private const string TEST_FILE_PATH = @"C:\git\MLDemos\MLDemos\Datasets\corefx-issues-train-test.tsv";
        private const string MODEL_FILE_PATH = @"C:\git\MLDemos\MLDemos\TrainedModels\GitHubIssues.zip";

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);
            //r trainedModel = Train(mlContext);
            //st(mlContext, trainedModel);
            //Predict(mlContext, "WebSockets communication is slow in my machine", "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine..");
            Predict(mlContext, "Stack overflow", "There's a memory error");
            Console.ReadLine();
        }

        private static ITransformer Train(MLContext mlContext)
        {
            var dataview = ReadData(mlContext, DATA_FILE_PATH);
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Area", DefaultColumnNames.Label)
                            .Append(mlContext.Transforms.Text.FeaturizeText("Title", "TitleFeaturized"))
                            .Append(mlContext.Transforms.Text.FeaturizeText("Description", "DescriptionFeaturized"))
                            .Append(mlContext.Transforms.Concatenate(DefaultColumnNames.Features, "TitleFeaturized", "DescriptionFeaturized"))
                            .AppendCacheCheckpoint(mlContext);
            ;
            var averagedPerceptron = mlContext.BinaryClassification.Trainers.AveragedPerceptron(DefaultColumnNames.Label,
                                                                                                DefaultColumnNames.Features,
                                                                                                numIterations: 10);
            var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptron);

            var trainingPipeline = pipeline.Append(trainer).Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            var trainedModel = trainingPipeline.Fit(dataview);

            using (var fs = new FileStream(MODEL_FILE_PATH, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainedModel, fs);

            return trainedModel;
        }

        private static void Test(MLContext mlContext, ITransformer trainedModel)
        {
            var testDataView = ReadData(mlContext, TEST_FILE_PATH);
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, DefaultColumnNames.Label, DefaultColumnNames.Score);
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy Macro: {metrics.AccuracyMacro:P2}");
            Console.WriteLine($"*       Accuracy Macro: {metrics.AccuracyMicro:P2}");
            Console.WriteLine($"*             Log Loss: {metrics.LogLoss:P2}");
            Console.WriteLine($"*                Top K: {metrics.TopK:P2}");
            Console.WriteLine($"*       Top K Accuracy: {metrics.TopKAccuracy:P2}");
            Console.WriteLine($"************************************************************");
        }

        private static IDataView ReadData(MLContext mlContext, string filePath)
        {
            var textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("ID", DataKind.Text, 0),
                    new TextLoader.Column("Area", DataKind.Text, 1),
                    new TextLoader.Column("Title", DataKind.Text, 2),
                    new TextLoader.Column("Description", DataKind.Text, 3),
                }
            });

            var dataview = textLoader.Read(filePath);
            return dataview;
        }

        private static void Predict(MLContext mlContext, string title, string description, ITransformer trainedModel = null)
        {
            var issue = new DataStructures.Issue { ID = "ID", Title = title, Description = description };

            if (trainedModel == null)
            {
                using (var stream = new FileStream(MODEL_FILE_PATH, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    trainedModel = mlContext.Model.Load(stream);
                }
            }

            var predFunction = trainedModel.MakePredictionFunction<DataStructures.Issue, DataStructures.Prediction>(mlContext);
            var prediction = predFunction.Predict(issue);

            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"Title: {issue.Title}");
            Console.WriteLine($"Description: {issue.Description}");
            Console.WriteLine($"Prediction: {prediction.Area}");
            Console.WriteLine($"==================================================");
        }
    }
}
