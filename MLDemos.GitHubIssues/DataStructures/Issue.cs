using Microsoft.ML.Runtime.Api;

namespace MLDemos.GitHubIssues.DataStructures
{
    public class Issue
    {
        [Column(ordinal: "0")]
        public string ID;

        [Column(ordinal: "1")]
        public string Area; 

        [Column(ordinal: "2")]
        public string Title;

        [Column(ordinal: "3")]
        public string Description;
    }
}
