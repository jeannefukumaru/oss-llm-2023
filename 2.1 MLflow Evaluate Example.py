# Databricks notebook source
# MAGIC %md 
# MAGIC # MLflow Tools for robustly evaluating LLM models
# MAGIC
# MAGIC As outlined in a [Databricks blog post](https://www.databricks.com/blog/announcing-mlflow-24-llmops-tools-robust-model-evaluation), "With model quality challenges like hallucination, response toxicity, and vulnerability to prompt injection, as well a lack of ground truth labels for many tasks, data scientists need to be extremely diligent about evaluating their models’ performance on a wide variety of data."
# MAGIC
# MAGIC To meet these needs, version MLflow 2.5.0 onwards provides a comprehensive set of LLMOps tools for model evaluation. 
# MAGIC
# MAGIC **In this lab, we will give a walkthrough of these tools.**
# MAGIC
# MAGIC By the end of the lab, you will be able to: 
# MAGIC - Track your evaluation datasets to ensure accurate comparisons
# MAGIC - Capture performance insights with mlflow.evaluate() for language models
# MAGIC - Inspect and compare LLM outputs with the new Artifact View

# COMMAND ----------

# MAGIC %pip install mlflow==2.5.0 evaluate==0.4.0 nltk rouge_score
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Track your evaluation datasets to ensure accurate comparisons
# MAGIC
# MAGIC Selecting a model with the best reported accuracy only makes sense if every model considered was evaluated on the same dataset.
# MAGIC
# MAGIC With MLflow Dataset Tracking, you can standardize the way you manage and analyze datasets during model development. With Dataset Tracking, you can quickly identify which datasets were used to develop and evaluate each of your models, ensuring fair comparison and simplifying model selection for production deployment.
# MAGIC
# MAGIC For this example, we will use the CNN Daily Mail dataset available from HuggingFace. `mlflow.data` accepts datasets as a Pandas DataFrame, so we will take a subset of the test dataset, convert it to a Pandas DataFrame, and register it as an `mlflow.data` object
# MAGIC
# MAGIC Later, we will use the `mlflow.log_input()` command to associate this evaluation dataset to a specific evaluation experiment run

# COMMAND ----------

from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# COMMAND ----------

test_pd = dataset["test"].to_pandas().iloc[0:5, :].drop(["id"], axis=1)
test_pd

# COMMAND ----------

mlflow_pd_test_data = mlflow.data.from_pandas(test_pd)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## HuggingFace pipelines
# MAGIC As with the previous lab, we will instantiate a HuggingFace pipeline relevant to our use case (in this case summarization) that we will log later using `mlflow.log_model()`
# MAGIC
# MAGIC We will compare two summarization pipelines, one using the `t5-small` model, and one using the `bart-large-cnn` model. We will log these models, their datasets, evaluation metrics and summarization outputs inside an MLflow Experiment for easy comparison later. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create our `t5-small` summarization pipeline

# COMMAND ----------

from transformers import pipeline
import mlflow

t5_summarizer = pipeline("summarization", model="t5-small", min_length=5, max_length=20)

text_to_summarize = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

t5_summarizer(text_to_summarize, min_length=5, max_length=20)

# COMMAND ----------

input_example = text_to_summarize

signature = mlflow.models.infer_signature(
    input_example,
    mlflow.transformers.generate_signature_output(t5_summarizer, input_example),
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Log the `t5-pipeline` model, the evaluation dataset, evaluation metrics and summarization outputs inside an MLflow Experiment
# MAGIC
# MAGIC This is where most of the heavy lifting happens. 
# MAGIC
# MAGIC - we log our dataset using `mlflow.log_input()`
# MAGIC - log our transformers pipeline using `mlflow.transformers.log_model()`
# MAGIC - evaluate our model on our test dataframe
# MAGIC - check that our desired evaluation metrics have been computed

# COMMAND ----------

with mlflow.start_run(run_name="t5-small") as run:
  
    mlflow.log_input(mlflow_pd_test_data, context="test")

    model_info = mlflow.transformers.log_model(
        transformers_model=summarizer,
        artifact_path="text_summarization",
        input_example=text_to_summarize,
        signature=signature,
    )

    evaluation_results = mlflow.evaluate(
    f"runs:/{model_info.run_id}/text_summarization",
    data=test_pd,
    model_type="text-summarization",
    targets="highlights",
    feature_names=["article"]
    )

    # Verify that ROUGE metrics are automatically computed for summarization
    assert "rouge1" in evaluation_results.metrics
    assert "rouge2" in evaluation_results.metrics

    # Verify that inputs and outputs are captured as a table for further analysis
    assert "eval_results_table" in evaluation_results.artifacts

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Log the `bart_large_cnn` model pipeline, the evaluation dataset, evaluation metrics and summarization outputs inside an MLflow Experiment

# COMMAND ----------

bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", min_length=5, max_length=20)

text_to_summarize = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

bart_summarizer(text_to_summarize, min_length=5, max_length=20)

# COMMAND ----------

with mlflow.start_run(run_name="bart-large-cnn") as run:
    mlflow.log_input(mlflow_pd_test_data, context="test")

    model_info = mlflow.transformers.log_model(
        transformers_model=bart_summarizer,
        artifact_path="text_summarization",
        input_example=text_to_summarize,
        signature=signature,
    )

    evaluation_results = mlflow.evaluate(
    f"runs:/{model_info.run_id}/text_summarization",
    data=test_pd,
    model_type="text-summarization",
    targets="highlights",
    feature_names=["article"]
    )

    # Verify that ROUGE metrics are automatically computed for summarization
    assert "rouge1" in evaluation_results.metrics
    assert "rouge2" in evaluation_results.metrics

    # Verify that inputs and outputs are captured as a table for further analysis
    assert "eval_results_table" in evaluation_results.artifacts

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Inspect and compare LLM outputs with the new Artifact View
# MAGIC
# MAGIC Without ground truth labels, many LLM developers need to manually inspect model outputs to assess quality.
# MAGIC
# MAGIC With the Evaluation tab inside the MLflow Experiment UI, we can see how each model summarizes a given document and identify differences.
# MAGIC
# MAGIC <img src="https://github.com/jeannefukumaru/govtech-llm-2023/raw/main/images/artifact_view">

# COMMAND ----------


