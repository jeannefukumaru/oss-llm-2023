# Databricks notebook source
# MAGIC %pip install mlflow==2.5.0 evaluate==0.4.0 nltk rouge_score
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Track your evaluation datasets to ensure accurate comparisons

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
# MAGIC ## HuggingFace pipeline

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

# COMMAND ----------


