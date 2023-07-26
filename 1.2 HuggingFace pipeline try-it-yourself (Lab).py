# Databricks notebook source
# MAGIC %md 
# MAGIC # HuggingFace transformers pipelines in MLflow
# MAGIC ### Lab: Text Generation

# COMMAND ----------

# MAGIC %pip install mlflow==2.5.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Load a transformers pipeline for text generation via the transformers library

# COMMAND ----------

import transformers
import mlflow

# <TODO: input the HuggingFace pipeline task name for text generation here>
task = "YOUR CODE HERE"

generation_pipeline = transformers.pipeline(
    task=task,
    model="declare-lab/flan-alpaca-base",
)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Create a model signature and input examples for the pipeline
# MAGIC
# MAGIC MLflow model signatures are a description of a model's expected inputs and outputs. This information is useful for downstream tooling - for example, deployment tools use model signatures to validate that the inputs passed to a model for inference are correct
# MAGIC
# MAGIC The model signature is stored in JSON format in the MLmodel file in your model artifacts, together with other model metadata. 
# MAGIC
# MAGIC <img src="https://github.com/jeannefukumaru/oss-llm-2023/raw/main/images/model_metadata.png">

# COMMAND ----------

# <TODO: provide a few input examples>
input_example = ["EXAMPLE", "EXAMPLE HERE", "EXAMPLE HERE"]

parameters = {"max_length": 512, "do_sample": True}

# COMMAND ----------

signature = mlflow.models.infer_signature(
    input_example,
    mlflow.transformers.generate_signature_output(generation_pipeline, input_example, parameters),
)

# COMMAND ----------

# <TODO: fill in the missing parameters> 
with mlflow.start_run() as run:
    model_info = mlflow.transformers.log_model(
        transformers_model="",  # YOUR HF PIPELINE HERE
        artifact_path="text_generator",
        input_example="",  # WHAT GOES HERE? 
        signature="",  # WHAT GOES HERE? 
        inference_config=parameters
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC # Predictions

# COMMAND ----------

sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)

print(
    sentence_generator.predict(
        ["tell me a story about rocks", "Tell me a joke about a dog that likes spaghetti"],
    )
)

# COMMAND ----------

# TODO: test with your own prompt
print(
    sentence_generator.predict(
        ["YOUR PROMPT HERE"],
    )
)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Further references 
# MAGIC HuggingFace transformers are not the only LLM model type that MLflow supports. We also support Open AI functions and Langchain. For more information, see this [blog post](https://www.databricks.com/blog/2023/04/18/introducing-mlflow-23-enhanced-native-llm-support-and-new-features.html) 
