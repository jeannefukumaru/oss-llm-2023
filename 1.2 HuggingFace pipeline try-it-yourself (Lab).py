# Databricks notebook source
# MAGIC %md 
# MAGIC # HuggingFace transformers pipelines in MLflow
# MAGIC ### Text Generation example (Lab)

# COMMAND ----------

# MAGIC %pip install mlflow==2.5.0
# MAGIC dbutils.library.restartPython()

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
