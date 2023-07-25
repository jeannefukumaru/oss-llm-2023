# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # HuggingFace transformers pipelines in MLflow 
# MAGIC
# MAGIC HuggingFace (HF) pipelines are a popular and accessible way of using models for inference. They abstract complex code from the lower-level HF libraries, thereby allowing developers to focus on the task at hand, whether it's translation, question-answering or text generation. 
# MAGIC
# MAGIC MLflow now offers native support for transformers via an `mlflow.transformers` flavour. With this new flavor, you can save or log a fully configured transformers pipeline or base model via the common MLflow tracking interface. 
# MAGIC
# MAGIC *"The native support of Hugging Face transformers library in MLflow makes it easy to work with over 170,000 free and publicly accessible machine learning models available on the Hugging Face Hub, the largest community and open platform for AI." - Jeff Boudier, Product Director, Hugging Face*
# MAGIC
# MAGIC **In this lab, we will go through:**
# MAGIC - How to log a transformers pipeline in MLflow for easy tracking and lifecycle management
# MAGIC - How to simplify validating your pipeline before deployment by using `mlflow` signatures. 
# MAGIC - How to to the MLflow UI to get critical information about a transformers model, such as its model card, dependencies and other reference information. 
# MAGIC

# COMMAND ----------

# MAGIC %pip install mlflow==2.5.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Load a transformers pipeline for translation via the transformers library

# COMMAND ----------

import transformers
import mlflow

translation_pipeline = transformers.pipeline(
    task="translation_en_to_fr",
    model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
    tokenizer=transformers.T5TokenizerFast.from_pretrained("t5-small", model_max_length=100),
)


# COMMAND ----------

# MAGIC %md 
# MAGIC # Create a model signature for the pipeline
# MAGIC
# MAGIC MLflow model signatures are a description of a model's expected inputs and outputs. This information is useful for downstream tooling - for example, deployment tools use model signatures to validate that the inputs passed to a model for inference are correct
# MAGIC
# MAGIC The model signature is stored in JSON format in the MLmodel file in your model artifacts, together with other model metadata. 
# MAGIC
# MAGIC <img src="https://github.com/jeannefukumaru/oss-llm-2023/raw/main/images/model_metadata.png">

# COMMAND ----------

signature = mlflow.models.infer_signature(
    "Hi there, chatbot!",
    mlflow.transformers.generate_signature_output(translation_pipeline, "Hi there, chatbot!"),
)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Log the translation pipeline to `mlflow`

# COMMAND ----------

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=translation_pipeline,
        artifact_path="french_translator",
        signature=signature,
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC For each of the pipeline types supported by the transformers package, metadata is collected to ensure that the exact requirements, versions of components, and reference information is available for both future reference and for serving of the saved model or pipeline. 
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC Additionally, the MLflow transformers flavor will automatically pull the state of the Model Card from the Hugging Face Hub upon saving or logging of a model or pipeline. This feature allows for a point-in-time reference of the state of the underlying model information for both general reference and auditing purposes.
# MAGIC
# MAGIC <img src="https://github.com/jeannefukumaru/oss-llm-2023/raw/main/images/model_card.png">

# COMMAND ----------

# MAGIC %md 
# MAGIC After logging our pipeline artifacts, these logged artifacts can be loaded natively as either a collection of components, a pipeline, or via pyfunc.
# MAGIC
# MAGIC In the cell below, we load our model and list out the available components inside it. 

# COMMAND ----------

translation_components = mlflow.transformers.load_model(
    model_info.model_uri, return_type="components"
)

for key, value in translation_components.items():
    print(f"{key} -> {type(value).__name__}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Predictions

# COMMAND ----------

response = translation_pipeline("MLflow is great!")

print(response)

# COMMAND ----------

reconstructed_pipeline = transformers.pipeline(**translation_components)

reconstructed_response = reconstructed_pipeline(
    "transformers makes using Deep Learning models easy and fun!"
)

print(reconstructed_response)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Further references 
# MAGIC HuggingFace transformers are not the only LLM model type that MLflow supports. We also support Open AI functions and Langchain. For more information, see this [blog post](https://www.databricks.com/blog/2023/04/18/introducing-mlflow-23-enhanced-native-llm-support-and-new-features.html) 
