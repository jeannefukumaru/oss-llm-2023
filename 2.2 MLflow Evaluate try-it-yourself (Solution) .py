# Databricks notebook source
# MAGIC %md 
# MAGIC # MLflow Tools for robustly evaluating LLM models
# MAGIC
# MAGIC **Lab: Text Generation**
# MAGIC
# MAGIC We covered summarization in the previous notebook. Now, for this lab, we are going to apply the concepts covered to text generation. Additionally, we will also show a variation on how to use  `mlflow.evaluate` to also take into account different model configurations
# MAGIC
# MAGIC **In this lab, we will give a walkthrough of these tools.**
# MAGIC
# MAGIC By the end of the lab, you will be able to: 
# MAGIC - Track your evaluation datasets to ensure accurate comparisons
# MAGIC - Capture performance insights with mlflow.evaluate() for language models
# MAGIC - Inspect and compare LLM outputs with the new Artifact View

# COMMAND ----------

# MAGIC %pip install mlflow==2.5.0 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Track your evaluation datasets to ensure accurate comparisons
# MAGIC
# MAGIC Selecting a model with the best reported accuracy only makes sense if every model considered was evaluated on the same dataset.
# MAGIC
# MAGIC With MLflow Dataset Tracking, you can standardize the way you manage and analyze datasets during model development. With Dataset Tracking, you can quickly identify which datasets were used to develop and evaluate each of your models, ensuring fair comparison and simplifying model selection for production deployment.
# MAGIC
# MAGIC Later, we will use the `mlflow.log_input()` command to associate this evaluation dataset to a specific evaluation experiment run

# COMMAND ----------

import pandas as pd 
eval_df = pd.DataFrame(
  {
    "question": [
      "Q: Does the spin of the earth have any significant effect on the time it takes to complete a trans-pacific flight vs a trans-atlantic flight?\nA:",
      "Q: What would the evolutionary benefits be for male mammals' testicles being located in such a vulnerable location instead of being inside the body?\nA:",
      "Q: Why do resistor values need to be color-coded, unlike capacitors wherein the capacitance is already printed on it?\nA:",
      "Q: Spacetime can be streched and bend. But can it vibrate? If so, does it has resonance frequency?\nA:",
      "Q: Is their a psychological or physical benefit to tennis players who moan, groan or scream every time they hit the ball?\nA:"

    ]
  }
)

# COMMAND ----------

mlflow_pd_test_data = mlflow.data.from_pandas(eval_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## HuggingFace pipelines
# MAGIC As with the previous lab, we will instantiate a HuggingFace pipeline relevant to our use case (in this case question-answering) that we will log later using `mlflow.log_model()`
# MAGIC
# MAGIC We will first set-up a text generation pipeline, one using the `distilgpt` modell. We will log this model, its datasets, and outputs inside an MLflow Experiment for easy comparison later. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create our `distigpt` text generation pipeline

# COMMAND ----------

from transformers import pipeline
import mlflow

distilgpt2_pipe = pipeline("text-generation", model="distilgpt2")

text = "Somatic hypermutation allows the immune system to"

distilgpt2_pipe(text)

# COMMAND ----------

input_example = ["Somatic hypermutation allows the immune system to"]

signature = mlflow.models.infer_signature(
    input_example,
    mlflow.transformers.generate_signature_output(distilgpt2_pipe, input_example),
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Log the `distilbert-pipe` pipeline, the evaluation dataset, evaluation metrics and summarization outputs inside an MLflow Experiment
# MAGIC
# MAGIC This is where most of the heavy lifting happens. 
# MAGIC
# MAGIC - we log our dataset using `mlflow.log_input()`
# MAGIC - log our transformers pipeline using `mlflow.transformers.log_model()`
# MAGIC - evaluate our model on our test dataframe
# MAGIC - check that our desired evaluation metrics have been computed

# COMMAND ----------

with mlflow.start_run(run_name="distilgpt-2") as run:
  
    mlflow.log_input(mlflow_pd_test_data, context="test")

    model_info = mlflow.transformers.log_model(
        transformers_model=distilgpt2_pipe,
        artifact_path="text-gen",
        input_example=input_example,
        signature=signature,
    )

    evaluation_results = mlflow.evaluate(
    f"runs:/{model_info.run_id}/text-gen",
    data=eval_df,
    model_type="text",
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Use `GenerationConfigs` to experiment with different model configurations, and then inspect the resulting output
# MAGIC
# MAGIC Apart from the `mlflow.transformers` flavour, we can also log transformer pipelines inside the more flexible `mlflow.pyfunc` format. Here, we customise the `predict` method of the `mlflow.pyfunc` model to take in different model configurations. 
# MAGIC
# MAGIC More information on `mlflow.pyfunc` models can be found here https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html

# COMMAND ----------

import json
from transformers import GenerationConfig
#example adapted from https://medium.com/@dliden/comparing-llms-with-mlflow-1c69553718df

class PyfuncTransformerWithParams(mlflow.pyfunc.PythonModel):
    """PyfuncTransformer is a class that extends the mlflow.pyfunc.PythonModel class
    and is used to create a custom MLflow model for text generation using Transformers.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__()

    def load_context(self, context):
        self.model = pipeline(
            "text-generation",
            model=self.model_name,
        )

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_dict(orient="records")
        elif not isinstance(model_input, list):
            model_input = [model_input]

        generated_text = []
        for record in model_input:
            input_text = record["input_text"]
            config_dict = record["config_dict"]
            # Update the GenerationConfig attributes with the provided config_dict
            gcfg = GenerationConfig.from_model_config(self.model.model.config)
            for key, value in json.loads(config_dict).items():
                if hasattr(gcfg, key):
                    setattr(gcfg, key, value)

            output = self.model(
                input_text,
                generation_config=gcfg,
            )
            generated_text.append(output[0]["generated_text"])

        return generated_text

# COMMAND ----------

questions = [
      "Q: Does the spin of the earth have any significant effect on the time it takes to complete a trans-pacific flight vs a trans-atlantic flight?\nA:",
      "Q: What would the evolutionary benefits be for male mammals' testicles being located in such a vulnerable location instead of being inside the body?\nA:",
      "Q: Why do resistor values need to be color-coded, unlike capacitors wherein the capacitance is already printed on it?\nA:",
      "Q: Spacetime can be streched and bend. But can it vibrate? If so, does it has resonance frequency?\nA:",
      "Q: Is their a psychological or physical benefit to tennis players who moan, groan or scream every time they hit the ball?\nA:"
    ]

config_dict_1 = {
    "do_sample": True,
    "top_k": 10,
    "max_length": 180,
}

config_dict_2 = {
    "do_sample": False,
    "max_length": 180,
}

config_dicts = [config_dict_1, config_dict_2]

eval_df_2 = pd.DataFrame({
  "input_text": questions * len(config_dicts),
  "config_dict": [
        json.dumps(config)
        for config in config_dicts
        for _ in range(len(questions))
    ],}
)

# COMMAND ----------

mlflow_pd_test_data_2 = mlflow.data.from_pandas(eval_df_2, name="evaluation_configurations")

# COMMAND ----------

with mlflow.start_run(run_name="distilgpt-2-inference-config") as run:
  
    eval_data = mlflow.log_input(mlflow_pd_test_data_2, context="input_data")

    pyfunc_transformers_model = PyfuncTransformerWithParams("distilgpt2")

    input_example_df = pd.DataFrame([
        {
            "input_text": "Q: What color is the sky?\nA:",
            "config_dict": {}, 
        }])

    model_info = mlflow.pyfunc.log_model(
        python_model=pyfunc_transformers_model,
        artifact_path="text-gen",
        input_example=input_example_df
    )

    model_uri = mlflow.pyfunc.load_model(f"runs:/{model_info.run_id}/text-gen")

    evaluation_results = mlflow.evaluate(
      model_uri,
      feature_names=["input_text", "config_dict"],
      data=eval_df_2,
      model_type="text",
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Inspect and compare LLM outputs and configurations with the Artifact View
# MAGIC
# MAGIC With the Evaluation tab inside the MLflow Experiment UI, we can see how each model performs on a given input and configuration, and identify differences. For example, without sampling, the model appears to repeat itself a lot more. 
# MAGIC
# MAGIC <img src="https://github.com/jeannefukumaru/oss-llm-2023/raw/main/images/Screenshot%202023-07-26%20at%205.02.35%20PM.png">
