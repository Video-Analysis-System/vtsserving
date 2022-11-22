import vtsserving
from vtsserving.io import JSON
from vtsserving.io import Text

vts_model = vtsserving.sklearn.get("20_news_group:latest")

target_names = vts_model.custom_objects["target_names"]
model_runner = vts_model.to_runner()

svc = vtsserving.Service("doc_classifier", runners=[model_runner])


@svc.api(input=Text(), output=JSON())
async def predict(input_doc: str):
    predictions = await model_runner.predict.async_run([input_doc])
    return {"result": target_names[predictions[0]]}


@svc.api(input=Text(), output=JSON())
async def predict_proba(input_doc: str):
    predictions = await model_runner.predict_proba.run([input_doc])
    return predictions[0]
