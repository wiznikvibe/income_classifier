from flask import Flask, render_template, request, jsonify 
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomClass

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def prediction_data():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomClass(
            age= int(request.form.get("age")),
            fnlwgt=int(request.form.get("fnlwgt")),
            workclass= request.form.get("workclass"),
            education=request.form.get("education"),
            education_num=int(request.form.get("education_num")),
            marital_status=request.form.get("marital_status"),
            occupation=request.form.get('occupation'),
            relationship=request.form.get('relationship'),
            race=request.form.get('race'),
            sex=request.form.get('sex'),
            capital_gain=int(request.form.get('capital_gain')),
            capital_loss=int(request.form.get('capital_loss')),
            hours_per_week=int(request.form.get('hours_per_week')),
            native_country=request.form.get('native_country')
        )

    final_data = data.get_data_DataFrame()
    pipeline_prediction = PredictionPipeline()
    pred = pipeline_prediction.predict(final_data)

    result = pred 
    if result == 0:
        return render_template("result.html", final_result = "Your Yearly Income is Less than Equal to 50k: {}".format(result[0]) )
    else:
        return render_template("result.html", final_result = "Your Yearly Income is More than Equal to 50k: {}".format(result[0]) )


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)