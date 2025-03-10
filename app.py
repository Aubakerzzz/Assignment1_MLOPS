from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Gather input features from the form
        weight = float(request.form['weight'])
        resolution = float(request.form['resolution'])
        ppi = float(request.form['ppi'])
        cpu_core = float(request.form['cpu_core'])
        cpu_freq = float(request.form['cpu_freq'])
        internal_mem = float(request.form['internal_mem'])
        ram = float(request.form['ram'])
        rear_cam = float(request.form['rear_cam'])
        front_cam = float(request.form['front_cam'])
        battery = float(request.form['battery'])
        thickness = float(request.form['thickness'])

        # Make prediction using the loaded model
        prediction = model.predict([[
            weight, resolution, ppi, cpu_core,
            cpu_freq, internal_mem, ram, rear_cam,
            front_cam, battery, thickness
        ]])

        # Render the prediction result template with the prediction
        return render_template('result.html', prediction=prediction[0])

    # Render the input form template
    return render_template('predict.html')


if __name__ == '__main__':
    # Run the Flask app
    app.run(port=8081, host="0.0.0.0")
