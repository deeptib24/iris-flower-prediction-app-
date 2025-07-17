# 🌸 Iris Flower Classification Web App  

An interactive web app to classify iris flowers using a trained **Random Forest Classifier**, built with **Streamlit** and **scikit-learn**.  
Predicts the flower species based on sepal and petal measurements.  

🚀 **[Live Demo](https://irisappdeploy.streamlit.app/)** 

---

## 📋 Features  
- 🌟 **User Input Sliders** — Provide custom measurements via sliders  
- 🔮 **Real-time Prediction** — Predict species based on user input  
- 📊 **Prediction Probability Bar Chart** — See confidence scores for each class  
- 📂 **Show Dataset Option** — View sample data from the Iris dataset  
- 🎨 **Clean UI with Streamlit** — Fast and interactive frontend  

---

## 🛠️ Built With  
- 🐍 Python  
- [Streamlit](https://streamlit.io/)  
- [scikit-learn](https://scikit-learn.org/)  
- [pandas](https://pandas.pydata.org/)  

---

## 💻 How to Run Locally  

```bash
# Clone the repo
git clone https://github.com/your-username/iris-flower-prediction-app.git
cd iris-flower-prediction-app

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run irisapp.py
