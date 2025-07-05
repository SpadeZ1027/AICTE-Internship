# AICTE-Internship

♻️ E-Waste Generation Classification using AI/ML

📄 Project Overview
Electronic waste (e-waste) is a rapidly growing global concern due to its environmental and health impacts. Manual classification for recycling is labor-intensive, error-prone, and inefficient. This project establishes an AI-powered e-waste classification system using EfficientNetV2B0, enabling accurate, automated, and rapid identification of e-waste categories to streamline recycling and promote sustainable waste management.

🎯 Objectives
Automate the classification of e-waste items using computer vision.
Provide disposal guidance based on the classified category.
Recommend smart bin sorting for efficient waste management.
Simulate hazard detection for safety in disposal and handling.
Deploy an interactive interface using Gradio for real-time testing.

🛠️ Technologies Used
Python 3.11
TensorFlow & Keras
EfficientNetV2B0 (Transfer Learning)
Gradio for UI deployment
Matplotlib, Seaborn for visualization
NumPy, PIL for image processing

🧩 Project Workflow
Data Collection & Exploration: Load, visualize, and understand e-waste image datasets.
Data Preprocessing: Resize, normalize, and augment images for training.
Model Selection: Use EfficientNetV2B0 with transfer learning for classification.
Model Training & Tuning: Compile, train, and tune hyperparameters with callbacks.
Model Evaluation: Analyze accuracy, loss trends, and confusion matrix.
Feature Integration: Disposal map, sorting recommendations, hazard simulation.
Deployment: Gradio-based interactive interface for user testing.

🚀 How to Run Locally
1️⃣ Clone the repository:
git clone https://github.com/yourusername/e-waste-classification.git
cd e-waste-classification
2️⃣ Install dependencies:
pip install -r requirements.txt
3️⃣ Run the Gradio app:
python app.py
or run inside Jupyter Notebook to test cells individually.

📊 Screenshots
![Screenshot 2025-07-05 172530](https://github.com/user-attachments/assets/2c0a6efe-bd27-4964-a826-0db99eaf885e)
![Screenshot 2025-07-05 172618](https://github.com/user-attachments/assets/e73702e6-fb41-4c9a-a8d7-734405209ea6)

🧪 Future Improvements
Integrate real hazard detection models using advanced CV techniques.
Connect to hardware smart bins for automated sorting.
Expand to detect brand-level e-waste auditing for producer responsibility tracking.
Optimize models for mobile and edge deployment in collection centers.

🤝 Contribution
Contributions, issues, and feature requests are welcome. Please open an issue or submit a pull request to collaborate.

✨ Acknowledgements
Faculty guidance for domain understanding.
TensorFlow and Gradio for model deployment.
Public datasets supporting e-waste research.

📩 Contact
If you have any questions, feel free to contact:
Name: Srishti Kaur
Email: srishtikaur2703@gmail.com
