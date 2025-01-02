import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import re


###To run code : streamlit run main.py
## Dictionary
disease_info_combined = {
    'Apple___Apple_scab': {
        'info': (
            "Apple scab is a fungal disease affecting apple trees, caused by Venturia inaequalis. "
            "Symptoms include olive-green to black lesions on leaves and premature leaf drop. "
            "Management involves using resistant varieties, applying fungicides, and removing fallen leaves."
        )
    },
    'Apple___Black_rot': {
        'info': (
            "Black rot is a fungal disease affecting apples, caused by Guignardia bidwellii. "
            "It presents with black, sunken lesions on fruit, leaf yellowing, and premature drop. "
            "Control measures include practicing good sanitation, applying fungicides, and avoiding overhead watering."
        )
    },
    'Apple___Cedar_apple_rust': {
        'info': (
            "Cedar apple rust is a fungal disease affecting apples, caused by Gymnosporangium juniperi-virginianae. "
            "Symptoms include yellow-orange spots on leaves and galls on cedar trees. "
            "Management strategies include planting resistant varieties, applying fungicides, and controlling nearby cedar trees."
        )
    },
    'Apple___healthy': {
        'info': (
            "Healthy apple trees show no signs of disease. "
            "They exhibit vibrant green leaves and good fruit production. "
            "Regular maintenance and proper nutrient management are essential for maintaining health."
        )
    },
    'Corn___Common_rust': {
        'info': (
            "Common rust is a fungal disease affecting corn, caused by Puccinia sorghi. "
            "It is characterized by reddish-brown pustules on leaves and stunted growth. "
            "Preventive measures include using resistant hybrids, applying fungicides, and practicing crop rotation."
        )
    },
    'Corn___healthy': {
        'info': (
            "Healthy corn plants show no signs of disease. "
            "They have strong, green plants with good ear development. "
            "Proper care and nutrient management are crucial for maintaining plant health."
        )
    },
    'Potato___Early_blight': {
        'info': (
            "Early blight is a fungal disease affecting potatoes, caused by Alternaria solani. "
            "It manifests as dark spots on leaves, yellowing, and leaf drop. "
            "Management includes using resistant varieties, applying fungicides, and practicing crop rotation."
        )
    },
    'Potato___Late_blight': {
        'info': (
            "Late blight is a serious disease affecting potatoes, caused by Phytophthora infestans. "
            "Symptoms include water-soaked spots on leaves and rapid decay of tubers. "
            "Control measures involve using resistant varieties, applying fungicides, and improving drainage."
        )
    },
    'Potato___healthy': {
        'info': (
            "Healthy potato plants show no signs of disease. "
            "They demonstrate strong growth and good tuber development. "
            "Regular care and proper fertilization are necessary for maintaining health."
        )
    },
    'Squash___Powdery_mildew': {
        'info': (
            "Powdery mildew is a fungal disease affecting squash, caused by Podosphaera xanthii. "
            "It is characterized by white powdery spots on leaves, leaf distortion, and reduced fruit quality. "
            "Management includes improving air circulation, applying fungicides, and practicing crop rotation."
        )
    },
    'Squash___healthy': {
        'info': (
            "Healthy squash plants show no signs of disease. "
            "They exhibit green leaves and healthy fruit production. "
            "Proper irrigation and fertilization are essential for plant health."
        )
    },
    'Tomato___Bacterial_spot': {
        'info': (
            "Bacterial spot affects tomato plants, caused by Xanthomonas campestris. "
            "Symptoms include dark, water-soaked spots on leaves, leaf drop, and fruit lesions. "
            "Management includes using resistant varieties, applying bactericides, and practicing crop rotation."
        )
    },
    'Tomato___Early_blight': {
        'info': (
            "Early blight is a fungal disease affecting tomatoes, caused by Alternaria solani. "
            "It manifests as dark, concentric spots on leaves, yellowing, and premature leaf drop. "
            "Control measures involve using fungicides and resistant varieties, along with crop rotation."
        )
    },
    'Tomato___Late_blight': {
        'info': (
            "Late blight is a serious disease affecting tomatoes, caused by Phytophthora infestans. "
            "Symptoms include water-soaked spots on leaves, rapid decay, and the potential to destroy entire crops. "
            "Management includes using fungicides and resistant varieties, as well as ensuring good air circulation."
        )
    },
    'Tomato___healthy': {
        'info': (
            "Healthy tomato plants show no signs of disease. "
            "They feature vibrant green leaves and good fruit development. "
            "Regular care and proper irrigation are crucial for maintaining health."
        )
    }
}


#TensorFlow model prediction function
def model_prediction(test_image):
    model = load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    prediction=model.predict(input_arr)
    result_index=np.argmax(prediction)
    return result_index

def format_disease_name(raw_prediction):
    cleaned_name = re.sub(r'\(.*?\)', '', raw_prediction)
    
    cleaned_name = cleaned_name.replace("_", " ").strip()
    
    formatted_name = cleaned_name.title()
    
    return formatted_name

# Sidebar 
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])
      

# Home Page
if app_mode == "Home":
    st.markdown(
        """
        <style>
            body {
                background-color: #000000; /* Black background */
                color: #FFFFFF; /* White text for contrast */
            }
            .main-header {
                font-size: 48px;
                font-weight: bold;
                text-align: center;
                margin-top: 20px;
                color: #A3D8D3; /* Light teal for header */
            }
            .sub-header {
                font-size: 22px;
                text-align: center;
                color: #B0BEC5; /* Light gray for sub-header */
                margin-bottom: 40px;
            }
            .content-block {
                background-color: #1E1E1E; /* Dark gray for content blocks */
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 2px 5px rgba(255, 255, 255, 0.1); /* Subtle shadow for depth */
                color: #FFFFFF; /* White text */
            }
            .footer {
                text-align: center;
                margin-top: 30px;
                padding: 10px;
                background-color: #A3D8D3; /* Light teal footer */
                color: #000000; /* Black text for footer */
                font-size: 14px;
            }
            .contact-section {
                background-color: #263238; /* Darker gray for contact section */
                padding: 30px;
                border-radius: 10px;
                color: #FFFFFF; /* White text */
            }
            .contact-header {
                text-align: center;
                font-size: 24px;
                color: #A3D8D3; /* Light teal for contact header */
                margin-bottom: 20px;
            }
            a {
                color: #76FF03; /* Bright green for links */
                text-decoration: none; /* Remove underline from links */
            }
            a:hover {
                text-decoration: underline; /* Underline on hover */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="main-header">PLANT DISEASE RECOGNITION SYSTEM</h1>', unsafe_allow_html=True)

    image_path = "home_page.jpg"
    st.image(image_path, use_column_width=True)

    st.markdown('<h2 class="sub-header">Real-Time Detection and Diagnosis for Healthy Crops</h2>', unsafe_allow_html=True)

    # Overview Section
    st.markdown(
        """
        <div class="content-block">
            <h3>🌱 Overview</h3>
            Welcome to our **Plant Disease Recognition System**, a revolutionary tool designed for farmers, gardeners, and agricultural enthusiasts. Utilizing cutting-edge machine learning algorithms, our application provides real-time diagnosis of plant diseases by analyzing uploaded images. With our system, you can quickly identify plant ailments, allowing for timely interventions that can protect your crops and enhance your agricultural yield.
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Features Section
    st.markdown(
        """
        <div class="content-block">
            <h3>🌟 Key Features</h3>
            <ul>
                <li><strong>Real-Time Diagnosis:</strong> Upload plant images and receive instant feedback on potential diseases, ensuring swift action to mitigate crop loss.</li>
                <li><strong>Comprehensive Disease Library:</strong> Access an extensive database detailing various plant diseases, their symptoms, and effective treatment methods, helping you make informed decisions.</li>
                <li><strong>User-Friendly Interface:</strong> Our intuitive design allows users of all skill levels to navigate the app easily, whether on a desktop or mobile device.</li>
                <li><strong>Plant Care Insights:</strong> After diagnosis, receive tailored recommendations for plant care, including preventive measures and specific treatments to enhance growth.</li>
                <li><strong>Expert Resources:</strong> Dive into educational content about best practices for plant health, pest management, and organic farming techniques.</li>
            </ul>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # How It Works Section
    st.markdown(
        """
        <div class="content-block">
            <h3>🛠️ How It Works</h3>
            <ol>
                <li><strong>Upload an Image:</strong> Simply capture a high-quality photo of the affected plant and upload it to our web app.</li>
                <li><strong>Image Analysis:</strong> Our advanced machine learning model analyzes the uploaded image for any visible signs of disease, including discoloration, wilting, and lesions.</li>
                <li><strong>Get Diagnosis:</strong> Within seconds, you’ll receive a detailed diagnosis of the potential disease, including an explanation of the symptoms observed and recommended treatments.</li>
                <li><strong>Follow Up:</strong> You can save your diagnoses for future reference and track the health of your plants over time, receiving alerts for necessary care actions.</li>
            </ol>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Contact Us Section
    # Contact Us Section
    st.markdown(
        """
        <div class="contact-section">
            <h2 class="contact-header">📞 Contact Us</h2>
            <p>If you have any questions, need support, or want to provide feedback, feel free to reach out to us:</p>
            <ul>
                <li><strong>Email:</strong>nikhilrghvsingh@gmail.com</li>
                <li><strong>LinkedIn:</strong><a href="https://www.linkedin.com/in/nikhilsingh0108/" target="_blank">Nikhil Singh</a></li>
                </ul>
        </div>
        """, 
        unsafe_allow_html=True
    )


    # Footer Section
    st.markdown(
        """
        <div class="footer">
            © 2024 Plant Disease Recognition System. All rights reserved.
        </div>
        """, 
        unsafe_allow_html=True
    )

    
#About page
elif(app_mode =="About"):
    st.header("About")
    st.markdown("""## About Dataset

This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on [this GitHub repository](https://github.com/your-repo-link). 

The dataset consists of approximately **87,000 RGB images** of healthy and diseased crop leaves, categorized into **38 different classes**. The total dataset is divided into an **80/20 ratio** for training and validation sets, preserving the directory structure. 

Additionally, a new directory containing **33 test images** has been created later for prediction purposes.
### Content
    1. Train(70295 images)
    2. Valid(17572 images)
    3. Test(33 images)                        
 """)






# Prediction Page
elif app_mode  == "Disease Recognition":
    st.header("🌱 Plant Disease Recognition")

    # Create a sidebar for instructions or information
    st.sidebar.header("Instructions")
    st.sidebar.write("Upload an image of a plant leaf to predict if it's healthy or diseased.")

    test_image = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

    # Create a layout with two columns
    col1, col2 = st.columns([2, 1])  # Adjust column width as needed

    with col1:
        if test_image is not None:
            st.image(test_image, use_column_width=True)


   # Initialize session state variables to store the prediction and raw_prediction
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None

    if 'raw_prediction' not in st.session_state:
        st.session_state.raw_prediction = None

    # Assuming you have the model and test_image defined earlier in your app
    with col2:
        # Predict Button
        if test_image is not None:
            # Predict button to trigger prediction
            if st.button("🔍 Predict"):
                st.write("### Our Prediction")
                result_index = model_prediction(test_image)

                # Define class names
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]

                raw_prediction = class_name[result_index]
                # Format the predicted disease name
                predicted_disease = format_disease_name(raw_prediction)

                # Store the prediction and raw_prediction in session state
                st.session_state.prediction = predicted_disease
                st.session_state.raw_prediction = raw_prediction

                # Display the formatted prediction with a success message
                st.success(f"Model predicts: **{predicted_disease}**")

        # Check if a prediction has been made and test_image is still available
        if st.session_state.prediction and test_image is not None:
            # Show the 'Know More' button after the prediction
            if st.button(f"Know More About {st.session_state.prediction}"):
                # Retrieve and display the disease information
                disease_info = disease_info_combined.get(st.session_state.raw_prediction, {}).get('info', 'No information available.')
                st.write(disease_info)

        # If the user clears the image, reset the session state
        if test_image is None:
            st.session_state.prediction = None
            st.session_state.raw_prediction = None
st.markdown(
    """
    ---
    ### About the Model
    This model has been trained on a dataset of plant diseases and can help identify various diseases based on leaf images. 
    For accurate results, ensure the uploaded image is clear and well-lit.
    """
)

