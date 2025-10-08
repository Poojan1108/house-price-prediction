import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load and prepare the model
print("Loading data and training model...")
data = pd.read_csv('USA_Housing.csv')

# Prepare features and target
X = data.drop(['Price', 'Address'], axis=1)
y = data['Price']

# Get column names for reference
feature_names = X.columns.tolist()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_scaled, y)

print(f"Model trained successfully!")
print(f"Model R² Score: {model.score(X_scaled, y):.4f}")
print(f"Features: {feature_names}")

# Prediction function
def predict_price(avg_area_income, avg_area_house_age, avg_area_num_rooms, 
                  avg_area_num_bedrooms, area_population):
    """
    Predict house price based on input features - All values in USD
    """
    try:
        # Create input array (all in USD)
        input_data = np.array([[avg_area_income, avg_area_house_age, 
                               avg_area_num_rooms, avg_area_num_bedrooms, 
                               area_population]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction (in USD)
        predicted_price_usd = model.predict(input_scaled)[0]
        
        # Format in USD
        price_display = f"${predicted_price_usd:,.2f}"
        
        return price_display
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface - Larger and More Descriptive
with gr.Blocks(title="House Price Prediction", css="""
    .gradio-container {font-size: 18px !important; max-width: 1400px !important;}
    .gr-button {font-size: 20px !important; padding: 15px !important; font-weight: bold !important;}
    .gr-box {padding: 20px !important;}
    label {font-size: 20px !important; font-weight: 600 !important;}
    .gr-text-input, .gr-number-input {font-size: 18px !important;}
    h1 {font-size: 48px !important; margin-bottom: 10px !important;}
    h2 {font-size: 32px !important;}
    h3 {font-size: 24px !important;}
    p {font-size: 18px !important;}
    .info {font-size: 16px !important; color: #666 !important;}
""") as demo:
    
    gr.Markdown("# 🏠 House Price Prediction System")
    gr.Markdown("## Enter property details to get an estimated price in US Dollars")
    gr.Markdown("---")
    
    # Detailed Instructions
    with gr.Accordion("📖 How to use this tool (Click to expand)", open=False):
        gr.Markdown("""
        ### Instructions:
        1. **Average Area Income**: Enter the average annual income (in $) of people living in the area
           - Example: $50,000 for middle-class area, $100,000 for premium area
        
        2. **House Age**: Select how old the property is (in years)
           - Newer houses (1-5 years) typically cost more
           - Older houses (10+ years) may cost less
        
        3. **Total Rooms**: Select total number of rooms (including living room, dining, kitchen, bedrooms)
           - Typical range: 5-10 rooms
        
        4. **Bedrooms**: Select number of bedrooms only
           - Typical: 2-4 bedrooms
        
        5. **Area Population**: Enter the population of the locality/neighborhood
           - Urban areas: 30,000 - 50,000+
           - Suburban areas: 15,000 - 30,000
        
        Click "Calculate Price" to get the estimated property price in US Dollars!
        """)
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Property Features")
            
            income = gr.Number(
                label="💰 Average Area Income ($/year)",
                value=68000,
                info="💡 Average annual income in the locality. Higher income areas have higher property prices."
            )
            
            house_age = gr.Slider(
                minimum=1,
                maximum=20,
                value=5.5,
                step=0.5,
                label="🏗️ House Age (years)",
                info="💡 How old is the property? Newer properties are generally more expensive."
            )
            
            num_rooms = gr.Slider(
                minimum=3,
                maximum=12,
                value=7.0,
                step=0.5,
                label="🚪 Total Rooms",
                info="💡 Total number of rooms (living room, dining, kitchen, bedrooms, etc.)"
            )
            
            num_bedrooms = gr.Slider(
                minimum=1,
                maximum=8,
                value=4.0,
                step=1,
                label="🛏️ Number of Bedrooms",
                info="💡 How many bedrooms in the house? More bedrooms = higher price."
            )
            
            population = gr.Number(
                label="👥 Area Population",
                value=36000,
                info="💡 Population of the locality. High-density areas may have different pricing."
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        clear_btn = gr.Button("🔄 Reset All Fields", size="lg", scale=1)
        predict_btn = gr.Button("💰 Calculate Property Price", variant="primary", size="lg", scale=3)
    
    gr.Markdown("### ")
    
    output = gr.Textbox(
        label="🎯 ESTIMATED PROPERTY PRICE (USD)",
        interactive=False,
        scale=2,
        show_label=True,
        container=True,
        elem_id="output_box"
    )
    
    gr.Markdown("---")
    gr.Markdown("### 📋 Quick Examples - Click to Try")
    gr.Markdown("*These are pre-filled examples for different types of properties*")
    
    gr.Examples(
        examples=[
            [79545, 5.68, 7.0, 4, 23087],     # Premium locality
            [68804, 5.60, 7.0, 4, 32388],     # Mid-range area
            [61481, 7.25, 6.5, 4, 36882],     # Older property
            [95000, 3.0, 8.5, 5, 45000],      # Luxury new property
            [55000, 10.0, 6.0, 3, 25000],     # Budget property
        ],
        inputs=[income, house_age, num_rooms, num_bedrooms, population],
        label=None,
        examples_per_page=5
    )
    
    gr.Markdown("---")
    
    # Connect the prediction function
    predict_btn.click(
        fn=predict_price,
        inputs=[income, house_age, num_rooms, num_bedrooms, population],
        outputs=output
    )
    
    clear_btn.click(
        fn=lambda: [68000, 5.5, 7.0, 4.0, 36000, ""],
        inputs=[],
        outputs=[income, house_age, num_rooms, num_bedrooms, population, output]
    )
    
    gr.Markdown("---")
    gr.Markdown("### 📈 About This Model")
    gr.Markdown("""
    - **🎯 Model Accuracy**: 91.8% (R² Score: 0.9180)
    - **🤖 Algorithm**: Linear Regression with Feature Scaling
    - **📊 Training Data**: USA Housing Dataset (5000+ properties)
    - **💵 Currency**: US Dollars (USD)
    - **✨ Technology**: Python, scikit-learn, Gradio
    
    *This is a machine learning project demonstrating property price prediction based on key features.*
    """)
    
    gr.Markdown("---")
    gr.Markdown("**💡 Powered by Machine Learning | Created for Faculty Presentation**")

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting House Price Prediction UI...")
    print("="*50 + "\n")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
