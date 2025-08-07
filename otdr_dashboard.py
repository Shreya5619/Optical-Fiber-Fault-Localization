import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import pickle
from datetime import datetime
import io
import base64
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="OTDR Fiber Fault Detection & Localization",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    /* Update sidebar background */
    .css-1d391kg {
        background-color: #2d2d2d;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-container {
        border: 2px solid #444444;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #2d2d2d;
    }
    .success-box {
        background-color: #1b5e20;
        border: 1px solid #4caf50;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    .error-box {
        background-color: #b71c1c;
        border: 1px solid #f44336;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    .info-box {
        background-color: #0d47a1;
        border: 1px solid #2196f3;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Fault class mapping based on your dataset
FAULT_CLASSES = {
    0: "Normal",
    1: "Fiber Tapping", 
    2: "Bad Splice",
    3: "Bending Event",
    4: "Dirty Connector",
    5: "Fiber Cut",
    6: "PC Connector",
    7: "Reflector"
}

# Initialize session state
if 'binary_prediction' not in st.session_state:
    st.session_state.binary_prediction = None
if 'detailed_predictions' not in st.session_state:
    st.session_state.detailed_predictions = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'otdr_trace' not in st.session_state:
    st.session_state.otdr_trace = None

# Helper functions
def load_model_safe(model_file):
    """Safely load a model file"""
    try:
        if hasattr(model_file, 'name') and model_file.name.endswith('.h5'):
            # For Streamlit, need to read bytes and save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                tmp.write(model_file.read())
                tmp.flush()
                return tf.keras.models.load_model(tmp.name)
        else:
            try:
                return joblib.load(model_file)
            except:
                try:
                    return pickle.load(model_file)
                except:
                    return None
    except Exception as e:
        return None

def extract_otdr_features(data_row):
    """Extract OTDR trace features (P1-P30) from data row"""
    if isinstance(data_row, pd.Series):
        # Extract P1 to P30 columns
        trace_cols = [f'P{i}' for i in range(1, 31)]
        return data_row[trace_cols].values
    return None

def preprocess_for_binary(data_row):
    """Preprocess data for binary classification"""
    features = []
    features.append(data_row['SNR'])  # Add SNR
    
    # Add OTDR trace points P1-P30
    for i in range(1, 31):
        features.append(data_row[f'P{i}'])
    
    return np.array(features).reshape(1, -1)

def simulate_model_prediction(model_file, input_data, prediction_type):
    """Simulate model prediction or use actual model if loaded"""
    if model_file is not None:
        try:
            model = load_model_safe(model_file)
            if model is not None:
                # Check if it's a Keras model
                if 'keras' in str(type(model)).lower():
                    pred = model.predict(input_data)
                    # Binary classification
                    if prediction_type == 'binary':
                        if pred.shape[-1] == 1:
                            pred_label = int(pred[0][0] > 0.5)
                            confidence = float(pred[0][0]) if pred_label == 1 else 1 - float(pred[0][0])
                        else:
                            pred_label = int(np.argmax(pred[0]))
                            confidence = float(np.max(pred[0]))
                        return {'prediction': pred_label, 'confidence': confidence}
                    # Class prediction (multi-class)
                    elif prediction_type == 'class':
                        pred_label = int(np.argmax(pred[0]))
                        return {'prediction': pred_label}
                    # Position, reflectance, loss (regression)
                    elif prediction_type in ['position', 'reflectance', 'loss']:
                        prediction_value = float(pred[0][0])
                        if prediction_type == 'position':
                            # Post-process for position prediction
                            position = np.round(prediction_value / 0.01) * 0.01
                            position = np.clip(position, 0, 0.30)
                            return {'prediction': position}
                        else:
                            return {'prediction': prediction_value}
                else:
                    # scikit-learn models
                    if prediction_type == 'binary':
                        pred = model.predict(input_data)[0]
                        prob = model.predict_proba(input_data)[0]
                        confidence = max(prob)
                        return {'prediction': bool(pred), 'confidence': confidence}
                    elif prediction_type == 'class':
                        pred = model.predict(input_data)[0]
                        return {'prediction': int(pred)}
                    elif prediction_type in ['position', 'reflectance', 'loss']:
                        pred = model.predict(input_data)[0]
                        if prediction_type == 'position':
                            # Post-process for position prediction
                            position = np.round(float(pred) / 0.01) * 0.01
                            position = np.clip(position, 0, 0.30)
                            return {'prediction': position}
                        else:
                            return {'prediction': float(pred)}
        except Exception as e:
            pass
    # Fallback: simulated output
    if prediction_type == 'binary':
        has_fault = np.random.choice([True, False], p=[0.7, 0.3])
        confidence = np.random.uniform(0.85, 0.99)
        return {'prediction': has_fault, 'confidence': confidence}
    elif prediction_type == 'class':
        return {'prediction': np.random.choice(list(range(8)))}
    elif prediction_type == 'position':
        return {'prediction': np.random.uniform(0.05, 0.95)}
    elif prediction_type == 'reflectance':
        return {'prediction': np.random.uniform(-0.5, 0.5)}
    elif prediction_type == 'loss':
        return {'prediction': np.random.uniform(-0.5, 0.5)}

# Header
st.markdown('<h1 class="main-header">🔍 OTDR-Based Fiber Fault Detection & Localization</h1>', unsafe_allow_html=True)

# Sidebar for model loading and configuration
st.sidebar.header("🔧 Model Configuration")

# Model file uploaders
st.sidebar.subheader("Load Pre-trained Models")
binary_model_file = st.sidebar.file_uploader("Binary Classification Model", type=['pkl', 'joblib', 'h5'], key="binary")
class_model_file = st.sidebar.file_uploader("Fault Class Detection Model", type=['pkl', 'joblib', 'h5'], key="class")
position_model_file = st.sidebar.file_uploader("Position Localization Model", type=['pkl', 'joblib'], key="position")
reflectance_model_file = st.sidebar.file_uploader("Reflectance Analysis Model", type=['pkl', 'joblib'], key="reflectance")
loss_model_file = st.sidebar.file_uploader("Loss Analysis Model", type=['pkl', 'joblib'], key="loss")
scaler_file = st.sidebar.file_uploader("Scaler for Classification Model", type=['pkl', 'joblib'], key="scaler")
# Model status indicators
st.sidebar.subheader("📊 Model Status")
models_loaded = {
    "Binary Classification": binary_model_file is not None,
    "Fault Class Detection": class_model_file is not None,
    "Position Localization": position_model_file is not None,
    "Reflectance Analysis": reflectance_model_file is not None,
    "Loss Analysis": loss_model_file is not None
}

for model_name, is_loaded in models_loaded.items():
    if is_loaded:
        st.sidebar.success(f"✅ {model_name}")
    else:
        st.sidebar.error(f"❌ {model_name}")

# Main dashboard layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📤 OTDR Data Input")
    
    # Input method selection
    input_method = st.radio("Select Input Method:", ["Upload OTDR Dataset", "Single OTDR Sample", "Manual OTDR Input"])
    
    if input_method == "Upload OTDR Dataset":
        uploaded_file = st.file_uploader("Upload OTDR_data.csv or similar", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ OTDR dataset uploaded successfully!")
                
                # Display dataset info
                st.info(f"Dataset shape: {df.shape}")
                st.dataframe(df.head())
                
                # Sample selection
                sample_idx = st.selectbox("Select sample for analysis:", range(len(df)))
                selected_sample = df.iloc[sample_idx]
                
                st.session_state.input_data = selected_sample
                st.session_state.otdr_trace = extract_otdr_features(selected_sample)
                
                # Display selected sample info
                st.subheader("Selected Sample Info")
                st.write(f"**SNR:** {selected_sample['SNR']:.3f}")
                if 'Class' in selected_sample:
                    actual_class = int(selected_sample['Class'])
                    st.write(f"**Actual Class:** {actual_class} ({FAULT_CLASSES.get(actual_class, 'Unknown')})")
                if 'Position' in selected_sample:
                    st.write(f"**Actual Position:** {selected_sample['Position']:.3f}")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    elif input_method == "Single OTDR Sample":
        st.subheader("Enter Single OTDR Sample")
        
        with st.form("single_sample_form"):
            snr = st.number_input("SNR", value=10.0, step=0.1)
            
            st.write("**OTDR Trace Points (P1-P30):**")
            trace_points = []
            
            # Create columns for trace points input
            cols = st.columns(3)
            for i in range(30):
                col_idx = i % 3
                with cols[col_idx]:
                    point_val = st.number_input(f"P{i+1}", value=0.5, step=0.01, key=f"p{i+1}")
                    trace_points.append(point_val)
            
            submitted = st.form_submit_button("Submit OTDR Sample")
            
            if submitted:
                # Create sample data
                sample_data = {'SNR': snr}
                for i, val in enumerate(trace_points):
                    sample_data[f'P{i+1}'] = val
                
                sample_series = pd.Series(sample_data)
                st.session_state.input_data = sample_series
                st.session_state.otdr_trace = np.array(trace_points)
                st.success("✅ OTDR sample submitted!")
    
    else:  # Manual OTDR Input
        if st.button("Generate Sample OTDR Data"):
            # Generate realistic OTDR trace
            np.random.seed(42)
            
            # Simulate OTDR trace with possible fault
            trace_length = 30
            base_trace = np.linspace(1.0, 0.1, trace_length)  # Decreasing power
            noise = np.random.normal(0, 0.05, trace_length)
            
            # Add fault signature randomly
            fault_pos = np.random.randint(5, 25)
            fault_type = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
            
            if fault_type > 0:  # Add fault
                if fault_type == 5:  # Fiber cut - sharp drop
                    base_trace[fault_pos:] *= 0.1
                elif fault_type in [2, 4]:  # Bad splice, dirty connector - loss
                    base_trace[fault_pos:] *= 0.7
                elif fault_type == 7:  # Reflector - spike
                    base_trace[fault_pos] += 0.3
            
            otdr_trace = base_trace + noise
            otdr_trace = np.clip(otdr_trace, 0, 1)  # Normalize
            
            # Create sample
            sample_data = {'SNR': np.random.uniform(8, 15)}
            for i, val in enumerate(otdr_trace):
                sample_data[f'P{i+1}'] = val
            
            sample_series = pd.Series(sample_data)
            st.session_state.input_data = sample_series
            st.session_state.otdr_trace = otdr_trace
            st.success("✅ Sample OTDR data generated!")

# Display OTDR trace visualization
if st.session_state.otdr_trace is not None:
    st.subheader("📊 OTDR Trace Visualization")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, 31)),
        y=st.session_state.otdr_trace,
        mode='lines+markers',
        name='OTDR Trace',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="OTDR Trace (P1-P30)",
        xaxis_title="Position Index",
        yaxis_title="Normalized Power",
        height=300,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("🔄 ML Processing Pipeline")
    
    if st.session_state.input_data is not None:
        # Step 1: Binary Classification
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.subheader("Step 1: Binary Fault Detection")
        st.write("*Determines if there is any fault in the fiber (Normal vs Fault)*")
        
        if st.button("🔍 Run Binary Classification", key="binary_btn"):
            with st.spinner("Running binary classification..."):
                # Prepare input for binary model
                input_features = preprocess_for_binary(st.session_state.input_data)
                
                # Get prediction
                result = simulate_model_prediction(binary_model_file, input_features, 'binary')
                st.session_state.binary_prediction = result
        
        if st.session_state.binary_prediction is not None:
            result = st.session_state.binary_prediction
            
            if result['prediction']:
                st.markdown(f'<div class="error-box">🚨 <strong>FAULT DETECTED</strong><br>Confidence: {result["confidence"]:.3f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box">✅ <strong>NO FAULT DETECTED</strong><br>Confidence: {result["confidence"]:.3f}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 2: Detailed Analysis (only if fault detected)
        if st.button("🔬 Run Detailed Analysis", key="detailed_btn"):
            with st.spinner("Running detailed analysis..."):
                predictions = {}

                # --- CLASSIFICATION SECTION WITH SCALER SUPPORT ---
                # Extract SNR and OTDR trace from the selected sample
                snr_input = np.array([[st.session_state.input_data['SNR']]])
                otdr_trace_input = np.array([st.session_state.input_data[f'P{i}'] for i in range(1, 31)]).reshape(1, -1)

                # Combine features correctly
                combined_input = np.hstack((snr_input, otdr_trace_input))

                # Apply scaler if provided
                if scaler_file is not None:
                    try:
                        scaler = joblib.load(scaler_file)
                        normalized_input = scaler.transform(combined_input)
                        st.info("✅ Scaler applied to classification data.")
                    except Exception as e:
                        st.error(f"Error loading or applying scaler: {e}")
                        st.stop()
                else:
                    normalized_input = combined_input  # No scaling

                # Split normalized input for model
                snr_normalized = normalized_input[:, 0].reshape(-1, 1)
                otdr_trace_normalized = normalized_input[:, 1:]

                # Fault Class Detection (pass as list for Keras multi-input)
                class_result = simulate_model_prediction(class_model_file, [otdr_trace_normalized, snr_normalized], 'class')
                predictions['class'] = {
                    'value': int(class_result['prediction']),
                    'name': FAULT_CLASSES.get(int(class_result['prediction']), 'Unknown')
                }

                # --- CONTINUE WITH OTHER MODELS AS BEFORE ---
                # Prepare input for the other models (position, reflectance, loss)
                input_features = preprocess_for_binary(st.session_state.input_data)

                # Position Localization  
                pos_result = simulate_model_prediction(position_model_file, input_features, 'position')
                predictions['position'] = {
                    'value': float(pos_result['prediction']),
                    'distance_km': float(pos_result['prediction']) * 100  # Convert to km (assuming 100km max)
                }

                # Reflectance Analysis
                if reflectance_model_file is not None:
                    refl_result = simulate_model_prediction(reflectance_model_file, input_features, 'reflectance')
                    predictions['reflectance'] = {
                        'value': float(refl_result['prediction'])
                    }

                # Loss Analysis
                if loss_model_file is not None:
                    loss_result = simulate_model_prediction(loss_model_file, input_features, 'loss')
                    predictions['loss'] = {
                        'value': float(loss_result['prediction'])
                    }

                st.session_state.detailed_predictions = predictions

                # Display detailed results
                if st.session_state.detailed_predictions is not None:
                    preds = st.session_state.detailed_predictions
                    
                    # Create metrics display
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Fault Class",
                            f"{preds['class']['value']}: {preds['class']['name']}",
                            "Predicted"
                        )
                    
                    with col2:
                        st.metric(
                            "Position",
                            f"{preds['position']['value']:.3f}",
                            f"~{preds['position']['distance_km']:.1f} km"
                        )
                    
                    with col3:
                        if 'reflectance' in preds:
                            st.metric(
                                "Reflectance",
                                f"{preds['reflectance']['value']:.3f}",
                                "Normalized"
                            )
                        else:
                            st.metric("Reflectance", "N/A")
                    
                    with col4:
                        if 'loss' in preds:
                            st.metric(
                                "Loss",
                                f"{preds['loss']['value']:.3f}",
                                "Normalized"
                            )
                        else:
                            st.metric("Loss", "N/A")
                    
                    # Fault location visualization on OTDR trace
                    if st.session_state.otdr_trace is not None:
                        st.subheader("🎯 Fault Localization on OTDR Trace")
                        
                        fig = go.Figure()
                        
                        # Plot OTDR trace
                        fig.add_trace(go.Scatter(
                            x=list(range(1, 31)),
                            y=st.session_state.otdr_trace,
                            mode='lines+markers',
                            name='OTDR Trace',
                            line=dict(color='blue', width=2),
                            marker=dict(size=4)
                        ))
                        
                        # Mark predicted fault position
                        fault_idx = int(preds['position']['value'] * 30)  # Convert to trace index
                        if 0 <= fault_idx < 30:
                            fig.add_trace(go.Scatter(
                                x=[fault_idx + 1],
                                y=[st.session_state.otdr_trace[fault_idx]],
                                mode='markers',
                                name=f'Predicted Fault Location',
                                marker=dict(color='red', size=15, symbol='diamond')
                            ))
                        
                        # Add actual fault position if available
                        if 'Position' in st.session_state.input_data:
                            actual_pos = st.session_state.input_data['Position']
                            actual_idx = int(actual_pos * 30)
                            if 0 <= actual_idx < 30:
                                fig.add_trace(go.Scatter(
                                    x=[actual_idx + 1],
                                    y=[st.session_state.otdr_trace[actual_idx]],
                                    mode='markers',
                                    name=f'Actual Fault Location',
                                    marker=dict(color='green', size=15, symbol='star')
                                ))
                        
                        fig.update_layout(
                            title="OTDR Trace with Fault Localization",
                            xaxis_title="Position Index (P1-P30)",
                            yaxis_title="Normalized Power",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Results Analysis Section
if st.session_state.detailed_predictions is not None:
    st.header("📊 Analysis Results & Performance")
    
    tab1, tab2, tab3 = st.tabs(["📈 Predictions vs Actual", "📋 Detailed Report", "📥 Export Results"])
    
    with tab1:
        st.subheader("Model Predictions vs Actual Values")
        
        # Compare predictions with actual values (if available)
        if all(col in st.session_state.input_data.index for col in ['Class', 'Position', 'Reflectance', 'Loss']):
            comparison_data = []
            
            # Class comparison
            actual_class = int(st.session_state.input_data['Class'])
            pred_class = st.session_state.detailed_predictions['class']['value']
            comparison_data.append({
                'Metric': 'Fault Class',
                'Actual': f"{actual_class} ({FAULT_CLASSES.get(actual_class, 'Unknown')})",
                'Predicted': f"{pred_class} ({FAULT_CLASSES.get(pred_class, 'Unknown')})",
                'Match': '✅' if actual_class == pred_class else '❌'
            })
            
            # Position comparison
            actual_pos = st.session_state.input_data['Position']
            pred_pos = st.session_state.detailed_predictions['position']['value']
            pos_error = abs(actual_pos - pred_pos)
            comparison_data.append({
                'Metric': 'Position',
                'Actual': f"{actual_pos:.3f}",
                'Predicted': f"{pred_pos:.3f}",
                'Match': f"Error: {pos_error:.3f}"
            })
            
            # Reflectance comparison
            if 'Reflectance' in st.session_state.input_data and 'reflectance' in st.session_state.detailed_predictions:
                actual_refl = st.session_state.input_data['Reflectance']
                pred_refl = st.session_state.detailed_predictions['reflectance']['value']
                refl_error = abs(actual_refl - pred_refl)
                comparison_data.append({
                    'Metric': 'Reflectance',
                    'Actual': f"{actual_refl:.3f}",
                    'Predicted': f"{pred_refl:.3f}",
                    'Match': f"Error: {refl_error:.3f}"
                })
            
            # Loss comparison
            if 'Loss' in st.session_state.input_data and 'loss' in st.session_state.detailed_predictions:
                actual_loss = st.session_state.input_data['Loss']
                pred_loss = st.session_state.detailed_predictions['loss']['value']
                loss_error = abs(actual_loss - pred_loss)
                comparison_data.append({
                    'Metric': 'Loss',
                    'Actual': f"{actual_loss:.3f}",
                    'Predicted': f"{pred_loss:.3f}",
                    'Match': f"Error: {loss_error:.3f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("Actual values not available in the input data for comparison.")
    
    with tab2:
        st.subheader("📋 Comprehensive Analysis Report")
        
        # Generate detailed report
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        preds = st.session_state.detailed_predictions
        
        report = f"""
        # OTDR Fiber Fault Analysis Report
        
        **Analysis Timestamp:** {report_time}
        **Sample SNR:** {st.session_state.input_data['SNR']:.3f}
        
        ## Binary Classification Results
        - **Fault Detection:** {'FAULT DETECTED' if st.session_state.binary_prediction['prediction'] else 'NO FAULT'}
        - **Confidence:** {st.session_state.binary_prediction['confidence']:.3f}
        
        ## Detailed Fault Analysis
        
        ### Fault Classification
        - **Predicted Class:** {preds['class']['value']} ({preds['class']['name']})
        - **Class Description:** {FAULT_CLASSES.get(preds['class']['value'], 'Unknown fault type')}
        
        ### Fault Localization
        - **Position Index:** {preds['position']['value']:.3f}
        - **Estimated Distance:** {preds['position']['distance_km']:.1f} km (assuming 100km total span)
        - **Trace Point:** P{int(preds['position']['value'] * 30) + 1}
        
        ### Fault Characteristics
        - **Reflectance:** {f"{preds['reflectance']['value']:.3f} (normalized)" if 'reflectance' in preds else 'N/A'}
        - **Loss:** {f"{preds['loss']['value']:.3f} (normalized)" if 'loss' in preds else 'N/A'}
        
        ## OTDR Trace Analysis
        - **Trace Length:** 30 points (P1-P30)
        - **Signal Quality:** Based on SNR of {st.session_state.input_data['SNR']:.3f}
        
        ## Recommendations
        Based on the detected fault type ({preds['class']['name']}):
        """
        
        # Add specific recommendations based on fault type
        fault_type = preds['class']['value']
        if fault_type == 1:  # Fiber Tapping
            report += "\n- Immediate security inspection required\n- Check for unauthorized access points\n- Consider fiber encryption"
        elif fault_type == 2:  # Bad Splice
            report += "\n- Schedule splice re-work\n- Verify splice loss specifications\n- Consider fusion splice instead of mechanical"
        elif fault_type == 3:  # Bending Event
            report += "\n- Check cable routing and support\n- Verify minimum bend radius compliance\n- Inspect for physical stress points"
        elif fault_type == 4:  # Dirty Connector
            report += "\n- Clean connectors with appropriate cleaning tools\n- Inspect end faces under microscope\n- Re-test after cleaning"
        elif fault_type == 5:  # Fiber Cut
            report += "\n- CRITICAL: Complete fiber break detected\n- Emergency repair required\n- Activate backup routes if available"
        elif fault_type == 6:  # PC Connector
            report += "\n- Verify connector type and specifications\n- Check insertion loss\n- Consider APC connectors if applicable"
        elif fault_type == 7:  # Reflector
            report += "\n- Identify source of reflection\n- Check for unterminated fibers\n- Verify proper connector end face preparation"
        
        report += f"""
        
        ## Technical Notes
        - Analysis performed using ML models trained on OTDR trace data
        - Position accuracy depends on OTDR resolution and model training
        - Normalized values require denormalization for absolute measurements
        - Consider environmental factors during maintenance planning
        
        ---
        *Report generated by AI-Driven OTDR Fault Detection System*
        """
        
        st.markdown(report)
    
    with tab3:
        st.subheader("📥 Export Analysis Results")
        
        if st.button("Generate CSV Export"):
            # Create export data
            export_data = {
                'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'SNR': [st.session_state.input_data['SNR']],
                'Binary_Prediction': [st.session_state.binary_prediction['prediction']],
                'Binary_Confidence': [st.session_state.binary_prediction['confidence']],
                'Fault_Class': [st.session_state.detailed_predictions['class']['value']],
                'Fault_Name': [st.session_state.detailed_predictions['class']['name']],
                'Position': [st.session_state.detailed_predictions['position']['value']],
                'Reflectance': [st.session_state.detailed_predictions.get('reflectance', {}).get('value', 'N/A')],
                'Loss': [st.session_state.detailed_predictions.get('loss', {}).get('value', 'N/A')]
            }
            
            # Add OTDR trace points
            for i, val in enumerate(st.session_state.otdr_trace):
                export_data[f'P{i+1}'] = [val]
            
            export_df = pd.DataFrame(export_data)
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"otdr_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("🔬 **OTDR-Based Fiber Fault Detection System** | Built with Streamlit for ML Model Integration")
st.markdown("*Dataset Format: SNR + 30 OTDR Trace Points (P1-P30) + Fault Classification & Localization*")